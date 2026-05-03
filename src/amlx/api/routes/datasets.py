from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

from fastapi import FastAPI, HTTPException

from amlx.schemas import DatasetFetchRequest


def _hf_get(path: str, params: dict) -> dict:
    url = f"https://datasets-server.huggingface.co{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "amlx/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="ignore")
        raise RuntimeError(f"HTTP {exc.code}: {body[:300]}") from exc


def _hf_resolve_config_split(dataset_id: str, wanted_split: str) -> tuple[str, str]:
    """Return (config, split) using the /splits endpoint."""
    data = _hf_get("/splits", {"dataset": dataset_id})
    entries = data.get("splits", [])
    if not entries:
        return "default", wanted_split
    # prefer wanted split, fallback to first available
    for entry in entries:
        if entry.get("split") == wanted_split:
            return entry.get("config", "default"), wanted_split
    first = entries[0]
    return first.get("config", "default"), first.get("split", wanted_split)


def _hf_fetch_rows(dataset_id: str, split: str, limit: int) -> list[dict]:
    config, resolved_split = _hf_resolve_config_split(dataset_id, split)
    HF_PAGE = 100  # HF Datasets Server hard cap per request
    target = min(limit, 500)
    rows: list[dict] = []
    offset = 0
    while len(rows) < target:
        batch_size = min(HF_PAGE, target - len(rows))
        data = _hf_get("/rows", {
            "dataset": dataset_id,
            "config": config,
            "split": resolved_split,
            "offset": offset,
            "limit": batch_size,
        })
        batch = data.get("rows", [])
        rows.extend(r["row"] for r in batch if isinstance(r.get("row"), dict))
        if len(batch) < batch_size:
            break  # no more rows available
        offset += batch_size
    return rows


def _row_to_text(row: dict) -> str | None:
    # Prefer explicit text field
    if "text" in row and isinstance(row["text"], str) and row["text"].strip():
        return row["text"].strip()

    # Instruction-following format → format as conversation
    instruction = row.get("instruction") or row.get("prompt") or row.get("input") or ""
    output = row.get("output") or row.get("response") or row.get("answer") or row.get("completion") or ""
    if instruction and output:
        return json.dumps({"text": f"### Instruction:\n{instruction}\n\n### Response:\n{output}"}, ensure_ascii=False)

    if instruction:
        return instruction.strip()

    # Any long string field
    for v in row.values():
        if isinstance(v, str) and len(v) > 20:
            return v.strip()

    return None


def _hf_search_datasets(query: str, limit: int) -> list[dict]:
    params = urllib.parse.urlencode({
        "search": query,
        "limit": min(limit, 100),
        "full": "false",
        "sort": "downloads",
        "direction": -1,
    })
    url = f"https://huggingface.co/api/datasets?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "amlx/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="ignore")
        raise RuntimeError(f"HuggingFace API error {exc.code}: {body[:200]}") from exc
    if not isinstance(data, list):
        return []
    results = []
    for item in data:
        dataset_id = item.get("id") or item.get("modelId") or ""
        if not dataset_id:
            continue
        results.append({
            "id": dataset_id,
            "downloads": item.get("downloads", 0),
            "tags": (item.get("tags") or [])[:6],
            "description": str(item.get("description") or "").strip()[:120],
        })
    return results


def register_datasets_routes(app: FastAPI) -> None:
    @app.get("/v1/datasets/search")
    def datasets_search(q: str = "", limit: int = 100) -> dict[str, object]:
        q = q.strip()
        if not q:
            return {"ok": True, "results": []}
        try:
            results = _hf_search_datasets(q, limit)
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"search failed: {exc}") from exc
        return {"ok": True, "query": q, "results": results}

    @app.post("/v1/datasets/fetch")
    def datasets_fetch(req: DatasetFetchRequest) -> dict[str, object]:
        dataset_id = req.dataset_id.strip()
        if not dataset_id:
            raise HTTPException(status_code=422, detail="dataset_id is required")
        try:
            rows = _hf_fetch_rows(dataset_id, req.split, req.limit)
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"fetch failed: {exc}") from exc

        samples: list[str] = []
        for row in rows:
            text = _row_to_text(row)
            if text:
                samples.append(text)

        return {
            "ok": True,
            "dataset_id": dataset_id,
            "split": req.split,
            "count": len(samples),
            "samples": samples,
        }
