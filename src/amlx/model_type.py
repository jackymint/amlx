from __future__ import annotations

import json
from pathlib import Path

_VLM_MODEL_TYPES: frozenset[str] = frozenset({
    "paligemma", "paligemma2",
    "idefics", "idefics2", "idefics3",
    "llava", "llava_next", "llava_next_video", "llava_onevision",
    "qwen2_vl", "qwen2_5_vl",
    "pixtral",
    "mllama",
    "internvl_chat",
    "phi3_v", "phi4_multimodal",
    "gemma4",
    "smolvlm", "smolvlm2",
    "aria", "molmo",
    "florence2",
})


def is_vlm(model_id: str) -> bool:
    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        return False
    for local_only in (True, False):
        try:
            config_path = hf_hub_download(
                model_id, "config.json", local_files_only=local_only
            )
            cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
            return cfg.get("model_type", "") in _VLM_MODEL_TYPES
        except Exception:
            continue
    return False
