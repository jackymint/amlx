from __future__ import annotations

from .shared import *

class ModelManagerCoreMixin:
    def __init__(
        self,
        *,
        models_dir: Path,
        downloader: Downloader | None = None,
        search_provider: SearchProvider | None = None,
        fine_tuner: FineTuner | None = None,
    ) -> None:
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._tasks: dict[str, DownloadTask] = {}
        self._finetune_tasks: dict[str, FineTuneTask] = {}
        self._quantize_tasks: dict[str, QuantizeTask] = {}
        self._download_procs: dict[str, subprocess.Popen[bytes]] = {}
        self._quantize_procs: dict[str, subprocess.Popen[bytes]] = {}
        self._lock = threading.Lock()
        self._downloader = downloader or self._default_downloader
        self._search_provider = search_provider or self._default_search_provider
        self._fine_tuner = fine_tuner or self._default_fine_tuner

    def system_profile(self) -> dict[str, Any]:
        machine = platform.machine().lower()
        system = platform.system().lower()
        ram_gb = round(self._total_ram_gb(), 1)
        free_disk_gb = round(shutil.disk_usage(self.models_dir).free / (1024**3), 1)
        apple_silicon = system == "darwin" and machine in {"arm64", "aarch64"}
        return {
            "system": system,
            "machine": machine,
            "apple_silicon": apple_silicon,
            "ram_gb": ram_gb,
            "free_disk_gb": free_disk_gb,
            "models_dir": str(self.models_dir),
        }

    def catalog(self, page: int = 1, per_page: int = 12) -> tuple[list[CatalogItem], int]:
        page = max(1, page)
        per_page = max(1, min(per_page, 100))
        profile = self.system_profile()
        enriched_all: list[CatalogItem] = []
        for item in CATALOG:
            enriched = dict(item)
            enriched["compatibility"] = self._compatibility(item, profile)
            enriched["capabilities"] = self._capabilities(enriched)
            enriched_all.append(enriched)
        total = len(enriched_all)
        start = (page - 1) * per_page
        end = start + per_page
        return enriched_all[start:end], total

    def search_online(self, query: str, page: int = 1, per_page: int = 12) -> tuple[list[CatalogItem], int]:
        page = max(1, page)
        per_page = max(1, min(per_page, 100))
        profile = self.system_profile()
        online_limit = max(50, min(200, per_page * 10))
        online = self._search_provider(query, online_limit)
        total = len(online)
        start = (page - 1) * per_page
        end = start + per_page
        online_page = online[start:end]
        items: list[CatalogItem] = []
        for item in online_page:
            disk_gb, min_ram_gb, rec_ram_gb = self._estimate_requirements(item.get("id", ""))
            enriched: CatalogItem = {
                "id": item.get("id", "unknown"),
                "label": item.get("label") or item.get("id", "unknown"),
                "size": item.get("size", "unknown"),
                "tags": item.get("tags", "online"),
                "disk_gb": disk_gb,
                "min_ram_gb": min_ram_gb,
                "rec_ram_gb": rec_ram_gb,
                "requires_apple_silicon": True,
                "source": "online",
            }
            enriched["compatibility"] = self._compatibility(enriched, profile)
            enriched["capabilities"] = self._capabilities(enriched)
            items.append(enriched)
        return items, total

    @staticmethod
    def _capabilities(item: CatalogItem) -> dict[str, bool]:
        ident = str(item.get("id", "")).lower()
        tags = str(item.get("tags", "")).lower()
        text = f"{ident} {tags}"

        vision_tokens = ("vlm", "vision", "multimodal", "llava", "qwen-vl", "gemma-vision")
        thinking_tokens = ("reasoning", "r1", "think", "thinking", "o1", "o3", "deepseek-r1")
        tool_tokens = ("instruct", "chat", "assistant", "function", "tool", "agent", "coder")
        coding_tokens = ("coder", "coding", "code", "dev", "programming")
        embedding_tokens = ("embedding", "embed")
        rerank_tokens = ("rerank", "reranker")
        audio_tokens = ("audio", "asr", "whisper", "speech", "tts", "stt", "voice")
        speech_tokens = ("speech", "tts", "stt", "voice", "asr", "whisper")

        vision = any(token in text for token in vision_tokens) or is_vlm(ident)
        thinking = any(token in text for token in thinking_tokens)
        tool = any(token in text for token in tool_tokens) and not vision
        coding = any(token in text for token in coding_tokens) and not vision
        embedding = any(token in text for token in embedding_tokens)
        rerank = any(token in text for token in rerank_tokens)
        audio = any(token in text for token in audio_tokens)
        speech = any(token in text for token in speech_tokens)

        return {
            "tool": bool(tool),
            "vision": bool(vision),
            "thinking": bool(thinking),
            "coding": bool(coding),
            "embedding": bool(embedding),
            "rerank": bool(rerank),
            "audio": bool(audio),
            "speech": bool(speech),
        }
