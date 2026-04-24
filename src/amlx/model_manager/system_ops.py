from __future__ import annotations

from .shared import *


class ModelManagerSystemOpsMixin:
    @staticmethod
    def _default_fine_tuner(args: Any, callback: Any) -> None:
        from mlx_lm import lora as mlx_lora  # type: ignore

        mlx_lora.run(args, training_callback=callback)

    def _default_downloader(self, model_id: str, target_dir: Path) -> Path:
        try:
            from huggingface_hub import snapshot_download
        except Exception as exc:
            raise RuntimeError(
                "huggingface_hub is required for downloads. Install with `pip install huggingface_hub`."
            ) from exc

        target_dir.parent.mkdir(parents=True, exist_ok=True)
        local_dir = self.models_dir / target_dir
        local_dir.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            resume_download=True,
        )
        return local_dir

    @staticmethod
    def _fuse_adapter_into_model(
        *,
        effective_model: str,
        adapter_path: Path,
        target_path: Path,
    ) -> None:
        if not adapter_path.exists() or not adapter_path.is_dir():
            raise ValueError(f"Adapter directory not found: {adapter_path}")

        try:
            from mlx_lm import fuse as mlx_fuse  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "mlx-lm is required for merge/save. Install with `pip install 'amlx[mlx]'`."
            ) from exc

        model, tokenizer, config = mlx_fuse.load(
            effective_model,
            adapter_path=str(adapter_path),
            return_config=True,
        )
        fused_linears = [
            (name, module.fuse(dequantize=False))
            for name, module in model.named_modules()
            if hasattr(module, "fuse")
        ]
        if fused_linears:
            model.update_modules(mlx_fuse.tree_unflatten(fused_linears))

        mlx_fuse.save(
            target_path,
            effective_model,
            model,
            tokenizer,
            config,
            donate_model=False,
        )

    @staticmethod
    def _dir_size_bytes(path: Path) -> int:
        if not path.exists():
            return 0
        total = 0
        try:
            for p in path.rglob("*"):
                if p.is_file():
                    total += p.stat().st_size
        except Exception:
            return total
        return total

    @staticmethod
    def _estimate_repo_size_bytes(model_id: str) -> int | None:
        try:
            from huggingface_hub import HfApi

            info = HfApi().model_info(model_id, files_metadata=True)
            siblings = getattr(info, "siblings", None) or []
            total = 0
            for sib in siblings:
                size = getattr(sib, "size", None)
                if isinstance(size, int) and size > 0:
                    total += size
            return total if total > 0 else None
        except Exception:
            return None

    def _default_search_provider(self, query: str, limit: int) -> list[dict[str, Any]]:
        try:
            from huggingface_hub import HfApi
        except Exception as exc:
            raise RuntimeError(
                "huggingface_hub is required for online search. Install with `pip install huggingface_hub`."
            ) from exc

        api = HfApi()
        models = api.list_models(
            author="mlx-community",
            search=query,
            sort="downloads",
            limit=limit,
        )
        out: list[dict[str, Any]] = []
        for model in models:
            model_id = str(getattr(model, "id", ""))
            if not model_id:
                continue
            disk_gb, _, _ = self._estimate_requirements(model_id)
            out.append(
                {
                    "id": model_id,
                    "label": model_id.split("/", 1)[-1],
                    "tags": "online, mlx-community",
                    "size": f"~{disk_gb:.1f} GB",
                }
            )
        return out
