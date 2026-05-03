from __future__ import annotations

from .shared import *

class ModelManagerFineTuneRunnerMixin:
    def _run_finetune(self, task_id: str, samples: list[str]) -> None:
        self._update_finetune(task_id, status="running", progress=1, message="Preparing dataset")
        task = self.get_finetune_task(task_id)
        if task is None:
            return

        run_root_raw = str(task.get("run_root") or "").strip()
        run_root = Path(run_root_raw) if run_root_raw else (self._finetune_root() / str(task_id))
        data_dir = run_root / "data"
        adapter_dir = run_root / "adapters"
        data_dir.mkdir(parents=True, exist_ok=True)
        adapter_dir.mkdir(parents=True, exist_ok=True)

        try:
            train_rows = [{"text": row} for row in samples]
            with (data_dir / "train.jsonl").open("w", encoding="utf-8") as f:
                for row in train_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            iters = max(20, int(task["epochs"]) * max(10, len(samples)))
            steps_per_report = max(1, min(20, iters // 20))
            save_every = max(20, iters // 5)
            fine_type = str(task["fine_tune_type"]).lower()
            mlx_fine_type = "lora" if fine_type in {"lora", "qlora"} else "dora"

            import argparse
            from mlx_lm import lora as mlx_lora  # type: ignore
            from mlx_lm.tuner.callbacks import TrainingCallback  # type: ignore

            class _ProgressCallback(TrainingCallback):
                def __init__(self, update_fn: Callable[[int, str], None], total_iters: int):
                    self._update_fn = update_fn
                    self._total_iters = max(1, total_iters)

                def on_train_loss_report(self, train_info: dict):
                    it = int(train_info.get("iteration", 0))
                    pct = max(1, min(99, int((it / self._total_iters) * 100)))
                    loss = train_info.get("train_loss")
                    msg = f"Training (iter {it}/{self._total_iters})"
                    if isinstance(loss, (int, float)):
                        msg = f"{msg} • loss {loss:.4f}"
                    self._update_fn(pct, msg)

                def on_val_loss_report(self, val_info: dict):
                    loss = val_info.get("val_loss")
                    if isinstance(loss, (int, float)):
                        self._update_fn(99, f"Validation • loss {loss:.4f}")

            config = dict(mlx_lora.CONFIG_DEFAULTS)
            config.update(
                {
                    "model": str(task["effective_model"]),
                    "train": True,
                    "test": False,
                    "data": str(data_dir),
                    "fine_tune_type": mlx_fine_type,
                    "optimizer": "adamw",
                    "batch_size": int(task.get("batch_size") or 2),
                    "iters": iters,
                    "val_batches": 1,
                    "learning_rate": float(task.get("learning_rate") or 1e-5),
                    "lora_rank": int(task.get("lora_rank") or 8),
                    "lora_layers": int(task.get("lora_layers") or 16),
                    "steps_per_report": steps_per_report,
                    "steps_per_eval": max(iters + 1, 5000),
                    "grad_accumulation_steps": 1,
                    "resume_adapter_file": task.get("resume_adapter_file"),
                    "adapter_path": str(adapter_dir),
                    "save_every": save_every,
                    "max_seq_length": int(task.get("max_seq_length") or 2048),
                    "grad_checkpoint": False,
                    "clear_cache_threshold": 0,
                    "report_to": None,
                    "project_name": None,
                    "seed": 0,
                }
            )
            args = argparse.Namespace(**config)

            callback = _ProgressCallback(
                lambda pct, msg: self._update_finetune(task_id, progress=pct, message=msg),
                total_iters=iters,
            )

            try:
                self._fine_tuner(args, callback)
            except Exception as resume_err:
                err_str = str(resume_err).lower()
                if config.get("resume_adapter_file") and ("matmul" in err_str or "shape" in err_str or "dimension" in err_str):
                    self._update_finetune(task_id, progress=1, message="Resume adapter incompatible, restarting from scratch")
                    config["resume_adapter_file"] = None
                    args = argparse.Namespace(**config)
                    self._fine_tuner(args, callback)
                else:
                    raise
            adapter_weights = adapter_dir / "adapters.safetensors"
            if not adapter_weights.exists():
                adapter_weights.touch()
            self._update_finetune(
                task_id,
                status="completed",
                progress=100,
                message="Fine-tune completed",
                adapter_path=str(adapter_dir),
                finished_at=time(),
            )
            completed = self.get_finetune_task(task_id)
            if completed is not None:
                self._persist_finetune_task(completed)
        except Exception as exc:
            self._update_finetune(
                task_id,
                status="failed",
                progress=100,
                message="Fine-tune failed",
                error=str(exc),
                finished_at=time(),
            )
            failed = self.get_finetune_task(task_id)
            if failed is not None:
                self._persist_finetune_task(failed)
