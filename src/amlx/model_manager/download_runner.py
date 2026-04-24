from __future__ import annotations

from .shared import *

class ModelManagerDownloadRunnerMixin:
    def _run_download(self, task_id: str) -> None:
        self._update(task_id, status="downloading", progress=0, message="Preparing download")

        task = self.get_task(task_id)
        if task is None:
            return

        model_id = str(task["model_id"])
        target_rel = self._model_path(model_id)
        target_dir = self.models_dir / target_rel
        log_path = self._download_log_path(task_id)
        last_pct_logged = -1

        def log(line: str) -> None:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

        try:
            log(f"[start] downloading {model_id}")
            marker = target_dir / "amlx_model.json"
            if marker.exists():
                self._update(
                    task_id,
                    status="completed",
                    progress=100,
                    message="Already installed",
                    local_path=str(target_dir),
                    finished_at=time(),
                )
                log("[done] already installed")
                return

            log("[info] fetching model artifacts")
            self._update(task_id, progress=1, message="Fetching model artifacts")
            expected_bytes = self._estimate_repo_size_bytes(model_id)
            start_bytes = self._dir_size_bytes(target_dir)
            if expected_bytes:
                log(f"[info] expected size ~{expected_bytes // (1024**3) or round(expected_bytes / (1024**2))} {'GB' if expected_bytes >= 1024**3 else 'MB'}")

            if self._downloader is self._default_downloader:
                from huggingface_hub import snapshot_download
                target_dir.parent.mkdir(parents=True, exist_ok=True)
                local_dir = self.models_dir / target_rel
                local_dir.mkdir(parents=True, exist_ok=True)

                result: dict[str, Path] = {}
                error_holder: list[Exception] = []
                done = threading.Event()
                cancel_event = threading.Event()

                def run_default_download() -> None:
                    try:
                        snapshot_download(repo_id=model_id, local_dir=str(local_dir), resume_download=True)
                        result["path"] = local_dir
                    except Exception as exc:
                        error_holder.append(exc)
                    finally:
                        done.set()

                worker = threading.Thread(target=run_default_download, daemon=True)
                worker.start()

                nonlocal_pct = [1]
                while not done.wait(timeout=1.0):
                    with self._lock:
                        if self._tasks.get(task_id) and self._tasks[task_id].cancelled:
                            cancel_event.set()
                            print(f"[amlx] download cancelled: {model_id}", flush=True)
                            log("[cancelled] download cancelled by user")
                            return
                    if expected_bytes and expected_bytes > 0:
                        current_bytes = max(0, self._dir_size_bytes(target_dir) - start_bytes)
                        ratio = min(1.0, current_bytes / expected_bytes)
                        rolling = max(nonlocal_pct[0], min(95, 1 + int(ratio * 94)))
                    else:
                        rolling = min(90, nonlocal_pct[0] + 1)
                    nonlocal_pct[0] = rolling
                    self._update(task_id, progress=rolling, message="Fetching model artifacts")
                    milestone = (rolling // 10) * 10
                    if milestone > last_pct_logged and milestone > 0:
                        log(f"[progress] {rolling}%")
                        last_pct_logged = milestone

                worker.join(timeout=0.1)
                if error_holder:
                    raise error_holder[0]
                path = result["path"]
            else:
                result: dict[str, Path] = {}
                error_holder: list[Exception] = []
                done = threading.Event()

                def run_download() -> None:
                    try:
                        result["path"] = self._downloader(model_id, target_rel)
                    except Exception as exc:
                        error_holder.append(exc)
                    finally:
                        done.set()

                worker = threading.Thread(target=run_download, daemon=True)
                worker.start()

                nonlocal_pct = [1]
                while not done.wait(timeout=1.0):
                    if expected_bytes and expected_bytes > 0:
                        current_bytes = max(0, self._dir_size_bytes(target_dir) - start_bytes)
                        ratio = min(1.0, current_bytes / expected_bytes)
                        rolling = max(nonlocal_pct[0], min(95, 1 + int(ratio * 94)))
                    else:
                        rolling = min(90, nonlocal_pct[0] + 1)
                    nonlocal_pct[0] = rolling
                    self._update(task_id, progress=rolling, message="Fetching model artifacts")
                    milestone = (rolling // 10) * 10
                    if milestone > last_pct_logged and milestone > 0:
                        log(f"[progress] {rolling}%")
                        last_pct_logged = milestone

                worker.join(timeout=0.1)
                if error_holder:
                    raise error_holder[0]
                path = result["path"]
            log("[info] finalizing")
            self._update(task_id, progress=98, message="Finalizing")

            marker.write_text(
                json.dumps({"model_id": model_id, "downloaded_at": int(time())}),
                encoding="utf-8",
            )

            self._update(
                task_id,
                status="completed",
                progress=100,
                message="Downloaded",
                local_path=str(path),
                finished_at=time(),
            )
            log("[done] download complete")
        except Exception as exc:
            self._update(
                task_id,
                status="failed",
                progress=100,
                message="Failed",
                error=str(exc),
                finished_at=time(),
            )
            log(f"[error] {exc}")
