from __future__ import annotations

from .shared import *

class ModelManagerPathsMetaMixin:
    def _update(self, task_id: str, **updates: str | int | float | None) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            for key, value in updates.items():
                if hasattr(task, key):
                    setattr(task, key, value)  # type: ignore[arg-type]
            task.updated_at = time()

    def _update_finetune(self, task_id: str, **updates: str | int | float | None) -> None:
        with self._lock:
            task = self._finetune_tasks.get(task_id)
            if task is None:
                return
            for key, value in updates.items():
                if hasattr(task, key):
                    setattr(task, key, value)  # type: ignore[arg-type]
            task.updated_at = time()
            snapshot = asdict(task)
        self._persist_finetune_task(snapshot)

    def _finetune_root(self) -> Path:
        return self.models_dir / ".finetunes"

    def _profile_runs_root(self, profile_slug: str) -> Path:
        return self._finetune_root() / "profiles" / profile_slug / "runs"

    def _download_log_path(self, task_id: str) -> Path:
        log_dir = self.models_dir / ".download_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir / f"{task_id}.log"

    @staticmethod
    def _normalize_profile_name(profile: str | None, *, model_id: str) -> str:
        raw = str(profile or "").strip()
        if raw:
            return raw
        base = str(model_id or "model").split("/")[-1]
        return f"{base}-default"

    @staticmethod
    def _profile_slug(profile_name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", profile_name.strip().lower()).strip("-")
        return slug or "default-profile"

    @staticmethod
    def _adapter_weights_file(adapter_dir: Path) -> str | None:
        candidate = adapter_dir / "adapters.safetensors"
        if candidate.exists() and candidate.stat().st_size > 0:
            return str(candidate)
        return None

    @staticmethod
    def _finetune_meta_path(run_root: Path) -> Path:
        return run_root / "task.json"

    def _persist_finetune_task(self, task: dict[str, str | int | float | None]) -> None:
        task_id = str(task.get("task_id") or "").strip()
        if not task_id:
            return
        run_root_raw = str(task.get("run_root") or "").strip()
        run_root = Path(run_root_raw) if run_root_raw else (self._finetune_root() / task_id)
        run_root.mkdir(parents=True, exist_ok=True)
        self._write_finetune_meta(run_root, task)

    def _write_finetune_meta(self, run_root: Path, task: dict[str, str | int | float | None]) -> None:
        payload = dict(task)
        try:
            self._finetune_meta_path(run_root).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            return

    def _scan_finetune_runs(self) -> list[dict[str, str | int | float | None]]:
        root = self._finetune_root()
        if not root.exists():
            return []
        out: list[dict[str, str | int | float | None]] = []
        run_dirs: list[Path] = []
        run_dirs.extend([p for p in root.glob("*") if p.is_dir() and p.name != "profiles"])
        profiles_root = root / "profiles"
        if profiles_root.exists():
            for profile_dir in profiles_root.glob("*"):
                runs_dir = profile_dir / "runs"
                if not runs_dir.exists():
                    continue
                run_dirs.extend([p for p in runs_dir.glob("*") if p.is_dir()])
        for run_root in sorted(run_dirs):
            meta = self._load_finetune_meta(run_root)
            if meta is not None:
                out.append(meta)
                continue
            recovered = self._recover_finetune_task_from_run(run_root)
            if recovered is not None:
                out.append(recovered)
        return out

    def _load_finetune_meta(self, run_root: Path) -> dict[str, str | int | float | None] | None:
        meta_path = self._finetune_meta_path(run_root)
        if not meta_path.exists():
            return None
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return None
        except Exception:
            return None
        data.setdefault("task_id", run_root.name)
        profile_name = str(data.get("profile") or "").strip()
        if not profile_name:
            profile_name = self._profile_name_from_run(run_root)
        data.setdefault("profile", profile_name)
        data.setdefault("profile_slug", self._profile_slug(profile_name))
        data.setdefault("round", 1)
        data.setdefault("resume_adapter_file", None)
        data.setdefault("run_root", str(run_root))
        data.setdefault("merged_path", None)
        return data  # type: ignore[return-value]

    def _recover_finetune_task_from_run(self, run_root: Path) -> dict[str, str | int | float | None] | None:
        adapter_dir = run_root / "adapters"
        if not adapter_dir.exists() or not adapter_dir.is_dir():
            return None
        effective_model = ""
        model_id = ""
        fine_tune_type = "qlora"
        adapter_config = adapter_dir / "adapter_config.json"
        if adapter_config.exists():
            try:
                data = json.loads(adapter_config.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    effective_model = str(
                        data.get("model")
                        or data.get("base_model")
                        or data.get("model_path")
                        or ""
                    )
                    fine_tune_type = str(data.get("fine_tune_type") or fine_tune_type)
            except Exception:
                pass
        model_id = effective_model
        if not model_id:
            model_id = run_root.name
        profile_name = self._profile_name_from_run(run_root)
        profile_slug = self._profile_slug(profile_name)
        ts = run_root.stat().st_mtime
        return {
            "task_id": run_root.name,
            "model_id": model_id,
            "effective_model": effective_model,
            "profile": profile_name,
            "profile_slug": profile_slug,
            "round": 1,
            "status": "completed",
            "progress": 100,
            "message": "Recovered from disk",
            "fine_tune_type": fine_tune_type,
            "epochs": 0,
            "train_samples": 0,
            "started_at": ts,
            "updated_at": ts,
            "finished_at": ts,
            "adapter_path": str(adapter_dir),
            "resume_adapter_file": None,
            "run_root": str(run_root),
            "merged_path": None,
            "error": None,
        }

    @staticmethod
    def _profile_name_from_run(run_root: Path) -> str:
        parts = run_root.parts
        if "profiles" in parts:
            idx = parts.index("profiles")
            if idx + 1 < len(parts):
                return parts[idx + 1]
        return "legacy-profile"

    def _kill_proc(proc: subprocess.Popen[bytes]) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    @staticmethod
    def _model_path(model_id: str) -> Path:
        return Path(model_id.replace("/", "--"))
