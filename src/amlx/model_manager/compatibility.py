from __future__ import annotations

from .shared import *


class ModelManagerCompatibilityMixin:
    @staticmethod
    def _total_ram_gb() -> float:
        if platform.system().lower() == "darwin":
            try:
                out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
                return int(out) / (1024**3)
            except Exception:
                pass

        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            total_pages = os.sysconf("SC_PHYS_PAGES")
            return (page_size * total_pages) / (1024**3)
        except Exception:
            return 0.0

    @staticmethod
    def _compatibility(item: CatalogItem, profile: dict[str, Any]) -> dict[str, Any]:
        min_ram = float(item.get("min_ram_gb", 8.0))
        rec_ram = float(item.get("rec_ram_gb", min_ram))
        disk = float(item.get("disk_gb", 6.0))
        requires_apple = bool(item.get("requires_apple_silicon", False))

        ram = float(profile.get("ram_gb", 0.0))
        free_disk = float(profile.get("free_disk_gb", 0.0))
        apple = bool(profile.get("apple_silicon", False))

        reasons: list[str] = []
        fit = "good"
        suitable = True

        if requires_apple and not apple:
            fit = "no"
            suitable = False
            reasons.append("Needs Apple Silicon for MLX runtime")

        if free_disk < disk:
            fit = "no"
            suitable = False
            reasons.append(f"Need ~{disk:.1f} GB free disk")

        if ram < min_ram:
            fit = "no"
            suitable = False
            reasons.append(f"Need at least {min_ram:.0f} GB RAM")
        elif ram < rec_ram and fit != "no":
            fit = "tight"
            reasons.append(f"Runs, but {rec_ram:.0f} GB RAM is recommended")
        elif fit != "no":
            fit = "good"
            reasons.append("Good fit for this machine")

        return {
            "fit": fit,
            "suitable": suitable,
            "summary": reasons[0] if reasons else "Unknown",
            "reasons": reasons,
            "required_ram_gb": min_ram,
            "recommended_ram_gb": rec_ram,
            "required_disk_gb": disk,
        }

    @staticmethod
    def _estimate_requirements(model_id: str) -> tuple[float, float, float]:
        lower = model_id.lower()
        match = re.search(r"(\d+(?:\.\d+)?)b", lower)
        params_b = float(match.group(1)) if match else 7.0

        if "2bit" in lower:
            bytes_per_param = 0.30
        elif "3bit" in lower:
            bytes_per_param = 0.40
        elif "4bit" in lower or "int4" in lower or "q4" in lower:
            bytes_per_param = 0.55
        elif "8bit" in lower or "int8" in lower:
            bytes_per_param = 1.05
        else:
            bytes_per_param = 0.70

        weights_gb = params_b * bytes_per_param
        disk_gb = max(0.5, round(weights_gb * 1.6, 1))
        min_ram_gb = max(4.0, round(weights_gb * 1.8 + 2.0, 1))
        rec_ram_gb = max(min_ram_gb + 2.0, round(weights_gb * 2.5 + 2.0, 1))
        return disk_gb, min_ram_gb, rec_ram_gb
