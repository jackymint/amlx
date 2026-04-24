from __future__ import annotations

import json
import os
import platform
import re
import shutil
import signal
import subprocess
import threading
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from time import sleep, time
from typing import Any, Callable

from amlx.model_type import is_vlm

CatalogItem = dict[str, Any]
Downloader = Callable[[str, Path], Path]
SearchProvider = Callable[[str, int], list[dict[str, Any]]]
FineTuner = Callable[[Any, Any], None]


CATALOG: list[CatalogItem] = [
    {
        "id": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        "label": "Qwen2.5 Coder 7B 4bit",
        "size": "~4.7 GB",
        "tags": "coding, balanced",
        "disk_gb": 8.0,
        "min_ram_gb": 8.0,
        "rec_ram_gb": 16.0,
        "requires_apple_silicon": True,
    },
    {
        "id": "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "label": "Llama 3.2 3B 4bit",
        "size": "~2.1 GB",
        "tags": "fast, lightweight",
        "disk_gb": 4.0,
        "min_ram_gb": 6.0,
        "rec_ram_gb": 12.0,
        "requires_apple_silicon": True,
    },
    {
        "id": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        "label": "Mistral 7B Instruct 4bit",
        "size": "~4.4 GB",
        "tags": "general, stable",
        "disk_gb": 8.0,
        "min_ram_gb": 8.0,
        "rec_ram_gb": 16.0,
        "requires_apple_silicon": True,
    },
    {
        "id": "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit",
        "label": "DeepSeek R1 Distill Qwen 7B",
        "size": "~4.9 GB",
        "tags": "reasoning, coding",
        "disk_gb": 9.0,
        "min_ram_gb": 12.0,
        "rec_ram_gb": 24.0,
        "requires_apple_silicon": True,
    },
]


@dataclass(slots=True)
class QuantizeTask:
    task_id: str
    model_id: str
    effective_model: str
    output_path: str
    q_bits: int
    q_group_size: int
    status: str
    progress: int
    message: str
    started_at: float
    updated_at: float
    finished_at: float | None = None
    error: str | None = None
    cancelled: bool = False


@dataclass(slots=True)
class DownloadTask:
    task_id: str
    model_id: str
    status: str
    progress: int
    message: str
    started_at: float
    updated_at: float
    finished_at: float | None = None
    local_path: str | None = None
    error: str | None = None
    cancelled: bool = False


@dataclass(slots=True)
class FineTuneTask:
    task_id: str
    model_id: str
    effective_model: str
    profile: str
    profile_slug: str
    round: int
    status: str
    progress: int
    message: str
    fine_tune_type: str
    epochs: int
    train_samples: int
    started_at: float
    updated_at: float
    learning_rate: float = 1e-5
    lora_rank: int = 8
    lora_layers: int = 16
    max_seq_length: int = 2048
    finished_at: float | None = None
    adapter_path: str | None = None
    resume_adapter_file: str | None = None
    run_root: str | None = None
    merged_path: str | None = None
    error: str | None = None


