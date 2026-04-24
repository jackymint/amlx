from __future__ import annotations

from dataclasses import dataclass

from amlx.adapters.base import ModelAdapter
from amlx.cache.prefix import PrefixCache
from amlx.scheduler import BatchScheduler

from .complete import InferenceCompleteMixin
from .helpers import InferenceHelpersMixin


@dataclass(slots=True)
class InferenceService(InferenceHelpersMixin, InferenceCompleteMixin):
    adapter: ModelAdapter
    cache: PrefixCache
    scheduler: BatchScheduler
