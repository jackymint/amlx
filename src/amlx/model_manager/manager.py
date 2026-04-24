from __future__ import annotations

from .compatibility import ModelManagerCompatibilityMixin
from .catalog_ops import ModelManagerCatalogOpsMixin
from .core import ModelManagerCoreMixin
from .download_runner import ModelManagerDownloadRunnerMixin
from .enqueue import ModelManagerEnqueueMixin
from .finetune_lookup import ModelManagerFineTuneLookupMixin
from .finetune_runner import ModelManagerFineTuneRunnerMixin
from .finetune_save import ModelManagerFineTuneSaveMixin
from .paths_meta import ModelManagerPathsMetaMixin
from .quantize_tasks import ModelManagerQuantizeTasksMixin
from .system_ops import ModelManagerSystemOpsMixin


class ModelManager(
    ModelManagerCoreMixin,
    ModelManagerCatalogOpsMixin,
    ModelManagerQuantizeTasksMixin,
    ModelManagerFineTuneLookupMixin,
    ModelManagerFineTuneSaveMixin,
    ModelManagerEnqueueMixin,
    ModelManagerDownloadRunnerMixin,
    ModelManagerFineTuneRunnerMixin,
    ModelManagerPathsMetaMixin,
    ModelManagerSystemOpsMixin,
    ModelManagerCompatibilityMixin,
):
    pass
