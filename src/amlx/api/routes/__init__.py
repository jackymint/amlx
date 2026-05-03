from .cache_chat import register_cache_and_chat_routes
from .datasets import register_datasets_routes
from .models_catalog import register_models_catalog_routes
from .models_ops import register_model_ops_routes
from .runtime import register_runtime_routes
from .ui import register_ui_routes

__all__ = [
    "register_cache_and_chat_routes",
    "register_datasets_routes",
    "register_model_ops_routes",
    "register_models_catalog_routes",
    "register_runtime_routes",
    "register_ui_routes",
]
