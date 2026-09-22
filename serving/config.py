"""Config-driven settings for the serving layer.

Precedence: environment variables (RECSYS_*) > config.yaml > built-in defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import yaml

SERVING_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.environ.get("RECSYS_CONFIG", os.path.join(SERVING_DIR, "config.yaml"))

_ENV_MAP = {
    "RECSYS_ARTIFACT_DIR": "artifact_dir",
    "RECSYS_MODEL_PATH": "model_path",
    "RECSYS_TOP_K_DEFAULT": "top_k_default",
    "RECSYS_TOP_K_MAX": "top_k_max",
    "RECSYS_HOST": "host",
    "RECSYS_PORT": "port",
    "RECSYS_WORKERS": "workers",
    "RECSYS_SEED": "seed",
    "RECSYS_TORCH_THREADS": "torch_threads",
}

_INT_FIELDS = {"top_k_default", "top_k_max", "port", "workers", "seed", "torch_threads"}


@dataclass
class Settings:
    artifact_dir: str = "../artifacts"
    model_path: str | None = None
    top_k_default: int = 10
    top_k_max: int = 100
    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1
    seed: int = 42
    torch_threads: int = 1

    @property
    def resolved_artifact_dir(self) -> str:
        d = self.artifact_dir
        return d if os.path.isabs(d) else os.path.normpath(os.path.join(SERVING_DIR, d))

    @property
    def resolved_model_path(self) -> str:
        if self.model_path:
            p = self.model_path
            return p if os.path.isabs(p) else os.path.normpath(os.path.join(SERVING_DIR, p))
        return os.path.join(self.resolved_artifact_dir, "twotower.pt")


def load_settings(config_path: str | None = None) -> Settings:
    """Load settings from YAML file, then apply RECSYS_* env overrides."""
    s = Settings()
    path = config_path or CONFIG_PATH
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            if hasattr(s, k) and v is not None:
                setattr(s, k, v)
    for env, attr in _ENV_MAP.items():
        raw = os.environ.get(env)
        if raw is None or raw == "":
            continue
        setattr(s, attr, int(raw) if attr in _INT_FIELDS else raw)
    if not (1 <= s.top_k_default <= s.top_k_max):
        raise ValueError("top_k_default must be within [1, top_k_max]")
    return s
