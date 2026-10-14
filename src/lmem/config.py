"""配置定义与加载。

约定：
- 所有字段都有内置默认值（与 configs/default.yaml 对应）；
- API Key 一律从环境变量读取，不落盘；
- 模型规格（上下文窗口等）未硬编码——由 probe 子命令实测。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

import yaml

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


def env_or_none(name: str) -> Optional[str]:
    v = os.environ.get(name)
    return v if v else None


@dataclass
class ModelConfig:
    name: str = "glm-5.3-flash"
    base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    api_key_env: str = "ZAI_API_KEY"
    temperature: float = 0.2
    max_tokens: int = 1500
    # None = 不向 API 传思考开关；True/False = 显式传 extra_body.thinking
    thinking: Optional[bool] = None
    extra_body: dict = field(default_factory=dict)

    def api_key(self) -> Optional[str]:
        return env_or_none(self.api_key_env)

    def has_key(self) -> bool:
        return self.api_key() is not None

    def apply(self, d: dict) -> "ModelConfig":
        allowed = {f for f in self.__dataclass_fields__ if f in d}
        return replace(self, **{k: d[k] for k in allowed})


@dataclass
class RouterConfig:
    # flash_only | cascade | by_type | flagship_only
    strategy: str = "cascade"
    cascade_confidence: float = 0.55
    # by_type 策略下，这些题型强制走旗舰
    complex_types: tuple = ("multi_session", "temporal_reasoning", "knowledge_update")
    type_routes: dict = field(
        default_factory=lambda: {
            "single_session_user": "flash",
            "single_session_assistant": "flash",
            "single_session_preference": "flash",
            "multi_session": "flagship",
            "temporal_reasoning": "flagship",
            "knowledge_update": "flagship",
            "abstention": "flash",
        }
    )

    def apply(self, d: dict) -> "RouterConfig":
        return replace(self, **{k: v for k, v in d.items() if k in self.__dataclass_fields__})


@dataclass
class GateConfig:
    coverage_min: float = 0.25
    confidence_min: float = 0.30

    def apply(self, d: dict) -> "GateConfig":
        return replace(self, **{k: v for k, v in d.items() if k in self.__dataclass_fields__})


@dataclass
class RetrieverConfig:
    bm25_top_k: int = 24
    final_top_k: int = 8
    as_of_filter: bool = True
    profile_max: int = 5

    def apply(self, d: dict) -> "RetrieverConfig":
        return replace(self, **{k: v for k, v in d.items() if k in self.__dataclass_fields__})


@dataclass
class WriterConfig:
    supersede_on_conflict: bool = True
    fact_confidence_min: float = 0.5

    def apply(self, d: dict) -> "WriterConfig":
        return replace(self, **{k: v for k, v in d.items() if k in self.__dataclass_fields__})


@dataclass
class HarnessConfig:
    backend: str = "auto"  # auto | openai | mock
    store_path: str = "runs/store.sqlite3"
    flash: ModelConfig = field(default_factory=ModelConfig)
    flagship: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            name="glm-5.3", base_url="https://open.bigmodel.cn/api/paas/v4", max_tokens=2000
        )
    )
    judge: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            name="glm-5.3", base_url="https://open.bigmodel.cn/api/paas/v4", max_tokens=800
        )
    )
    router: RouterConfig = field(default_factory=RouterConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    writer: WriterConfig = field(default_factory=WriterConfig)

    def effective_backend(self) -> str:
        """auto：任一模型配了 Key 就走真实 API，否则 Mock 演示模式。"""
        if self.backend == "auto":
            return "openai" if (self.flash.has_key() or self.flagship.has_key()) else "mock"
        return self.backend

    def apply(self, d: dict) -> "HarnessConfig":
        c = self
        for key in ("backend", "store_path"):
            if key in d:
                c = replace(c, **{key: d[key]})
        for key, cls in (
            ("flash", ModelConfig),
            ("flagship", ModelConfig),
            ("judge", ModelConfig),
        ):
            if isinstance(d.get(key), dict):
                c = replace(c, **{key: getattr(c, key).apply(d[key])})
        for key, attr in (
            ("router", "router"),
            ("gate", "gate"),
            ("retriever", "retriever"),
            ("writer", "writer"),
        ):
            if isinstance(d.get(key), dict):
                c = replace(c, **{attr: getattr(c, attr).apply(d[key])})
        return c


def load_config(path: Optional[str] = None) -> HarnessConfig:
    """从 YAML 加载配置；文件不存在时返回内置默认值。"""
    cfg = HarnessConfig()
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    if p.exists():
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        cfg = cfg.apply(data)
    return cfg
