"""模型路由：flash / flagship 双模型调度策略（config 可切换，消融用）。"""

from __future__ import annotations

from ..config import ModelConfig, RouterConfig
from .generator import AnswerObj


def pick_model(cfg: RouterConfig, qtype: str, flash_cfg: ModelConfig,
               flagship_cfg: ModelConfig) -> ModelConfig:
    """按策略决定首轮生成用哪个模型。"""
    if cfg.strategy == "flagship_only":
        return flagship_cfg
    if cfg.strategy == "by_type":
        return flagship_cfg if cfg.type_routes.get(qtype) == "flagship" else flash_cfg
    return flash_cfg  # flash_only 与 cascade 首轮都用 flash


def should_upgrade(cfg: RouterConfig, answer: AnswerObj, qtype: str) -> bool:
    """cascade 策略下判断是否升级旗舰重答。"""
    if cfg.strategy != "cascade":
        return False
    if answer.abstained and answer.confidence >= 0.7:
        return False  # 明确的证据不足弃权不必升级
    return answer.confidence < cfg.cascade_confidence
