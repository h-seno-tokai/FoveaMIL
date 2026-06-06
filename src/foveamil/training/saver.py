"""検証指標の改善時にモデル重みを保存する管理器

``save_metric`` が ``"loss"`` のとき検証損失の最小，``"f1"`` のとき検証 weighted F1，
``"macro_f1"`` のとき検証 macro F1 の最大を更新したときに ``model_best_{metric}.pt`` を
保存する任意の接尾辞でも保存でき，
保存済みの best 重みパスを返す重みは ``weights_dir`` に保存する（未指定なら
``save_path`` にフォールバック）
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# 検証損失の最小を追う save_metric
SAVE_METRIC_LOSS = "loss"
# 検証 weighted F1 の最大を追う save_metric
SAVE_METRIC_F1 = "f1"
# 検証 macro F1 の最大を追う save_metric（不均衡多クラスの主指標向き）
SAVE_METRIC_MACRO_F1 = "macro_f1"
# 選べる save_metric の一覧
SAVE_METRICS = (SAVE_METRIC_LOSS, SAVE_METRIC_F1, SAVE_METRIC_MACRO_F1)
# save_metric → (summary のキー, 大きいほど良いか)
_METRIC_SPEC = {
    SAVE_METRIC_LOSS: ("val_loss", False),
    SAVE_METRIC_F1: ("val_weighted_f1", True),
    SAVE_METRIC_MACRO_F1: ("val_macro_f1", True),
}


class ModelSaver:
    """検証指標の改善時にモデル重みを保存する管理器

    Args:
        save_path: 既定の保存ディレクトリ（``weights_dir`` 未指定時の重み保存先）
        save_metric: ``"loss"``（検証損失最小）/ ``"f1"``（検証 weighted F1 最大）
        weights_dir: 重み（``.pt``）の保存ディレクトリ``None`` なら ``save_path``
    """

    def __init__(
        self,
        save_path: str,
        save_metric: str = SAVE_METRIC_LOSS,
        weights_dir: Optional[str] = None,
    ) -> None:
        if save_metric not in SAVE_METRICS:
            raise ValueError(
                f"save_metric must be one of {SAVE_METRICS}, got '{save_metric}'"
            )
        self.save_path = save_path
        self.save_metric = save_metric
        self.weights_dir = weights_dir if weights_dir is not None else save_path
        self.best_epoch: Optional[int] = None
        self.best_value: Optional[float] = None
        os.makedirs(self.weights_dir, exist_ok=True)

    def _best_filename(self) -> str:
        """現在の ``save_metric`` に対応する best ファイル名を返す"""
        return f"model_best_{self.save_metric}.pt"

    def __call__(
        self, model: nn.Module, summary: Dict[str, float], epoch: Optional[int] = None
    ) -> None:
        """検証指標が改善していれば best 重みを保存し best epoch/値を記録する

        Args:
            model: 保存対象のモデル
            summary: ``save_metric`` に対応するキー（``val_loss`` /
                ``val_weighted_f1`` / ``val_macro_f1``）を含む辞書
            epoch: 現在のエポック（best 更新時に記録する）
        """
        key, higher_is_better = _METRIC_SPEC[self.save_metric]
        value = float(summary[key])
        # best 未保存（初回）は指標値に依らず必ず保存する 全エポックで val 指標が
        # 改善しない退化 combo でも best 重みが存在し test が last で黙って評価される
        # 事故を防ぐ
        if self.best_epoch is None or (
            value > self.best_value
            if higher_is_better
            else value < self.best_value
        ):
            if self.best_value is None:
                logger.info("%s=%.6f (initial), saving best model", key, value)
            else:
                logger.info(
                    "%s improved (%.6f -> %.6f), saving best model",
                    key,
                    self.best_value,
                    value,
                )
            self._save(model, self._best_filename())
            self.best_epoch = epoch
            self.best_value = value

    def _save(self, model: nn.Module, filename: str) -> str:
        """``model.state_dict()`` を ``weights_dir/filename`` に保存しパスを返す"""
        path = os.path.join(self.weights_dir, filename)
        torch.save(model.state_dict(), path)
        return path

    def save_model(self, model: nn.Module, suffix: str) -> str:
        """``model_{suffix}.pt`` として重みを保存しパスを返す

        Args:
            model: 保存対象のモデル
            suffix: ファイル名の接尾辞
        """
        path = self._save(model, f"model_{suffix}.pt")
        logger.info("saved model as model_%s.pt", suffix)
        return path

    def load_best_path(self) -> Optional[str]:
        """best 重みファイルのパスを返す（無ければ ``None``）"""
        path = os.path.join(self.weights_dir, self._best_filename())
        if os.path.exists(path):
            return path
        return None
