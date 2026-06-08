"""FoveaMIL の学習ループ（Lazy 方式）

バッチサイズ 1 の DataLoader でエポックごとに train→val を回し，検証損失で学習率を
調整しつつ best 重みを保存する各サンプルは最低倍率の特徴のみを読み込み，モデルの
段階 forward が返す選択結果に応じて高倍率の子パッチを :class:`FeatureAccessor` で
都度ロードし，選択重みを子特徴へ掛けて補助アテンションへ勾配を流す``test`` で best
重みを読み直して test 指標を返し，混同行列を保存する重み（``.pt``）は ``weights_dir``
へ，tensorboard・混同行列は ``save_path`` へ分けて保存する
tensorboard と混同行列 PNG は任意機能で，依存が無ければ自動的に省く
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    WeightedRandomSampler,
)

from foveamil.models import FoveaMIL
from foveamil.models.regularizers import ForwardContext, iter_active_regularizers
from foveamil.training.accessor import FeatureAccessor
from foveamil.training.config import TrainConfig
from foveamil.training.dataset import feature_bag_collate
from foveamil.training.hierarchy import validate_magnification_hierarchy
from foveamil.training.losses import build_loss
from foveamil.training.metrics import MetricLogger
from foveamil.training.saver import ModelSaver
from foveamil.training.zoom_driver import build_zoom_driver

logger = logging.getLogger(__name__)

# DataLoader のバッチサイズ
BATCH_SIZE = 1
# Lazy ロードはメインプロセスで処理するためワーカ数は 0 固定
NUM_WORKERS = 0
# Lazy ロードでは pin_memory を使わない
PIN_MEMORY = False
# Adam のモーメント係数
ADAM_BETAS = (0.9, 0.999)
# Adam の数値安定化 eps
ADAM_EPS = 1e-8
# ReduceLROnPlateau のモード
SCHEDULER_MODE = "min"
# perturbed top-k へ渡す平滑化引数のキー
TOPK_PERTURBED_KEY = "sigma"
# fast_sparse top-k へ渡す平滑化引数のキー
TOPK_SPARSE_KEY = "epsilon"
# top-k 手法名
TOPK_PERTURBED = "perturbed"
TOPK_FAST_SPARSE = "fast_sparse"
# 補助アテンション正規化器名（追加引数を取るもの）
AUX_NORM_TEMPERATURE = "temperature"
AUX_NORM_ENTMAX = "entmax"
# 各正規化器が取る追加引数のキー
AUX_NORM_TEMPERATURE_KEY = "temperature"
AUX_NORM_ALPHA_KEY = "alpha"
# DPP 選択コントローラ名
SELECTOR_DPP = "dpp"
# 自己アテンション集約器名
AGGREGATOR_SELF_ATTN = "self_attn"
# last モデルの接尾辞
LAST_SUFFIX = "last"
# 評価する split 名
SPLIT_VAL = "val"
SPLIT_TEST = "test"
SPLIT_TRAIN = "train"
# 予測 CSV のファイル名テンプレート
PREDICTIONS_CSV_TEMPLATE = "predictions_{split}.csv"
# 指標 JSON のファイル名テンプレート
METRICS_JSON_TEMPLATE = "metrics_{split}.json"
# 学習履歴 CSV のファイル名
HISTORY_CSV = "history.csv"
# 混同行列の保存ファイル名テンプレート（split 別・生/正規化）
CONFUSION_MATRIX_NPY_TEMPLATE = "confusion_matrix_{split}.npy"
CONFUSION_MATRIX_NORM_NPY_TEMPLATE = "confusion_matrix_{split}_normalized.npy"
CONFUSION_MATRIX_PNG_TEMPLATE = "confusion_matrix_{split}.png"
CONFUSION_MATRIX_NORM_PNG_TEMPLATE = "confusion_matrix_{split}_normalized.png"
# 予測 CSV の確率/​logit 列名テンプレート
PROB_COL_TEMPLATE = "prob_{i}"
LOGIT_COL_TEMPLATE = "logit_{i}"


def _topk_kwargs(config: TrainConfig) -> dict:
    """``topk_method`` に応じて ``k_sigma`` を平滑化引数へ写像する

    ``perturbed`` なら ``{"sigma": k_sigma}``，``fast_sparse`` なら
    ``{"epsilon": k_sigma}``それ以外は空辞書を返す
    """
    if config.topk_method == TOPK_PERTURBED:
        return {TOPK_PERTURBED_KEY: config.k_sigma}
    if config.topk_method == TOPK_FAST_SPARSE:
        return {TOPK_SPARSE_KEY: config.k_sigma}
    return {}


def regularizer_loss(regularizers, context: ForwardContext, label: Tensor):
    """有効な正則化項と寄与損失を合算する

    各正則化項 ``reg`` の ``reg.weight * reg(context, label)`` と，
    ``context.extra_losses`` の各値を足し合わせる項が無ければ ``0.0`` を返す

    Args:
        regularizers: :class:`Regularizer` のリスト
        context: 段階 forward の文脈
        label: 正解クラス ``[B]``

    Returns:
        合算した補助損失（スカラまたは ``0.0``）
    """
    total = 0.0
    for reg in regularizers:
        total = total + reg.weight * reg(context, label)
    for value in context.extra_losses.values():
        total = total + value
    return total


def _aux_norm_kwargs(config: TrainConfig) -> dict:
    """``aux_norm`` に応じて温度 / α を追加引数へ写像する

    ``temperature`` なら ``{"temperature": aux_norm_temperature}``，``entmax`` なら
    ``{"alpha": aux_norm_alpha}``それ以外は空辞書を返す
    """
    if config.aux_norm == AUX_NORM_TEMPERATURE:
        return {AUX_NORM_TEMPERATURE_KEY: config.aux_norm_temperature}
    if config.aux_norm == AUX_NORM_ENTMAX:
        return {AUX_NORM_ALPHA_KEY: config.aux_norm_alpha}
    return {}


def _selector_kwargs(config: TrainConfig) -> dict:
    """選択コントローラへ渡す追加引数を設定から組み立てる

    ``selector=="dpp"`` なら ``similarity`` / ``temperature`` / ``quality_beta`` /
    ``rbf_gamma`` / ``use_gumbel`` を返すそれ以外は空辞書を返す（既定 top-k は追加引数を
    持たない）
    """
    if config.selector == SELECTOR_DPP:
        return {
            "similarity": config.dpp_similarity,
            "temperature": config.dpp_temperature,
            "quality_beta": config.dpp_quality_beta,
            "rbf_gamma": config.dpp_rbf_gamma,
            "use_gumbel": config.dpp_use_gumbel,
        }
    return {}


def _aggregator_kwargs(config: TrainConfig) -> dict:
    """``aggregator`` に応じて集約器固有の追加引数を組み立てる

    ``aggregator=="self_attn"`` なら ``num_heads`` / ``num_landmarks`` を返す
    それ以外（既定 ``abmil``）は空辞書を返し，bit 互換の従来挙動を保つ
    """
    if config.aggregator == AGGREGATOR_SELF_ATTN:
        return {
            "num_heads": config.aggregator_num_heads,
            "num_landmarks": config.aggregator_num_landmarks,
        }
    return {}


def build_foveamil_from_config(config: TrainConfig, num_layers: int) -> FoveaMIL:
    """設定と倍率数から FoveaMIL を構築する（学習・再構築で同一の組立を共有する）

    Args:
        config: 学習設定
        num_layers: 倍率数

    Returns:
        構築した :class:`FoveaMIL`
    """
    return FoveaMIL(
        in_feat_dim=config.in_feat_dim,
        hidden_feat_dim=config.hidden_feat_dim,
        out_feat_dim=config.out_feat_dim,
        dropout=config.drop_out,
        proj_num_layers=config.proj_num_layers,
        proj_layer_norm=config.proj_layer_norm,
        k_sample=config.k_sample,
        n_cls=config.n_cls,
        num_layers=num_layers,
        topk_method=config.topk_method,
        topk_kwargs=_topk_kwargs(config),
        aux_norm=config.aux_norm,
        aux_norm_kwargs=_aux_norm_kwargs(config),
        selector=config.selector,
        selector_kwargs=_selector_kwargs(config),
        fusion=config.fusion,
        aggregator=config.aggregator,
        aggregator_kwargs=_aggregator_kwargs(config),
        head_type=config.head_type,
        head_hidden_dim=config.head_hidden_dim,
        instance_loss=config.instance_loss,
        inst_k=config.inst_k,
        inst_subtyping=config.inst_subtyping,
    )


def _seed_everything(seed: int) -> None:
    """乱数シードを固定する"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _class_weights(dataset, n_cls: int) -> torch.Tensor:
    """データセットからサンプルごとのクラス頻度逆数重みを作る"""
    labels = [dataset.get_label(idx) for idx in range(len(dataset))]
    counts = np.bincount(labels, minlength=n_cls).astype(np.float64)
    total = float(len(labels))
    per_class = np.where(counts > 0, total / counts, 0.0)
    weights = [per_class[label] for label in labels]
    return torch.DoubleTensor(weights)


def _class_frequencies(dataset, n_cls: int) -> List[int]:
    """データセットからクラスごとのサンプル件数を返す（不均衡対応損失の入力）"""
    labels = [dataset.get_label(idx) for idx in range(len(dataset))]
    counts = np.bincount(labels, minlength=n_cls)
    return [int(c) for c in counts]


class Trainer:
    """FoveaMIL の学習・検証・評価を司る

    Args:
        config: 学習設定
        train_ds: 学習データセット
        val_ds: 検証データセット
        test_ds: 評価データセット
        save_path: ログ・結果（tensorboard・混同行列）の出力先ディレクトリ
        weights_dir: 重み（``.pt``）の保存先ディレクトリ``None`` なら ``save_path``
    """

    def __init__(
        self,
        config: TrainConfig,
        train_ds,
        val_ds,
        test_ds,
        save_path: str,
        weights_dir: Optional[str] = None,
    ) -> None:
        self.config = config
        self.save_path = save_path
        self.weights_dir = weights_dir if weights_dir is not None else save_path
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.weights_dir, exist_ok=True)

        _seed_everything(config.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_cls = config.n_cls

        # 高倍率の子パッチを都度ロードするための情報を学習データセットから取得する
        self.feature_root = train_ds.feature_root
        self.encoder = train_ds.encoder
        self.magnifications = train_ds.magnifications
        self.feature_type = train_ds.feature_type
        self.num_layers = len(self.magnifications)
        validate_magnification_hierarchy(self.magnifications)

        self.model = build_foveamil_from_config(config, self.num_layers).to(
            self.device
        )
        self.zoom_driver = build_zoom_driver(config, self.model)
        self.instance_enabled = config.instance_loss
        self.bag_weight = config.bag_weight
        self.regularizers = iter_active_regularizers(config)

        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.class_names = self._resolve_class_names(train_ds)
        self.history: List[Dict[str, Any]] = []

        self.train_loader = self._build_train_loader(train_ds)
        self.val_loader = self._build_eval_loader(val_ds)
        self.test_loader = self._build_eval_loader(test_ds)

        self.criterion = build_loss(
            config.loss_type,
            _class_frequencies(train_ds, self.n_cls),
            tau=config.loss_tau,
            beta=config.loss_cb_beta,
            ldam_max_margin=config.loss_ldam_max_margin,
        ).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.lr,
            betas=ADAM_BETAS,
            eps=ADAM_EPS,
            weight_decay=config.reg,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode=SCHEDULER_MODE,
            factor=config.scheduler_decay_rate,
            patience=config.scheduler_patience,
        )
        self.model_saver = ModelSaver(
            self.save_path, config.save_metric, weights_dir=self.weights_dir
        )
        self.tb_writer = self._build_tb_writer()

    def _build_train_loader(self, train_ds) -> DataLoader:
        """学習用 DataLoader を作る（重み付き or ランダムサンプラ）"""
        if self.config.is_weighted_sampler:
            weights = _class_weights(train_ds, self.n_cls)
            sampler = WeightedRandomSampler(weights, len(weights))
        else:
            sampler = RandomSampler(train_ds)
        return DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            collate_fn=feature_bag_collate,
        )

    def _build_eval_loader(self, dataset) -> DataLoader:
        """検証・評価用 DataLoader を作る（順次サンプラ）"""
        return DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            sampler=SequentialSampler(dataset),
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            collate_fn=feature_bag_collate,
        )

    def _build_tb_writer(self):
        """tensorboard が利用可能なら SummaryWriter を作る（無ければ ``None``）"""
        try:
            from torch.utils.tensorboard import SummaryWriter

            return SummaryWriter(log_dir=self.save_path)
        except Exception as exc:  # noqa: BLE001
            logger.info("tensorboard disabled: %s", exc)
            return None

    def _log_scalars(self, metric_dict: Dict[str, float], step: int) -> None:
        """tensorboard が有効なら scalar をログする"""
        if self.tb_writer is None:
            return
        for key, value in metric_dict.items():
            self.tb_writer.add_scalar(key, value, step)

    def _forward(
        self, base_feats: Tensor, slide_id: str, label: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor, ForwardContext]:
        """最低倍率特徴から段階 forward で予測と forward 文脈を返す

        子特徴ローダを :class:`FeatureAccessor` から組み，:attr:`zoom_driver` に倍率
        ごとのズーム駆動を委ねる駆動は次倍率と子 global index からローダを呼び子を
        都度ロードし，各倍率のプーリング表現と選択を :class:`ForwardContext` に集める
        ``label`` は探索系の駆動が補助損失に使う（既定駆動は無視する）

        Args:
            base_feats: 最低倍率の全特徴 ``[1, N, in_feat_dim]``
            slide_id: スライド識別子
            label: 正解クラス ``[B]``（学習時の補助損失用無ければ推論）

        Returns:
            ``(logits, Y_hat, Y_prob, context)``
        """
        accessor = FeatureAccessor(
            self.feature_root, self.encoder, slide_id, self.feature_type
        )
        try:
            def child_loader(next_mag: float, child_global_indices) -> Tensor:
                return accessor.load_patches(next_mag, child_global_indices).unsqueeze(0)

            return self.zoom_driver.run(
                base_feats,
                self.magnifications,
                child_loader,
                self.device,
                label=label,
            )
        finally:
            accessor.close()

    def _regularizer_loss(self, context: ForwardContext, label: Tensor):
        """有効な正則化項と寄与損失を合算する（``regularizer_loss`` へ委譲する）"""
        return regularizer_loss(self.regularizers, context, label)

    def _train_one_epoch(self) -> float:
        """1 エポック学習し平均損失を返す

        分類損失は ``config.loss_type`` で選ぶ損失（既定は素 CE）``instance_enabled`` なら
        最低倍率の全バッグ主アテンションでインスタンス補助損失を計算し
        ``bag·bag_weight + inst·(1-bag_weight)`` を最小化する補助損失が有効なら
        ``分類損失 + Σ w_i·reg_i + Σ extra_losses`` を最小化する
        """
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0.0
        for base_feats, slide_id, label in self.train_loader:
            label = label.to(self.device)
            if self.instance_enabled:
                # 単一倍率の bag forward と補助損失を同一の射影・主アテンションから得る
                logits, _, _, inst_loss = self.model.forward_with_instance_loss(
                    base_feats.to(self.device), label
                )
                loss = (
                    self.bag_weight * self.criterion(logits, label)
                    + (1.0 - self.bag_weight) * inst_loss
                )
            else:
                logits, _, _, context = self._forward(base_feats, slide_id, label)
                loss = self.criterion(logits, label) + self._regularizer_loss(
                    context, label
                )
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return total_loss / len(self.train_loader)

    def _evaluate(self, loader) -> tuple:
        """``loader`` を評価し ``(平均損失, 指標辞書, MetricLogger, slide_ids)`` を返す

        各サンプルの logit も MetricLogger に渡し，slide_id を loader 順に集める
        （予測 CSV の行と一致させる）空 loader は損失 ``nan``
        """
        self.model.eval()
        metric_logger = MetricLogger(n_cls=self.n_cls)
        slide_ids: List[str] = []
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for base_feats, slide_id, label in loader:
                label = label.to(self.device)
                logits, Y_hat, Y_prob, _ = self._forward(base_feats, slide_id)
                metric_logger.log(Y_hat, label, Y_prob, Y_logit=logits)
                slide_ids.append(slide_id)
                total_loss += self.criterion(logits, label).item()
                n += 1
        avg_loss = total_loss / n if n else float("nan")
        return avg_loss, metric_logger.get_summary(), metric_logger, slide_ids

    def train(self) -> None:
        """最大エポック数まで train→val を回し best/last 重みを保存する"""
        logger.info(
            "start training: %d-layer FoveaMIL, max_epochs=%d, device=%s",
            self.num_layers,
            self.config.max_epochs,
            self.device,
        )
        for epoch in range(self.config.max_epochs):
            train_loss = self._train_one_epoch()
            val_loss, val_metrics, _, _ = self._evaluate(self.val_loader)

            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                "epoch [%d/%d] train_loss=%.4f val_loss=%.4f val_wF1=%.4f lr=%.2e",
                epoch + 1,
                self.config.max_epochs,
                train_loss,
                val_loss,
                val_metrics["weighted_f1"],
                current_lr,
            )
            self._log_scalars({"train/loss": train_loss}, epoch)
            self._log_scalars(
                {f"val/{k}": v for k, v in val_metrics.items()}, epoch
            )
            self._log_scalars({"val/loss": val_loss, "lr": current_lr}, epoch)
            self._record_history(epoch, current_lr, train_loss, val_loss, val_metrics)

            self.scheduler.step(val_loss)
            self.model_saver(
                self.model,
                {
                    "val_loss": val_loss,
                    "val_weighted_f1": val_metrics["weighted_f1"],
                    "val_macro_f1": val_metrics["macro_f1"],
                },
                epoch=epoch,
            )

        self.model_saver.save_model(self.model, LAST_SUFFIX)
        self._save_history()
        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()

    def _load_best(self) -> None:
        """best 重みがあれば読み直す（無ければ現状のまま）"""
        best_path = self.model_saver.load_best_path()
        if best_path is None:
            logger.info("best model not found, testing with current weights")
            return
        self.model.load_state_dict(
            torch.load(best_path, map_location=self.device)
        )
        logger.info("loaded best model from %s", best_path)

    def _resolve_class_names(self, dataset) -> List[str]:
        """クラス整数→クラス名の並びを得る（label_dict 優先，無ければ config/添字）"""
        label_dict = getattr(dataset, "label_dict", None)
        if label_dict:
            inverse = {idx: name for name, idx in label_dict.items()}
            return [str(inverse.get(i, i)) for i in range(self.n_cls)]
        if self.config.classes:
            return [str(c) for c in self.config.classes]
        return [str(i) for i in range(self.n_cls)]

    def _record_history(
        self, epoch: int, lr: float, train_loss: float, val_loss: float,
        val_metrics: Dict[str, float],
    ) -> None:
        """1 エポックの学習履歴を蓄積する"""
        row: Dict[str, Any] = {
            "epoch": epoch, "lr": lr,
            "train_loss": train_loss, "val_loss": val_loss,
        }
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        self.history.append(row)

    def _save_history(self) -> None:
        """学習履歴を ``history.csv`` に保存する"""
        if not self.history:
            return
        path = os.path.join(self.save_path, HISTORY_CSV)
        pd.DataFrame(self.history).to_csv(path, index=False)

    def _save_predictions(
        self, split: str, slide_ids: List[str], arrays: Dict[str, Any]
    ) -> None:
        """split の生予測を ``predictions_{split}.csv`` に保存する"""
        data: Dict[str, Any] = {
            "slide_id": [str(s) for s in slide_ids],
            "y_true": arrays["y_true"],
            "y_pred": arrays["y_pred"],
        }
        df = pd.DataFrame(data)
        prob = arrays["y_prob"]
        if prob is not None:
            for i in range(prob.shape[1]):
                df[PROB_COL_TEMPLATE.format(i=i)] = prob[:, i]
        logit = arrays["y_logit"]
        if logit is not None:
            for i in range(logit.shape[1]):
                df[LOGIT_COL_TEMPLATE.format(i=i)] = logit[:, i]
        path = os.path.join(self.save_path, PREDICTIONS_CSV_TEMPLATE.format(split=split))
        df.to_csv(path, index=False)

    def _save_metrics_json(
        self, split: str, metrics: Dict[str, float], loss: float, n_samples: int
    ) -> None:
        """split の指標を ``metrics_{split}.json`` に保存する（loss・件数を併記）"""
        payload: Dict[str, Any] = dict(metrics)
        payload["loss"] = loss
        payload["n_samples"] = n_samples
        path = os.path.join(self.save_path, METRICS_JSON_TEMPLATE.format(split=split))
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def _save_confusion_png(
        self, cm: np.ndarray, png_name: str, title: str, normalized: bool
    ) -> None:
        """混同行列を PNG 保存する（クラス名ラベル，matplotlib 無ければ省く）"""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            im = ax.imshow(cm, cmap="Blues")
            fig.colorbar(im, ax=ax)
            ticks = list(range(len(self.class_names)))
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels(self.class_names, rotation=45, ha="right")
            ax.set_yticklabels(self.class_names)
            fmt = "{:.2f}" if normalized else "{:d}"
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    value = cm[i, j] if normalized else int(cm[i, j])
                    ax.text(j, i, fmt.format(value), ha="center", va="center")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(title)
            fig.tight_layout()
            fig.savefig(os.path.join(self.save_path, png_name), dpi=300)
            plt.close(fig)
        except Exception as exc:  # noqa: BLE001
            logger.info("confusion matrix PNG skipped: %s", exc)

    def _save_confusion_matrix(self, split: str, metric_logger: MetricLogger) -> None:
        """split の混同行列（生・行正規化）を npy と PNG で保存する"""
        cm = metric_logger.get_confusion_matrix()
        cm_norm = metric_logger.get_confusion_matrix(normalize=True)
        np.save(
            os.path.join(self.save_path, CONFUSION_MATRIX_NPY_TEMPLATE.format(split=split)),
            cm,
        )
        np.save(
            os.path.join(
                self.save_path, CONFUSION_MATRIX_NORM_NPY_TEMPLATE.format(split=split)
            ),
            cm_norm,
        )
        self._save_confusion_png(
            cm, CONFUSION_MATRIX_PNG_TEMPLATE.format(split=split),
            f"Confusion Matrix ({split})", normalized=False,
        )
        self._save_confusion_png(
            cm_norm, CONFUSION_MATRIX_NORM_PNG_TEMPLATE.format(split=split),
            f"Confusion Matrix ({split}, normalized)", normalized=True,
        )

    def _class_breakdown(self) -> Dict[str, Dict[str, int]]:
        """train/val/test の split 別クラス内訳（クラス名→件数）を返す"""
        breakdown: Dict[str, Dict[str, int]] = {}
        for split, dataset in (
            (SPLIT_TRAIN, self.train_ds),
            (SPLIT_VAL, self.val_ds),
            (SPLIT_TEST, self.test_ds),
        ):
            counts = dataset.class_counts()
            breakdown[split] = {
                self.class_names[i] if i < len(self.class_names) else str(i): int(c)
                for i, c in sorted(counts.items())
            }
        return breakdown

    def selection_info(self) -> Dict[str, Any]:
        """モデル選択の記録（save_metric / best_epoch / best_value / n_epochs）を返す"""
        return {
            "save_metric": self.config.save_metric,
            "best_epoch": self.model_saver.best_epoch,
            "best_value": self.model_saver.best_value,
            "n_epochs": self.config.max_epochs,
        }

    def evaluate_best(self, save_train: bool = False) -> Dict[str, Any]:
        """best 重みで val/test（任意で train）を評価し成果物を保存する

        各 split の予測 CSV・指標 JSON・混同行列を ``save_path`` に保存する

        Args:
            save_train: ``True`` なら train split も評価・保存する

        Returns:
            ``{"val": {...}, "test": {...}, ["train": {...},] "selection": {...},
            "class_breakdown": {...}}``
        """
        self._load_best()
        splits = [(SPLIT_VAL, self.val_loader), (SPLIT_TEST, self.test_loader)]
        if save_train:
            splits.append((SPLIT_TRAIN, self._build_eval_loader(self.train_ds)))

        results: Dict[str, Any] = {}
        for split, loader in splits:
            loss, summary, metric_logger, slide_ids = self._evaluate(loader)
            logger.info("%s_loss=%.4f", split, loss)
            self._log_scalars({f"{split}/{k}": v for k, v in summary.items()}, 0)
            self._save_predictions(split, slide_ids, metric_logger.get_arrays())
            self._save_metrics_json(split, summary, loss, len(slide_ids))
            self._save_confusion_matrix(split, metric_logger)
            results[split] = summary

        results["selection"] = self.selection_info()
        results["class_breakdown"] = self._class_breakdown()
        return results

    def test(self) -> Dict[str, float]:
        """best 重みで評価し test 指標を返す（val/test の成果物も保存する）"""
        return self.evaluate_best()[SPLIT_TEST]
