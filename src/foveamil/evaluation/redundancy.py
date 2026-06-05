"""倍率間表現の冗長性を診断する（再学習なし）

学習済み FoveaMIL の段階 forward を no_grad で再現し，融合へ入る各倍率のプーリング
表現 ``M_i``（識別器ヘッド直前，和を取る前の入力）をスライドごとに集める集めた
``[L, D]`` 行列から，倍率間の余弦類似度（生・中心化），Pearson 相関，線形 CKA，
積み上げ行列のスペクトル・実効ランクを計算しスライド集合で集約する余弦類似度・相関は
``L×L`` 行列の上三角平均で 1 値にまとめ，CKA は倍率対ごとに集約する
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch

from foveamil.training.accessor import FeatureAccessor

# 倍率対が作れない最小の倍率数
_MIN_LAYERS = 2
# ノルム・分散の数値安定化 eps
_EPS = 1e-8
# バッチ軸の添字（バッチサイズ 1 前提）
_BATCH = 0
# 最低倍率の添字（倍率列は低→高）
_LOWEST_MAG_IDX = 0
# 実効ランクで無視する特異値の下限（相対）
_SPECTRUM_EPS = 1e-12


def collect_magnification_vectors(
    loaded_model,
    feature_root: str,
    encoder: str,
    slide_id: str,
    magnifications: Sequence[float],
    feature_type: str,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """学習済みモデルから 1 スライドの融合入力倍率ベクトル ``[L, D]`` を集める

    最低倍率の全特徴を読み，段階 forward の選択結果に応じて高倍率の子パッチを
    :class:`FeatureAccessor` で都度ロードする Lazy 駆動を no_grad で再現し，各倍率の
    プーリング表現 ``M_i``（融合直前）を行に並べる子特徴へ選択重みを掛ける処理は
    学習ループと揃え，モデルが実際に見た表現と一致させる

    Args:
        loaded_model: 学習済み FoveaMIL（``forward_layer`` を持つ）
        feature_root: 特徴ルートディレクトリ
        encoder: エンコーダ名
        slide_id: スライド識別子
        magnifications: 倍率の列（低→高, モデルの倍率数と一致させる）
        feature_type: ``"mean"`` / ``"cls"`` / ``"concat"``
        device: 推論デバイス``None`` なら CPU

    Returns:
        融合入力倍率ベクトル ``[L, D]``（float32）
    """
    from foveamil.training.hierarchy import (
        children_per_parent,
        compute_child_indices,
    )

    device = device or torch.device("cpu")
    loaded_model.eval()
    loaded_model.to(device)
    num_layers = len(magnifications)

    accessor = FeatureAccessor(feature_root, encoder, slide_id, feature_type)
    try:
        m_rows: List[np.ndarray] = []
        x = (
            accessor.load_all(magnifications[_LOWEST_MAG_IDX])
            .float()
            .unsqueeze(0)
            .to(device)
        )
        global_idx: Optional[np.ndarray] = None
        with torch.no_grad():
            for layer_idx in range(num_layers):
                M, select_indices, select_weight = loaded_model.forward_layer(
                    x, layer_idx
                )
                m_rows.append(M[_BATCH, 0].cpu().numpy())
                if layer_idx >= num_layers - 1:
                    continue
                mag = magnifications[layer_idx]
                next_mag = magnifications[layer_idx + 1]
                cpp = children_per_parent(mag, next_mag)
                sel_local = select_indices[_BATCH].cpu().numpy()
                child = compute_child_indices(sel_local, global_idx, children=cpp)
                x_next = (
                    accessor.load_patches(next_mag, child)
                    .float()
                    .unsqueeze(0)
                    .to(device)
                )
                w_child = select_weight.repeat_interleave(cpp, dim=1)
                x = x_next * w_child.unsqueeze(-1)
                global_idx = child
    finally:
        accessor.close()

    return np.stack(m_rows, axis=0).astype(np.float32)


def _upper_off_diag(matrix: np.ndarray) -> np.ndarray:
    """``L×L`` 行列の上三角（対角除く）の値列を返す"""
    rows, cols = np.triu_indices(matrix.shape[0], k=1)
    return matrix[rows, cols]


def _safe_nanmean(values) -> float:
    """有限値があれば nan 無視平均を，無ければ ``nan`` を返す（警告を出さない）"""
    arr = np.asarray(values, dtype=np.float64)
    if not np.isfinite(arr).any():
        return float("nan")
    return float(np.nanmean(arr))


def cosine_similarity_matrix(vectors: np.ndarray, center: bool = False) -> np.ndarray:
    """倍率ベクトル ``[L, D]`` の余弦類似度行列 ``[L, L]`` を返す

    ``center=True`` は各ベクトルを次元方向に中心化してから余弦を取る（Pearson 相関に
    一致する）

    Args:
        vectors: 倍率ベクトル ``[L, D]``
        center: 次元方向の中心化を行うか

    Returns:
        余弦類似度行列 ``[L, L]``
    """
    v = vectors.astype(np.float64)
    if center:
        v = v - v.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    normalized = v / np.clip(norms, _EPS, None)
    return normalized @ normalized.T


def mean_pairwise_cosine(vectors: np.ndarray, center: bool = False) -> float:
    """余弦類似度行列の非対角（上三角）平均を返す"""
    matrix = cosine_similarity_matrix(vectors, center=center)
    off = _upper_off_diag(matrix)
    return float(off.mean()) if off.size else float("nan")


def linear_cka(a: np.ndarray, b: np.ndarray) -> float:
    """2 つの表現集合 ``[N, D]`` の線形 CKA を返す

    各表現を標本方向に中心化し，``CKA = ||Aᵀ B||_F^2 / (||Aᵀ A||_F ||Bᵀ B||_F)`` を
    計算する標本数 N は 2 つで共通とする（同一スライド集合の倍率対）

    Args:
        a: 表現集合 ``[N, D_a]``
        b: 表現集合 ``[N, D_b]``

    Returns:
        線形 CKA（0–1）標本不足や定数表現なら ``nan``
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if a.shape[0] < _MIN_LAYERS:
        return float("nan")
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    hsic_ab = np.linalg.norm(a.T @ b, ord="fro") ** 2
    norm_a = np.linalg.norm(a.T @ a, ord="fro")
    norm_b = np.linalg.norm(b.T @ b, ord="fro")
    denom = norm_a * norm_b
    if denom < _EPS:
        return float("nan")
    return float(hsic_ab / denom)


def effective_rank(vectors: np.ndarray) -> float:
    """積み上げ行列 ``[L, D]`` の実効ランク（スペクトルエントロピーの指数）を返す

    特異値を正規化して確率分布とみなし ``exp(-Σ p log p)`` を返す全行が共線なら 1 に，
    互いに直交なら ``min(L, D)`` に近づく

    Args:
        vectors: 倍率ベクトル ``[L, D]``

    Returns:
        実効ランク（``1 .. min(L, D)``）
    """
    sv = np.linalg.svd(vectors.astype(np.float64), compute_uv=False)
    total = sv.sum()
    if total < _EPS:
        return float("nan")
    p = sv / total
    p = p[p > _SPECTRUM_EPS]
    entropy = -(p * np.log(p)).sum()
    return float(np.exp(entropy))


def singular_value_spectrum(vectors: np.ndarray) -> np.ndarray:
    """積み上げ行列 ``[L, D]`` の特異値（降順）を返す"""
    return np.linalg.svd(vectors.astype(np.float64), compute_uv=False)


def aggregate_redundancy(slide_vectors: Sequence[np.ndarray]) -> dict:
    """スライドごとの倍率ベクトル ``[L, D]`` 列を集約し冗長性指標をまとめる

    各スライドで余弦類似度（生・中心化）・実効ランクを計算し平均する倍率対ごとの
    線形 CKA と Pearson 相関は，スライドを標本とした表現集合 ``[N, D]`` から計算し
    ``L×L`` 行列にまとめる結果は JSON 化可能なプリミティブで返す

    Args:
        slide_vectors: 各スライドの倍率ベクトル ``[L, D]`` の列（L・D は共通）

    Returns:
        集約指標の辞書空入力や単一倍率なら ``n_slides`` / ``n_layers`` のみ
    """
    if not slide_vectors:
        return {"n_slides": 0, "n_layers": 0}
    num_layers = slide_vectors[0].shape[0]
    vec_dim = slide_vectors[0].shape[1]
    result: dict = {"n_slides": len(slide_vectors), "n_layers": int(num_layers)}
    if num_layers < _MIN_LAYERS:
        return result

    cos_raw = [mean_pairwise_cosine(v, center=False) for v in slide_vectors]
    cos_centered = [mean_pairwise_cosine(v, center=True) for v in slide_vectors]
    eff_rank = [effective_rank(v) for v in slide_vectors]
    spectra = np.stack([singular_value_spectrum(v) for v in slide_vectors], axis=0)

    result["mean_cosine"] = _safe_nanmean(cos_raw)
    result["mean_cosine_centered"] = _safe_nanmean(cos_centered)
    result["mean_effective_rank"] = _safe_nanmean(eff_rank)
    result["max_effective_rank"] = float(min(num_layers, vec_dim))
    result["mean_singular_values"] = np.nanmean(spectra, axis=0).tolist()

    cka, pearson = _pairwise_cka_pearson(slide_vectors, num_layers)
    result["cka_matrix"] = cka.tolist()
    result["pearson_matrix"] = pearson.tolist()
    result["mean_cka"] = _safe_nanmean(_upper_off_diag(cka))
    result["mean_pearson"] = _safe_nanmean(_upper_off_diag(pearson))
    return result


def _pairwise_cka_pearson(
    slide_vectors: Sequence[np.ndarray], num_layers: int
) -> tuple:
    """倍率対の線形 CKA と Pearson 相関の ``L×L`` 行列を返す

    スライドを標本とし，各倍率の表現集合 ``[N, D]`` を作って倍率対ごとに計算する
    """
    stacked = np.stack(slide_vectors, axis=0)
    per_layer = [stacked[:, i, :] for i in range(num_layers)]
    cka = np.eye(num_layers, dtype=np.float64)
    pearson = np.eye(num_layers, dtype=np.float64)
    for i in range(num_layers):
        for j in range(i + 1, num_layers):
            value = linear_cka(per_layer[i], per_layer[j])
            cka[i, j] = cka[j, i] = value
            corr = _flattened_pearson(per_layer[i], per_layer[j])
            pearson[i, j] = pearson[j, i] = corr
    return cka, pearson


def _flattened_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """2 表現集合 ``[N, D]`` を平坦化した Pearson 相関係数を返す"""
    a_flat = a.astype(np.float64).ravel()
    b_flat = b.astype(np.float64).ravel()
    if a_flat.size < _MIN_LAYERS:
        return float("nan")
    a_c = a_flat - a_flat.mean()
    b_c = b_flat - b_flat.mean()
    denom = np.linalg.norm(a_c) * np.linalg.norm(b_c)
    if denom < _EPS:
        return float("nan")
    return float((a_c @ b_c) / denom)


def save_heatmap(
    matrix: np.ndarray,
    labels: Sequence[str],
    title: str,
    out_png: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "viridis",
) -> bool:
    """``L×L`` 行列をヒートマップ PNG に保存する成功で ``True``

    matplotlib が無ければ描かず ``False`` を返す（依存が無くても診断を止めない）
    ``vmin`` / ``vmax`` で値域を，``cmap`` で配色を指定する負値を含む相関には
    ``vmin=-1`` と発散配色を渡す

    Args:
        matrix: ``L×L`` 行列
        labels: 倍率ラベル（軸目盛）
        title: 図タイトル
        out_png: 出力 PNG パス
        vmin: カラースケール下限
        vmax: カラースケール上限
        cmap: matplotlib のカラーマップ名

    Returns:
        描画できたか
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return False

    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax)
    ticks = list(range(len(labels)))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="w")
    ax.set_xlabel("Magnification")
    ax.set_ylabel("Magnification")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return True
