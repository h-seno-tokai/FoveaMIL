"""``foveamil-visualize`` コマンド

サブコマンド ``overview`` / ``zoom`` / ``compare`` をディスパッチする薄い層
sweep 出力（``--sweep-root`` の best_by_val）から combo・モデル・症例を解決し，保存済み
予測で成功/失敗を選び，WSI に attention を重ねた図を出す内部パス・ホスト名は env/引数で
受ける``--dry-run`` は解決した combo/fold/モデル設定/症例だけ表示して描画しない
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional, Sequence

from foveamil.visualization import loader as loader_mod
from foveamil.visualization import visualize as viz
from foveamil.visualization.builders import zoom as zoom_mod
from foveamil.visualization.render.palette import DIM_FACTOR

logger = logging.getLogger(__name__)

# サブコマンド名と実行関数の対応
_RUNNERS = {
    viz.VIEW_OVERVIEW: viz.run_overview,
    viz.VIEW_ZOOM: viz.run_zoom,
    viz.VIEW_COMPARE: viz.run_compare,
}


def _require_matplotlib() -> None:
    """描画が主目的なので matplotlib 不在は明示エラーにする"""
    try:
        import matplotlib  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "matplotlib が必要です（pip install -e . で導入）: " + str(exc)
        )


def _spec_from_args(args: argparse.Namespace) -> viz.VizSpec:
    """argparse の Namespace から :class:`VizSpec` を組む"""
    return viz.VizSpec(
        sweep_root=args.sweep_root,
        select=args.select,
        combo_index=args.combo_index,
        combo_dir=args.combo_dir,
        fold=args.fold,
        weights_root=args.weights_root,
        split=args.split,
        outcome=args.outcome,
        slide_id=args.slide_id,
        n=args.n,
        per_class=args.per_class,
        target_class=args.target_class,
        feature_root=args.feature_root,
        coords_root=args.coords_root,
        wsi_base_path=args.wsi_base_path,
        wsi_overrides_csv=args.wsi_overrides_csv,
        out_dir=args.out_dir,
        device=args.device,
        dpi=args.dpi,
        thumb_mag=getattr(args, "thumb_mag", None),
        norm=getattr(args, "norm", "percentile"),
        parent_mag=getattr(args, "parent_mag", None),
        parent_pick=getattr(args, "parent_pick", zoom_mod.PICK_TOP_AUX),
        n_parents=getattr(args, "n_parents", 4),
        zoom_px=getattr(args, "zoom_px", zoom_mod.DEFAULT_ZOOM_PX),
        dim_factor=getattr(args, "dim_factor", DIM_FACTOR),
        chain=getattr(args, "chain", False),
    )


def _dry_run(spec: viz.VizSpec) -> int:
    """解決した combo/fold/モデル設定/症例だけ表示する"""
    ctx = viz._resolve_context(spec)
    cases = viz._select_slides(spec, ctx)
    print(f"combo_dir : {ctx.combo_dir}")
    print(f"fold      : {spec.fold}  weights: {ctx.weights_dir or ctx.fold_dir}")
    print(f"encoder   : {ctx.loaded.encoder}  feature_type: {ctx.loaded.feature_type}")
    print(f"mags      : {ctx.loaded.magnifications}  n_cls: {ctx.loaded.n_cls}")
    print(f"classes   : {ctx.loaded.classes}")
    print(f"症例数    : {len(cases)}")
    for c in cases:
        print(f"  {c.slide_id}  GT:{c.y_true} 予測:{c.y_pred} correct:{c.correct}")
    return 0


def run(args: argparse.Namespace) -> int:
    """サブコマンドを実行する"""
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    spec = _spec_from_args(args)
    if args.dry_run:
        return _dry_run(spec)
    _require_matplotlib()
    saved = _RUNNERS[args.command](spec)
    logger.info("wrote %d figure(s) to %s", len(saved), args.out_dir)
    return 0


def _add_common(parser: argparse.ArgumentParser) -> None:
    """サブコマンド共通の引数を足す"""
    parser.add_argument("--sweep-root", default=None, help="sweep の出力ルート（best_by_val を読む）")
    parser.add_argument("--select", default=loader_mod.SELECT_BEST_VAL,
                        choices=[loader_mod.SELECT_BEST_VAL, loader_mod.SELECT_ORACLE, loader_mod.SELECT_INDEX])
    parser.add_argument("--combo-index", type=int, default=None, help="select=index 時の combo 連番")
    parser.add_argument("--combo-dir", default=None, help="combo を直接指定（best 自動選択を上書き）")
    parser.add_argument("--fold", default="1", help="可視化に使う fold 番号")
    parser.add_argument("--weights-root", default=None, help="重み（.pt）のルート（Dataset 側）")
    parser.add_argument("--split", default="test", choices=["val", "test", "train"])
    parser.add_argument("--outcome", default="both", choices=["success", "failure", "both"])
    parser.add_argument("--slide-id", nargs="+", default=None, help="症例を明示（指定時は自動抽出しない）")
    parser.add_argument("--n", type=int, default=None, help="抽出する総件数")
    parser.add_argument("--per-class", type=int, default=None, help="正解クラスごとの件数")
    parser.add_argument("--target-class", type=int, default=None, help="正解クラスで絞る")
    parser.add_argument("--feature-root", required=True, help="{encoder}/{mag}x/{slide}.h5 のルート")
    parser.add_argument("--coords-root", default=None, help="actual_max_mag を取る座標ルート")
    parser.add_argument("--wsi-base-path", default=None, help="未指定時 env WSI_BASE_PATH")
    parser.add_argument("--wsi-overrides-csv", default=None, help="slide_id,path の CSV")
    parser.add_argument("--out-dir", required=True, help="図の保存先（home）")
    parser.add_argument("--device", default="cpu", help="cpu / cuda:N")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true", help="解決結果だけ表示し描画しない")
    parser.add_argument("--verbose", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    """``foveamil-visualize`` の引数パーサを構築する"""
    parser = argparse.ArgumentParser(
        prog="foveamil-visualize",
        description="Render FoveaMIL attention figures (overview / zoom / compare) "
        "from a sweep output, with no retraining.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_over = sub.add_parser(viz.VIEW_OVERVIEW, help="WSI 全体オーバーレイ格子")
    _add_common(p_over)
    p_over.add_argument("--thumb-mag", type=float, default=None, help="サムネ倍率（既定 最低倍率）")
    p_over.add_argument("--norm", default="percentile", choices=["percentile", "minmax", "raw"])

    p_zoom = sub.add_parser(viz.VIEW_ZOOM, help="階層ズーム照明（40倍課題の解）")
    _add_common(p_zoom)
    p_zoom.add_argument("--parent-mag", type=float, default=None, help="拡大する親倍率（既定 最終の1つ下）")
    p_zoom.add_argument("--parent-pick", default=zoom_mod.PICK_TOP_AUX,
                        choices=[zoom_mod.PICK_TOP_AUX, zoom_mod.PICK_TOP_PRIMARY, zoom_mod.PICK_INDEX])
    p_zoom.add_argument("--n-parents", type=int, default=4)
    p_zoom.add_argument("--zoom-px", type=int, default=zoom_mod.DEFAULT_ZOOM_PX)
    p_zoom.add_argument("--dim-factor", type=float, default=DIM_FACTOR)
    p_zoom.add_argument("--chain", action="store_true", help="多段ズーム連鎖（中心窩経路図）")

    p_cmp = sub.add_parser(viz.VIEW_COMPARE, help="成功 vs 失敗 対比格子")
    _add_common(p_cmp)
    p_cmp.add_argument("--thumb-mag", type=float, default=None)
    p_cmp.add_argument("--norm", default="percentile", choices=["percentile", "minmax", "raw"])

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``foveamil-visualize`` コンソールスクリプトのエントリポイント"""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
