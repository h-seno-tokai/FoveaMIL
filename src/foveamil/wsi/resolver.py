"""``slide_id`` を WSI ファイルの絶対パスに解決する

解決は次の順に試みる:
  1. オーバーライド表（``slide_id,path`` の CSV）に該当があればそのパスを使う
  2. なければルートディレクトリ配下を ``glob`` して ``{slide_id}.{ext}`` を探す
     （OpenSlide 対応拡張子再帰探索に対応）
  3. 0 件・複数ヒットは ``slide_id`` と探索条件を含む例外を送出する

ルートディレクトリは引数で渡す未指定時は環境変数 ``WSI_BASE_PATH`` を使う
WSI が複数のディレクトリに分かれている場合は ``os.pathsep``（Linux では ``:``）区切りで
複数ルートを与えられる全ルートを横断して探索し，一意に決まらなければ例外を送出する
"""

from __future__ import annotations

import glob
import os
from typing import Dict, Iterable, Optional

import pandas as pd

# OpenSlide が開ける代表的な WSI 拡張子（小文字・先頭ドット無し）
# 探索順ではなく集合判定にのみ使う
SUPPORTED_WSI_EXTENSIONS: tuple[str, ...] = (
    "svs",
    "tif",
    "tiff",
    "ndpi",
    "mrxs",
    "scn",
    "svslide",
    "vms",
    "vmu",
    "bif",
)

# 既定のルートディレクトリを与える環境変数名
WSI_BASE_PATH_ENV = "WSI_BASE_PATH"


def _split_base_paths(base_path: Optional[object]) -> list[str]:
    """``base_path`` を探索ルートのリストに正規化する

    文字列は ``os.pathsep`` 区切りで複数ルートとして解釈するリスト・タプルは
    各要素をルートとする空要素は捨てる``None`` は空リストにする
    """
    if base_path is None:
        return []
    if isinstance(base_path, str):
        parts = base_path.split(os.pathsep)
    else:
        parts = list(base_path)
    return [p for p in (str(x).strip() for x in parts) if p]


class WSIResolutionError(RuntimeError):
    """``slide_id`` に対応する WSI が一意に解決できなかったときの例外"""


class WSIResolver:
    """``slide_id`` を WSI ファイルの絶対パスへ解決する

    Args:
        base_path: WSI 置き場のルートディレクトリ``os.pathsep`` 区切りの文字列か
            文字列のリストで複数ルートを与えられる``None`` の場合は環境変数
            ``WSI_BASE_PATH`` を使うオーバーライド表だけで全件解決できるなら省略可
        overrides: ``slide_id -> 絶対パス`` の辞書ここに載っている slide_id は
            ファイル探索より優先される
        recursive: 各ルート配下をサブディレクトリまで再帰探索するか（既定 True）
    """

    def __init__(
        self,
        base_path: Optional[object] = None,
        overrides: Optional[Dict[str, str]] = None,
        recursive: bool = True,
    ) -> None:
        if base_path is None:
            base_path = os.environ.get(WSI_BASE_PATH_ENV)
        self.base_paths: list[str] = _split_base_paths(base_path)
        self.overrides: Dict[str, str] = dict(overrides) if overrides else {}
        self.recursive = recursive

    @classmethod
    def from_overrides_csv(
        cls,
        csv_path: str,
        base_path: Optional[str] = None,
        recursive: bool = True,
    ) -> "WSIResolver":
        """``slide_id,path`` 列を持つ CSV からオーバーライド表を読んで生成する

        Args:
            csv_path: ``slide_id,path`` 列を持つ CSV のパス
            base_path: WSI 置き場ルート（オーバーライドに無い slide_id 用）
            recursive: 再帰探索するか

        Returns:
            オーバーライド表を読み込んだ :class:`WSIResolver`
        """
        df = pd.read_csv(csv_path)
        missing = {"slide_id", "path"} - set(df.columns)
        if missing:
            raise ValueError(
                f"overrides CSV {csv_path!r} must have columns 'slide_id,path' "
                f"(missing: {sorted(missing)})"
            )
        overrides = {
            str(row.slide_id): str(row.path) for row in df.itertuples(index=False)
        }
        return cls(base_path=base_path, overrides=overrides, recursive=recursive)

    def resolve(self, slide_id: str) -> str:
        """1 件の ``slide_id`` を WSI の絶対パスに解決する

        Args:
            slide_id: 拡張子なしの WSI ベース名（例 ``"SAMPLE_0001"``）

        Returns:
            解決された WSI ファイルの絶対パス

        Raises:
            WSIResolutionError: 該当ファイルが 0 件，または複数ヒットした場合
        """
        if slide_id in self.overrides:
            path = self.overrides[slide_id]
            if not os.path.isfile(path):
                raise WSIResolutionError(
                    f"override path for slide_id {slide_id!r} does not exist: {path!r}"
                )
            return os.path.abspath(path)

        if not self.base_paths:
            raise WSIResolutionError(
                f"cannot resolve slide_id {slide_id!r}: no override given and "
                f"base_path is unset (pass base_path or set ${WSI_BASE_PATH_ENV})"
            )

        matches = self._glob_matches(slide_id)
        unique = sorted(set(matches))
        if len(unique) == 0:
            raise WSIResolutionError(
                f"no WSI file found for slide_id {slide_id!r} under {self.base_paths!r} "
                f"(searched extensions: {', '.join(SUPPORTED_WSI_EXTENSIONS)}; "
                f"recursive={self.recursive})"
            )
        if len(unique) > 1:
            raise WSIResolutionError(
                f"multiple WSI files match slide_id {slide_id!r} under "
                f"{self.base_paths!r}: {unique}"
            )
        return os.path.abspath(unique[0])

    def resolve_many(self, slide_ids: Iterable[str]) -> Dict[str, str]:
        """複数の ``slide_id`` を解決し，``slide_id -> パス`` の辞書を返す

        1 件でも解決に失敗するとその場で :class:`WSIResolutionError` を送出する
        失敗をスキップしたい場合は呼び出し側で 1 件ずつ :meth:`resolve` を使う

        Args:
            slide_ids: 解決したい slide_id の列

        Returns:
            入力順を保った ``slide_id -> 絶対パス`` の辞書
        """
        resolved: Dict[str, str] = {}
        for slide_id in slide_ids:
            resolved[slide_id] = self.resolve(slide_id)
        return resolved

    def _glob_matches(self, slide_id: str) -> list[str]:
        """全ルート配下で ``{slide_id}.{ext}`` に一致するファイルを列挙する"""
        matches: list[str] = []
        for root in self.base_paths:
            for ext in SUPPORTED_WSI_EXTENSIONS:
                if self.recursive:
                    pattern = os.path.join(root, "**", f"{slide_id}.{ext}")
                    matches.extend(glob.glob(pattern, recursive=True))
                else:
                    pattern = os.path.join(root, f"{slide_id}.{ext}")
                    matches.extend(glob.glob(pattern))
        return matches
