"""
numpy_compat.py - NumPy 2.0+ 互換性パッチ

NumPy 2.0 で削除された ndarray.ptp を復元し、
quantities / neo パッケージとの互換性を維持する。

使い方:
    import numpy_compat   # neo をインポートする前に実行
    import neo
"""
import numpy as np

def _apply_ptp_patch():
    """np.ndarray.ptp が存在しない場合にモンキーパッチを適用"""
    if not hasattr(np.ndarray, 'ptp'):
        def _ptp(self, axis=None, out=None, keepdims=False):
            return np.ptp(self, axis=axis, out=out, keepdims=keepdims)
        np.ndarray.ptp = _ptp
        print("[numpy_compat] np.ndarray.ptp パッチ適用済み (NumPy 2.0+ 対応)")

_apply_ptp_patch()
