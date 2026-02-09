"""
data_loader.py - PLXデータ読み込みの一元管理

PLXファイルからLFP（低周波）・Wideband（広帯域）・スパイク・イベントデータを
一括で読み込み、後段のGUI/解析モジュールに渡す。
"""
import numpy_compat  # NumPy 2.0+ 互換パッチ（neo/quantities より先に読み込む）
import numpy as np
import os
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any


@dataclass
class PlxData:
    """PLXファイルから読み込んだ全データを格納"""
    filepath: str = ""
    basename: str = ""
    output_dir: str = ""

    # LFPデータ
    lfp_raw: np.ndarray = None          # (n_samples, n_channels)
    lfp_fs: int = 0                      # サンプリングレート (Hz)
    lfp_times: np.ndarray = None         # 時間軸 (秒)

    # Widebandデータ（スパイクソーティング用）
    wideband_raw: np.ndarray = None      # (n_samples, n_channels)
    wideband_fs: int = 0

    # イベント・同期
    segment: Any = None                  # neo.Segment
    frame_times: np.ndarray = None
    stim_times: np.ndarray = None
    session_times: np.ndarray = None

    # 動画情報
    video_file: str = ""
    n_video_frames: int = 0

    # チャンネル情報
    channel_order: List[int] = field(default_factory=lambda:
        [8, 7, 9, 6, 12, 3, 11, 4, 14, 1, 15, 0, 13, 2, 10, 5])
    n_channels: int = 0
    original_ch_numbers: List[int] = field(default_factory=list)

    # Trim範囲
    trim_start: float = 0.0
    trim_end: float = 0.0

    # スパイクデータ（PLXに含まれるソート済みスパイク）
    has_sorted_spikes: bool = False
    spike_trains: list = field(default_factory=list)

    @property
    def duration(self) -> float:
        if self.lfp_times is not None and len(self.lfp_times) > 0:
            return float(self.lfp_times[-1] - self.lfp_times[0])
        return 0.0


def load_plx(filepath: str,
             channel_order: List[int] = None,
             load_wideband: bool = True,
             verbose: bool = True) -> PlxData:
    """
    PLXファイルを読み込み、PlxDataに格納

    Parameters
    ----------
    filepath : str
        PLXファイルパス
    channel_order : list or None
        チャンネル並び替え順（Noneでデフォルト）
    load_wideband : bool
        Widebandデータも読み込むか
    verbose : bool

    Returns
    -------
    PlxData
    """
    import neo

    if channel_order is None:
        channel_order = [8, 7, 9, 6, 12, 3, 11, 4, 14, 1, 15, 0, 13, 2, 10, 5]

    data = PlxData(
        filepath=filepath,
        basename=os.path.splitext(os.path.basename(filepath))[0],
        output_dir=os.path.dirname(filepath),
        channel_order=channel_order,
    )

    log = print if verbose else lambda *a: None
    log(f"[DataLoader] 読み込み: {os.path.basename(filepath)}")

    # Neo で PLX 読み込み
    plx = neo.io.PlexonIO(filename=filepath)
    block = plx.read()[0]
    seg = block.segments[0]
    data.segment = seg

    # ============================================================
    # アナログ信号の振り分け
    # ============================================================
    for i, sig in enumerate(seg.analogsignals):
        fs = int(sig.sampling_rate)
        n_ch = sig.shape[1] if sig.ndim > 1 else 1

        if fs < 5000:
            # LFP (低サンプリングレート)
            raw = np.array(sig)
            if raw.ndim == 1:
                raw = raw[:, np.newaxis]
            # チャンネル並び替え
            n_available = raw.shape[1]
            valid_order = [ch for ch in channel_order if ch < n_available]
            data.lfp_raw = raw[:, valid_order]
            data.lfp_fs = fs
            data.n_channels = data.lfp_raw.shape[1]
            data.lfp_times = np.arange(len(data.lfp_raw)) / fs
            data.original_ch_numbers = valid_order
            log(f"  LFP: {data.lfp_raw.shape}, fs={fs}Hz")

        elif load_wideband and fs >= 20000:
            # Wideband (高サンプリングレート)
            raw = np.array(sig)
            if raw.ndim == 1:
                raw = raw[:, np.newaxis]
            n_available = raw.shape[1]
            valid_order = [ch for ch in channel_order if ch < n_available]
            data.wideband_raw = raw[:, valid_order]
            data.wideband_fs = fs
            log(f"  Wideband: {data.wideband_raw.shape}, fs={fs}Hz")

    # ============================================================
    # イベントデータ
    # ============================================================
    # フレーム同期
    data.frame_times = _get_frame_times(seg.events, verbose=verbose)

    # 刺激イベント
    data.session_times, data.stim_times = _get_stim_events(seg.events, verbose=verbose)

    # ============================================================
    # 動画情報
    # ============================================================
    video_file = os.path.splitext(filepath)[0] + '.mp4'
    if os.path.exists(video_file):
        import cv2
        cap = cv2.VideoCapture(video_file)
        data.n_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        data.video_file = video_file
        log(f"  動画: {data.n_video_frames} フレーム")

        # Trim範囲
        n_sync = min(len(data.frame_times), data.n_video_frames)
        data.frame_times = data.frame_times[:n_sync]
        data.trim_start = float(data.frame_times[0])
        data.trim_end = float(data.frame_times[-1])
        log(f"  Trim: {data.trim_start:.2f}s ~ {data.trim_end:.2f}s")
    else:
        log(f"  動画なし（Trimはデータ全長を使用）")
        if data.lfp_times is not None:
            data.trim_start = float(data.lfp_times[0])
            data.trim_end = float(data.lfp_times[-1])

    # ============================================================
    # スパイクデータ確認
    # ============================================================
    if len(seg.spiketrains) > 0:
        data.has_sorted_spikes = True
        data.spike_trains = seg.spiketrains
        n_sorted = sum(1 for st in seg.spiketrains
                       if st.annotations.get('unit_id', 0) > 0)
        log(f"  スパイク: {len(seg.spiketrains)} trains ({n_sorted} sorted)")

    log(f"[DataLoader] 完了: {data.duration:.1f}秒のデータ")
    return data


# ============================================================
# 内部ヘルパー
# ============================================================

def _get_frame_times(events, min_count=1000, max_interval=1.0, verbose=True):
    """フレーム同期イベント取得"""
    for evt in events:
        if len(evt.times) >= min_count:
            times = np.array(evt.times)
            intervals = np.diff(times)
            large_gaps = np.where(intervals > max_interval)[0]
            if len(large_gaps) > 0 and large_gaps[0] < 10:
                times = times[large_gaps[0] + 1:]
            if verbose:
                print(f"  フレーム同期: {evt.name}, {len(times)}個")
            return times
    # フレーム同期がなければ空
    if verbose:
        print("  フレーム同期: なし")
    return np.array([])


def _get_stim_events(events, session_name="EVT02", stim_name="EVT01", verbose=True):
    """刺激イベント取得"""
    session_times, stim_times = None, None
    for evt in events:
        if len(evt.times) > 0:
            if evt.name == session_name:
                session_times = np.array(evt.times)
            if evt.name == stim_name:
                stim_times = np.array(evt.times)
    if verbose:
        n_stim = len(stim_times) if stim_times is not None else 0
        print(f"  刺激イベント: {n_stim}個")
    return session_times, stim_times
