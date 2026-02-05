"""
data_loader.py - PLXファイルからの統一データ読み込み

全データストリーム（LFP, wideband, events）を一元管理する
RecordingSession データクラスを提供。

使い方:
    session = load_plx_session("data.plx", channel_order=[8,7,9,...])
    print(session)
    print(session.stim_summary())
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
import os


# ============================================================
# データクラス
# ============================================================

@dataclass
class RecordingSession:
    """
    1つのPLX記録セッションの全データを保持
    
    Attributes
    ----------
    filename : str
        PLXファイルパス
    basename : str
        ファイル名（拡張子なし）
    
    lfp_raw : ndarray (n_samples, n_channels)
        生LFPデータ（チャンネル順序適用済み）
    fs_lfp : int
        LFPサンプリングレート (通常 1000 Hz)
    
    wideband : ndarray (n_samples, n_channels)
        広帯域データ（チャンネル順序適用済み）
    fs_wideband : int
        広帯域サンプリングレート (通常 40000 Hz)
    
    analog_input : ndarray or None
        アナログ入力チャンネル
    fs_ai : int
        アナログ入力サンプリングレート
    
    stim_times : ndarray
        各刺激のタイムスタンプ（秒）
    trial_starts : ndarray
        各trialの開始タイムスタンプ（秒）
    frame_times : ndarray
        カメラフレーム同期タイムスタンプ（秒）
    
    channel_order : list
        物理→論理チャンネルマッピング
    duration : float
        記録の全長（秒）
    
    n_trials : int
        trial数
    n_stim_per_trial : int
        trial当たりの刺激数
    stim_freq : float
        刺激周波数 (Hz)
    iti : float
        Inter-trial interval (秒)
    """
    # ファイル情報
    filename: str = ""
    basename: str = ""
    
    # LFPデータ
    lfp_raw: np.ndarray = field(default_factory=lambda: np.array([]))
    fs_lfp: int = 1000
    
    # 広帯域データ
    wideband: np.ndarray = field(default_factory=lambda: np.array([]))
    fs_wideband: int = 40000
    
    # アナログ入力
    analog_input: Optional[np.ndarray] = None
    fs_ai: int = 1000
    
    # イベント
    stim_times: np.ndarray = field(default_factory=lambda: np.array([]))
    trial_starts: np.ndarray = field(default_factory=lambda: np.array([]))
    frame_times: np.ndarray = field(default_factory=lambda: np.array([]))
    all_events: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # チャンネル設定
    channel_order: List[int] = field(default_factory=list)
    
    # メタデータ
    duration: float = 0.0
    n_trials: int = 0
    n_stim_per_trial: int = 0
    stim_freq: float = 0.0
    iti: float = 0.0
    
    # Neo segmentへの参照（追加解析用）
    _segment: Any = field(default=None, repr=False)
    
    def __repr__(self):
        lines = [
            f"=== RecordingSession: {self.basename} ===",
            f"  Duration: {self.duration:.1f}s",
            f"  LFP:      {self.lfp_raw.shape} @ {self.fs_lfp} Hz",
            f"  Wideband: {self.wideband.shape} @ {self.fs_wideband} Hz",
            f"  Channels: {len(self.channel_order)} (order: {self.channel_order})",
            f"  Stimulus: {len(self.stim_times)} events "
            f"({self.n_trials} trials × {self.n_stim_per_trial} stim @ {self.stim_freq:.0f}Hz)",
            f"  ITI:      {self.iti:.1f}s",
            f"  Frames:   {len(self.frame_times)} sync events",
        ]
        return "\n".join(lines)
    
    @property
    def n_channels(self) -> int:
        """チャンネル数"""
        if self.lfp_raw.ndim == 2:
            return self.lfp_raw.shape[1]
        return 0
    
    @property
    def lfp_times(self) -> np.ndarray:
        """LFP時間軸（秒）"""
        return np.arange(self.lfp_raw.shape[0]) / self.fs_lfp
    
    @property
    def wideband_times(self) -> np.ndarray:
        """広帯域時間軸（秒）"""
        return np.arange(self.wideband.shape[0]) / self.fs_wideband
    
    @property
    def stim_sessions(self) -> np.ndarray:
        """刺激をtrial×stim行列に整形"""
        if self.n_trials > 0 and self.n_stim_per_trial > 0:
            return self.stim_times.reshape(self.n_trials, self.n_stim_per_trial)
        return self.stim_times
    
    @property
    def trial_ranges(self) -> List[Tuple[float, float]]:
        """各trialの(開始, 終了)時刻"""
        sessions = self.stim_sessions
        if sessions.ndim == 2:
            return [(s[0], s[-1]) for s in sessions]
        return []
    
    def get_video_path(self) -> str:
        """対応する動画ファイルパス"""
        return os.path.splitext(self.filename)[0] + '.mp4'
    
    def get_output_dir(self) -> str:
        """出力ディレクトリ"""
        return os.path.dirname(self.filename)


# ============================================================
# データ読み込み
# ============================================================

def load_plx_session(
    plx_file: str,
    channel_order: List[int] = None,
    stim_event: str = "EVT01",
    trial_event: str = "EVT02",
    frame_min_count: int = 1000,
    verbose: bool = True
) -> RecordingSession:
    """
    PLXファイルから全データを読み込み、RecordingSessionを返す
    
    Parameters
    ----------
    plx_file : str
        PLXファイルパス
    channel_order : list or None
        チャンネル並び順。Noneの場合はデフォルト（A1x16-5mm-50-703）
    stim_event : str
        刺激イベントのチャンネル名
    trial_event : str
        trialイベントのチャンネル名
    frame_min_count : int
        フレーム同期イベントの最小数（これ以上あるイベントを自動検出）
    verbose : bool
        詳細出力
    
    Returns
    -------
    session : RecordingSession
    """
    import neo
    
    if channel_order is None:
        # NeuroNexus A1x16-5mm-50-703 デフォルト
        channel_order = [8, 7, 9, 6, 12, 3, 11, 4, 14, 1, 15, 0, 13, 2, 10, 5]
    
    log = print if verbose else lambda *a, **kw: None
    log(f"=== Loading: {os.path.basename(plx_file)} ===")
    
    # PLX読み込み
    plx = neo.io.PlexonIO(filename=plx_file)
    data = plx.read()
    seg = data[0].segments[0]
    
    session = RecordingSession(
        filename=plx_file,
        basename=os.path.splitext(os.path.basename(plx_file))[0],
        channel_order=channel_order,
        _segment=seg,
    )
    
    # --------------------------------------------------------
    # アナログ信号を分類
    # --------------------------------------------------------
    for i, sig in enumerate(seg.analogsignals):
        fs = int(sig.sampling_rate)
        shape = sig.shape
        stream_id = sig.annotations.get('stream_id', '')
        
        log(f"  Signal {i}: {stream_id}, shape={shape}, fs={fs}Hz")
        
        if stream_id == 'FP' or (fs == 1000 and shape[1] >= 16):
            # LFP
            session.lfp_raw = np.array(sig[:, channel_order])
            session.fs_lfp = fs
            log(f"    → LFP: {session.lfp_raw.shape}")
            
        elif stream_id == 'SPKC' or (fs >= 20000 and shape[1] >= 16):
            # Wideband
            session.wideband = np.array(sig[:, channel_order])
            session.fs_wideband = fs
            log(f"    → Wideband: {session.wideband.shape}")
            
        elif stream_id == 'AI' or shape[1] <= 4:
            # Analog Input
            session.analog_input = np.array(sig)
            session.fs_ai = fs
            log(f"    → Analog Input: {session.analog_input.shape}")
    
    session.duration = len(session.lfp_raw) / session.fs_lfp
    
    # --------------------------------------------------------
    # イベント解析
    # --------------------------------------------------------
    log(f"\n  Events:")
    
    for evt in seg.events:
        name = evt.name
        times = np.array(evt.times).astype(float)
        n = len(times)
        
        if n == 0:
            continue
        
        session.all_events[name] = times
        log(f"    {name}: {n} events ({times[0]:.1f}s - {times[-1]:.1f}s)")
        
        # 刺激イベント
        if name == stim_event:
            session.stim_times = times
            log(f"      → Stimulus events")
        
        # Trialイベント
        elif name == trial_event:
            session.trial_starts = times
            log(f"      → Trial start events")
        
        # フレーム同期（自動検出: min_count以上のイベント）
        elif n >= frame_min_count:
            session.frame_times = _clean_frame_times(times)
            log(f"      → Frame sync ({len(session.frame_times)} after cleanup)")
    
    # --------------------------------------------------------
    # 刺激構造の解析
    # --------------------------------------------------------
    if len(session.stim_times) > 1:
        _analyze_stim_structure(session, verbose)
    
    log(f"\n{session}")
    return session


def _clean_frame_times(times: np.ndarray, max_interval: float = 1.0) -> np.ndarray:
    """フレーム同期タイムスタンプのクリーンアップ（初期の不安定部分を除去）"""
    intervals = np.diff(times)
    large_gaps = np.where(intervals > max_interval)[0]
    if len(large_gaps) > 0 and large_gaps[0] < 10:
        times = times[large_gaps[0] + 1:]
    return times


def _analyze_stim_structure(session: RecordingSession, verbose: bool = True):
    """刺激タイミングからtrial構造を推定"""
    log = print if verbose else lambda *a, **kw: None
    
    stim_isi_ms = np.diff(session.stim_times) * 1000
    median_isi = np.median(stim_isi_ms)
    
    # Trial境界の検出（ISIが中央値の3倍以上をギャップとする）
    gap_threshold = max(median_isi * 3, 500)  # 最低500ms
    long_gaps = np.where(stim_isi_ms > gap_threshold)[0]
    
    n_trials = len(long_gaps) + 1
    n_stim_total = len(session.stim_times)
    
    if n_trials > 1:
        n_stim_per_trial = n_stim_total // n_trials
        stim_freq = 1000.0 / median_isi  # Hz
        iti_values = stim_isi_ms[long_gaps] / 1000.0  # 秒に変換
        iti = float(np.mean(iti_values))
    else:
        n_stim_per_trial = n_stim_total
        stim_freq = 1000.0 / median_isi if median_isi > 0 else 0
        iti = 0.0
    
    session.n_trials = n_trials
    session.n_stim_per_trial = n_stim_per_trial
    session.stim_freq = stim_freq
    session.iti = iti
    
    # trial_startsが空の場合、推定する
    if len(session.trial_starts) == 0 and n_trials > 1:
        trial_starts = [session.stim_times[0]]
        for gap_idx in long_gaps:
            trial_starts.append(session.stim_times[gap_idx + 1])
        session.trial_starts = np.array(trial_starts)
        log(f"  Trial starts estimated from stimulus gaps: {n_trials} trials")
    
    log(f"\n  Stimulus structure:")
    log(f"    {n_trials} trials × {n_stim_per_trial} stim @ {stim_freq:.1f}Hz")
    log(f"    ITI: {iti:.1f}s")
    log(f"    Stim ISI: {median_isi:.0f}ms (median)")


# ============================================================
# ユーティリティ
# ============================================================

def load_sorting_results(filepath: str) -> Dict[int, Any]:
    """
    保存済みのスパイクソーティング結果を読み込む
    
    Parameters
    ----------
    filepath : str
        save_sorting_results()で保存したNPZファイル
    
    Returns
    -------
    results : dict
        {channel: ChannelSortResult}
    
    Notes
    -----
    spike_sorting.pyのsave_sorting_results()と対応。
    NPZキー形式: ch{N}_unit{M}_times, ch{N}_unit{M}_waveforms, etc.
    """
    from spike_sorting import ChannelSortResult, SpikeUnit
    
    data = np.load(filepath, allow_pickle=True)
    
    # チャンネル一覧を抽出
    channels = set()
    for key in data.files:
        if key.endswith('_fs'):
            ch = int(key.replace('ch', '').replace('_fs', ''))
            channels.add(ch)
    
    results = {}
    for ch in sorted(channels):
        result = ChannelSortResult(
            channel=ch,
            fs=float(data[f'ch{ch}_fs']),
            threshold=float(data[f'ch{ch}_threshold']),
            sigma=float(data[f'ch{ch}_sigma']),
        )
        
        # 全スパイク情報
        if f'ch{ch}_all_indices' in data:
            result.all_spike_indices = data[f'ch{ch}_all_indices']
            result.all_waveforms = data[f'ch{ch}_all_waveforms']
            result.all_pca_features = data[f'ch{ch}_all_pca']
            result.labels = data[f'ch{ch}_labels']
            result.waveform_time_ms = data[f'ch{ch}_waveform_time']
        
        # ユニット
        unit_ids = set()
        for key in data.files:
            if key.startswith(f'ch{ch}_unit') and key.endswith('_indices'):
                uid = int(key.replace(f'ch{ch}_unit', '').replace('_indices', ''))
                unit_ids.add(uid)
        
        for uid in sorted(unit_ids):
            prefix = f'ch{ch}_unit{uid}_'
            unit = SpikeUnit(
                unit_id=uid,
                spike_indices=data[prefix + 'indices'],
                spike_times=data[prefix + 'times'],
                waveforms=data[prefix + 'waveforms'],
                pca_features=data[prefix + 'pca'],
                is_noise=bool(data[prefix + 'is_noise']),
                is_mua=bool(data[prefix + 'is_mua']),
                snr=float(data[prefix + 'snr']),
                isi_violation_rate=float(data[prefix + 'isi_viol']),
            )
            result.units.append(unit)
        
        results[ch] = result
    
    print(f"Loaded sorting results: {len(results)} channels, "
          f"{sum(len(r.units) for r in results.values())} units")
    
    return results
