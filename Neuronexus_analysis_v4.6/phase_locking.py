"""
phase_locking.py - 位相ロック解析の中核関数

スパイク-LFP位相ロックの計算、統計検定、解析ワークフロー
"""
import numpy as np
from scipy import signal
from scipy.signal import hilbert
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass


# ============================================================
# データクラス
# ============================================================

@dataclass
class PhaseLockingResult:
    """位相ロック解析結果を格納"""
    mrl: float                    # Mean Resultant Length
    ppc: float                    # Pairwise Phase Consistency
    preferred_phase: float        # 平均位相（ラジアン）
    z_stat: float                 # Rayleigh統計量
    p_value: float                # p値
    n_spikes: int                 # スパイク数
    spike_phases: np.ndarray      # 各スパイクの位相
    significant: bool             # 有意かどうか (p < 0.05)
    
    @property
    def preferred_phase_deg(self) -> float:
        """平均位相を度数法で返す"""
        return np.degrees(self.preferred_phase)


# ============================================================
# LFP位相抽出
# ============================================================

def extract_instantaneous_phase(
    lfp_data: np.ndarray,
    fs: int,
    freq_band: Tuple[float, float],
    filter_order: int = 4
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    LFPから瞬時位相を抽出
    
    Parameters
    ----------
    lfp_data : np.ndarray (n_samples,) or (n_samples, n_channels)
        LFPデータ
    fs : int
        サンプリング周波数
    freq_band : tuple (low, high)
        周波数帯域 (Hz)
    filter_order : int
        バンドパスフィルタの次数
    
    Returns
    -------
    phase : np.ndarray
        瞬時位相 (-π to π)
    amplitude : np.ndarray
        瞬時振幅
    filtered_lfp : np.ndarray
        フィルタ後のLFP
    
    Notes
    -----
    - filtfiltを使用してゼロ位相フィルタリング
    - ヒルベルト変換で解析信号を生成
    """
    nyq = 0.5 * fs
    low, high = freq_band
    
    # 入力を1Dに変換（必要に応じて）
    is_1d = lfp_data.ndim == 1
    if is_1d:
        lfp_data = lfp_data[:, np.newaxis]
    
    # バンドパスフィルタ（Butterworth, ゼロ位相）
    sos = signal.butter(filter_order, [low/nyq, high/nyq], btype='bandpass', output='sos')
    filtered_lfp = signal.sosfiltfilt(sos, lfp_data, axis=0)
    
    # ヒルベルト変換（チャンネルごと）
    n_samples, n_channels = filtered_lfp.shape
    phase = np.zeros_like(filtered_lfp)
    amplitude = np.zeros_like(filtered_lfp)
    
    for ch in range(n_channels):
        analytic_signal = hilbert(filtered_lfp[:, ch])
        phase[:, ch] = np.angle(analytic_signal)
        amplitude[:, ch] = np.abs(analytic_signal)
    
    # 1D入力の場合は1Dで返す
    if is_1d:
        phase = phase.flatten()
        amplitude = amplitude.flatten()
        filtered_lfp = filtered_lfp.flatten()
    
    return phase, amplitude, filtered_lfp


def get_spike_phases(
    spike_times: np.ndarray,
    lfp_phase: np.ndarray,
    lfp_times: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    各スパイク時点でのLFP位相を取得
    
    Parameters
    ----------
    spike_times : np.ndarray
        スパイクタイムスタンプ（秒）
    lfp_phase : np.ndarray
        LFPの瞬時位相
    lfp_times : np.ndarray
        LFPタイムスタンプ
    
    Returns
    -------
    spike_phases : np.ndarray
        各スパイク時点での位相
    valid_mask : np.ndarray (bool)
        有効なスパイクのマスク（時間範囲内）
    """
    # 時間範囲チェック
    t_start, t_end = lfp_times[0], lfp_times[-1]
    valid_mask = (spike_times >= t_start) & (spike_times <= t_end)
    valid_spikes = spike_times[valid_mask]
    
    if len(valid_spikes) == 0:
        return np.array([]), valid_mask
    
    # 最近傍補間でLFPインデックスを取得
    spike_indices = np.searchsorted(lfp_times, valid_spikes)
    
    # 境界処理
    spike_indices = np.clip(spike_indices, 0, len(lfp_phase) - 1)
    
    spike_phases = lfp_phase[spike_indices]
    
    return spike_phases, valid_mask


# ============================================================
# 位相ロック指標
# ============================================================

def compute_mean_resultant_length(phases: np.ndarray) -> Tuple[float, float]:
    """
    Mean Resultant Length (MRL) を計算
    
    Parameters
    ----------
    phases : np.ndarray
        位相データ（ラジアン）
    
    Returns
    -------
    mrl : float
        MRL値 (0-1)
    preferred_phase : float
        平均位相（ラジアン）
    
    Formula
    -------
    MRL = |1/n × Σ e^(iφₖ)|
    """
    if len(phases) == 0:
        return 0.0, 0.0
    
    # 単位円上のベクトルの平均
    mean_vector = np.mean(np.exp(1j * phases))
    mrl = np.abs(mean_vector)
    preferred_phase = np.angle(mean_vector)
    
    return float(mrl), float(preferred_phase)


def compute_pairwise_phase_consistency(phases: np.ndarray) -> float:
    """
    Pairwise Phase Consistency (PPC) を計算
    
    バイアスの少ない位相ロック指標（Vinck et al., 2010）
    
    Parameters
    ----------
    phases : np.ndarray
    
    Returns
    -------
    ppc : float
        PPC値
    
    Formula
    -------
    PPC = (Σᵢ Σⱼ cos(φᵢ - φⱼ)) / (n(n-1)/2)
    
    Notes
    -----
    PPCはスパイク数に依存しない、より安定した指標
    """
    n = len(phases)
    if n < 2:
        return 0.0
    
    # 効率的な計算: Σcos(φᵢ-φⱼ) = (Σcos(φ))² + (Σsin(φ))² - n
    sum_cos = np.sum(np.cos(phases))
    sum_sin = np.sum(np.sin(phases))
    
    # PPCの公式
    ppc = (sum_cos**2 + sum_sin**2 - n) / (n * (n - 1))
    
    return float(ppc)


def rayleigh_test(phases: np.ndarray) -> Tuple[float, float, float]:
    """
    Rayleigh検定 - 位相分布の一様性を検定
    
    H0: 位相は一様分布（位相ロックなし）
    
    Parameters
    ----------
    phases : np.ndarray
    
    Returns
    -------
    mrl : float
        Mean Resultant Length
    z_stat : float
        Rayleigh統計量 z = n × MRL²
    p_value : float
        p値
    
    Notes
    -----
    p < 0.05 で位相分布が一様でない（位相ロックあり）と判断
    """
    n = len(phases)
    if n == 0:
        return 0.0, 0.0, 1.0
    
    mrl, _ = compute_mean_resultant_length(phases)
    
    # Rayleigh統計量
    R = n * mrl
    z = R ** 2 / n
    
    # p値の近似計算（Mardia & Jupp, 2000）
    # 小さいnでも適用可能な修正版
    if n < 50:
        # 小標本での修正
        p_value = np.exp(-z) * (1 + (2*z - z**2) / (4*n) - 
                               (24*z - 132*z**2 + 76*z**3 - 9*z**4) / (288*n**2))
    else:
        # 大標本近似
        p_value = np.exp(-z)
    
    # p値を0-1に制限
    p_value = np.clip(p_value, 0, 1)
    
    return float(mrl), float(z), float(p_value)


def compute_phase_locking_value(
    lfp_phase1: np.ndarray,
    lfp_phase2: np.ndarray
) -> float:
    """
    Phase Locking Value (PLV) - LFP間の位相同期
    
    Parameters
    ----------
    lfp_phase1, lfp_phase2 : np.ndarray
        2つのLFPチャンネルの瞬時位相
    
    Returns
    -------
    plv : float
        PLV値 (0-1)
    
    Notes
    -----
    PLV = |mean(e^(i*(φ1 - φ2)))|
    """
    if len(lfp_phase1) != len(lfp_phase2):
        raise ValueError("Phase arrays must have same length")
    
    phase_diff = lfp_phase1 - lfp_phase2
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
    
    return float(plv)


def compute_coherence(
    lfp1: np.ndarray,
    lfp2: np.ndarray,
    fs: int,
    nperseg: int = 1024
) -> Tuple[np.ndarray, np.ndarray]:
    """
    LFP間のコヒーレンスを計算
    
    Parameters
    ----------
    lfp1, lfp2 : np.ndarray
        2つのLFPチャンネル
    fs : int
        サンプリング周波数
    nperseg : int
        セグメント長
    
    Returns
    -------
    freqs : np.ndarray
        周波数
    coherence : np.ndarray
        コヒーレンス値
    """
    freqs, coherence = signal.coherence(lfp1, lfp2, fs=fs, nperseg=nperseg)
    return freqs, coherence


# ============================================================
# 解析ワークフロー
# ============================================================

def analyze_spike_phase_locking(
    spike_times: np.ndarray,
    lfp_phase: np.ndarray,
    lfp_times: np.ndarray,
    min_spikes: int = 50
) -> Optional[PhaseLockingResult]:
    """
    単一ユニットの位相ロック解析
    
    Parameters
    ----------
    spike_times : np.ndarray
    lfp_phase : np.ndarray
    lfp_times : np.ndarray
    min_spikes : int
        解析に必要な最小スパイク数
    
    Returns
    -------
    result : PhaseLockingResult or None
        スパイク数が足りない場合はNone
    """
    # スパイク時点の位相を取得
    spike_phases, valid_mask = get_spike_phases(spike_times, lfp_phase, lfp_times)
    n_spikes = len(spike_phases)
    
    if n_spikes < min_spikes:
        return None
    
    # 各指標を計算
    mrl, preferred_phase = compute_mean_resultant_length(spike_phases)
    ppc = compute_pairwise_phase_consistency(spike_phases)
    _, z_stat, p_value = rayleigh_test(spike_phases)
    
    return PhaseLockingResult(
        mrl=mrl,
        ppc=ppc,
        preferred_phase=preferred_phase,
        z_stat=z_stat,
        p_value=p_value,
        n_spikes=n_spikes,
        spike_phases=spike_phases,
        significant=(p_value < 0.05)
    )


def analyze_spike_lfp_coupling(
    spike_times: np.ndarray,
    lfp_data: np.ndarray,
    lfp_times: np.ndarray,
    fs: int,
    freq_bands: Optional[Dict[str, Tuple[float, float]]] = None,
    min_spikes: int = 50,
    verbose: bool = True
) -> Dict[str, Dict[int, Optional[PhaseLockingResult]]]:
    """
    スパイク-LFP位相ロック解析のメイン関数
    
    Parameters
    ----------
    spike_times : np.ndarray
        スパイクタイムスタンプ
    lfp_data : np.ndarray (n_samples, n_channels)
        LFPデータ
    lfp_times : np.ndarray
        LFPタイムスタンプ
    fs : int
        サンプリング周波数
    freq_bands : dict or None
        解析する周波数帯域
        デフォルト: {'delta': (1,4), 'theta': (4,12), 'gamma': (30,80)}
    min_spikes : int
        解析に必要な最小スパイク数
    verbose : bool
        詳細出力
    
    Returns
    -------
    results : dict
        {
            'band_name': {
                channel_idx: PhaseLockingResult or None
            }
        }
    """
    if freq_bands is None:
        freq_bands = {
            'delta': (1, 4),
            'theta': (4, 12),
            'beta': (12, 30),
            'gamma': (30, 80)
        }
    
    # 入力が1Dの場合は2Dに変換
    if lfp_data.ndim == 1:
        lfp_data = lfp_data[:, np.newaxis]
    
    n_samples, n_channels = lfp_data.shape
    results = {}
    
    for band_name, freq_band in freq_bands.items():
        if verbose:
            print(f"  {band_name} ({freq_band[0]}-{freq_band[1]} Hz)...")
        
        results[band_name] = {}
        
        # 各チャンネルの位相を抽出
        lfp_phase, _, _ = extract_instantaneous_phase(lfp_data, fs, freq_band)
        
        for ch in range(n_channels):
            ch_phase = lfp_phase[:, ch] if lfp_phase.ndim > 1 else lfp_phase
            result = analyze_spike_phase_locking(
                spike_times, ch_phase, lfp_times, min_spikes
            )
            results[band_name][ch] = result
            
            if verbose and result is not None:
                sig_marker = "*" if result.significant else ""
                print(f"    Ch{ch}: MRL={result.mrl:.3f}, "
                      f"PPC={result.ppc:.3f}, "
                      f"p={result.p_value:.4f}{sig_marker}, "
                      f"n={result.n_spikes}")
    
    return results


def analyze_phase_locking_by_condition(
    spike_times: np.ndarray,
    lfp_data: np.ndarray,
    lfp_times: np.ndarray,
    fs: int,
    condition_masks: Dict[str, np.ndarray],
    freq_band: Tuple[float, float] = (4, 12),
    lfp_channel: int = 0,
    min_spikes: int = 30,
    verbose: bool = True
) -> Dict[str, Optional[PhaseLockingResult]]:
    """
    条件別（baseline/stim/post）の位相ロック解析
    
    Parameters
    ----------
    spike_times : np.ndarray
    lfp_data : np.ndarray (n_samples, n_channels)
    lfp_times : np.ndarray
    fs : int
    condition_masks : dict
        {'baseline': bool_mask, 'stim': bool_mask, 'post': bool_mask}
    freq_band : tuple
        解析する周波数帯域
    lfp_channel : int
        位相抽出に使うLFPチャンネル
    min_spikes : int
    verbose : bool
    
    Returns
    -------
    results : dict
        各条件での位相ロック結果
    """
    # LFP位相を抽出
    if lfp_data.ndim > 1:
        lfp_1d = lfp_data[:, lfp_channel]
    else:
        lfp_1d = lfp_data
    
    lfp_phase, _, _ = extract_instantaneous_phase(lfp_1d, fs, freq_band)
    
    results = {}
    
    for condition, mask in condition_masks.items():
        # マスク範囲内のスパイクを抽出
        mask_times = lfp_times[mask]
        if len(mask_times) == 0:
            results[condition] = None
            continue
        
        t_start, t_end = mask_times[0], mask_times[-1]
        
        # 条件内のスパイクを選択
        condition_spikes = spike_times[
            (spike_times >= t_start) & (spike_times <= t_end)
        ]
        
        # さらにマスク内のスパイクのみに絞る
        spike_indices = np.searchsorted(lfp_times, condition_spikes)
        spike_indices = np.clip(spike_indices, 0, len(mask) - 1)
        in_mask = mask[spike_indices]
        condition_spikes = condition_spikes[in_mask]
        
        # 位相ロック解析
        if len(condition_spikes) >= min_spikes:
            result = analyze_spike_phase_locking(
                condition_spikes, lfp_phase, lfp_times, min_spikes
            )
        else:
            result = None
        
        results[condition] = result
        
        if verbose:
            if result is not None:
                sig_marker = "*" if result.significant else ""
                print(f"  {condition}: MRL={result.mrl:.3f}, "
                      f"p={result.p_value:.4f}{sig_marker}, "
                      f"n={result.n_spikes}")
            else:
                print(f"  {condition}: スパイク不足 (< {min_spikes})")
    
    return results


# ============================================================
# ユーティリティ関数
# ============================================================

def circular_mean(phases: np.ndarray) -> float:
    """位相の円周平均を計算"""
    return np.angle(np.mean(np.exp(1j * phases)))


def circular_std(phases: np.ndarray) -> float:
    """位相の円周標準偏差を計算"""
    mrl, _ = compute_mean_resultant_length(phases)
    if mrl > 0:
        return np.sqrt(-2 * np.log(mrl))
    return np.inf


def phase_to_degrees(phase_rad: float) -> float:
    """ラジアンを度に変換（0-360度）"""
    deg = np.degrees(phase_rad)
    return deg % 360


def wrap_to_pi(phase: np.ndarray) -> np.ndarray:
    """位相を-πからπの範囲に正規化"""
    return np.angle(np.exp(1j * phase))
