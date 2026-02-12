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


# ============================================================
# 多重比較補正 (FDR)
# ============================================================

def fdr_correction(
    p_values: np.ndarray,
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Benjamini-Hochberg FDR補正

    多チャンネル × 多帯域の検定で発生する偽陽性を制御する。
    例: 16ch × 5帯域 = 80回の検定 → 補正なしだと期待偽陽性4個

    Parameters
    ----------
    p_values : np.ndarray
        p値の配列（任意の形状、NaN許容）
    alpha : float
        FDR制御水準（デフォルト 0.05）

    Returns
    -------
    rejected : np.ndarray (bool, 元の形状)
        補正後に有意ならTrue
    p_adjusted : np.ndarray (float, 元の形状)
        補正後のp値（元のp値以上、1.0以下）

    Notes
    -----
    Benjamini & Hochberg (1995) の手順:
      1. p値を昇順ソート: p(1) ≤ p(2) ≤ ... ≤ p(m)
      2. 各 p(k) の補正値: p_adj(k) = p(k) × m / k
      3. 単調性を保証: 右から累積min
      4. p_adj < alpha なら棄却

    使用例
    ------
    >>> # 位相ロック結果から p値行列を作成
    >>> p_mat = np.array([[0.001, 0.03, 0.8],
    ...                   [0.01,  0.04, 0.5]])  # 2帯域 × 3ch
    >>> rejected, p_adj = fdr_correction(p_mat, alpha=0.05)
    >>> rejected
    array([[ True,  True, False],
           [ True,  True, False]])
    """
    original_shape = p_values.shape
    p_flat = p_values.flatten().astype(float)
    n = len(p_flat)

    # NaN → 補正対象外（p=1.0扱い）
    valid_mask = ~np.isnan(p_flat)
    n_valid = np.sum(valid_mask)

    if n_valid == 0:
        return np.zeros(original_shape, dtype=bool), np.ones(original_shape)

    # 有効なp値だけ取り出してソート
    p_valid = p_flat[valid_mask]
    sorted_idx = np.argsort(p_valid)
    sorted_p = p_valid[sorted_idx]

    # BH補正: p_adj(k) = p(k) * m / k
    rank = np.arange(1, n_valid + 1)
    adjusted = sorted_p * n_valid / rank

    # 単調性の保証（右→左に累積min）
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    # ソート順を元に戻す
    p_adj_valid = np.empty(n_valid)
    p_adj_valid[sorted_idx] = adjusted

    # 全体配列に埋め戻す
    p_adj_flat = np.ones(n)
    p_adj_flat[valid_mask] = p_adj_valid

    rejected_flat = p_adj_flat < alpha

    return rejected_flat.reshape(original_shape), p_adj_flat.reshape(original_shape)


def apply_fdr_to_phase_results(
    phase_results: Dict[str, Dict[str, Dict[int, Optional['PhaseLockingResult']]]],
    alpha: float = 0.05
) -> Tuple[Dict[str, Dict[str, Dict[int, bool]]], int, int]:
    """
    位相ロック解析結果全体にFDR補正を一括適用

    全ユニット × 全帯域 × 全チャンネルのp値をまとめて補正し、
    補正後の有意性マップを返す。

    Parameters
    ----------
    phase_results : dict
        {unit_key: {band: {ch: PhaseLockingResult or None}}}
        analyze_spike_lfp_coupling() の出力を集めたもの
    alpha : float
        FDR制御水準

    Returns
    -------
    fdr_significant : dict
        {unit_key: {band: {ch: bool}}}
        補正後に有意ならTrue
    n_total : int
        検定の総数
    n_significant : int
        補正後に有意な数

    使用例
    ------
    >>> # 解析実行後
    >>> fdr_sig, n_total, n_sig = apply_fdr_to_phase_results(phase_results)
    >>> print(f"FDR補正: {n_sig}/{n_total} 有意")
    >>> # ヒートマップ描画時
    >>> for band in bands:
    ...     for ch in range(n_ch):
    ...         if not fdr_sig[unit_key][band][ch]:
    ...             # グレーアウト表示
    """
    # 全p値とキーを収集
    keys = []       # (unit_key, band, ch)
    p_values = []

    for unit_key, band_results in phase_results.items():
        for band, ch_results in band_results.items():
            for ch, result in ch_results.items():
                keys.append((unit_key, band, ch))
                if result is not None:
                    p_values.append(result.p_value)
                else:
                    p_values.append(np.nan)

    if len(p_values) == 0:
        return {}, 0, 0

    p_array = np.array(p_values)
    rejected, _ = fdr_correction(p_array, alpha)

    # 結果をdict構造に戻す
    fdr_significant = {}
    for i, (unit_key, band, ch) in enumerate(keys):
        if unit_key not in fdr_significant:
            fdr_significant[unit_key] = {}
        if band not in fdr_significant[unit_key]:
            fdr_significant[unit_key][band] = {}
        fdr_significant[unit_key][band][ch] = bool(rejected[i])

    n_total = int(np.sum(~np.isnan(p_array)))
    n_significant = int(np.sum(rejected))

    return fdr_significant, n_total, n_significant


# ============================================================
# Spike-Triggered Average (STA)
# ============================================================

def compute_spike_triggered_average(
    spike_times: np.ndarray,
    lfp_data: np.ndarray,
    lfp_times: np.ndarray,
    fs: int,
    window_ms: Tuple[float, float] = (-50.0, 50.0),
    max_spikes: int = 5000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Spike-Triggered Average (STA) を計算

    Parameters
    ----------
    spike_times : np.ndarray
        スパイクタイムスタンプ（秒）
    lfp_data : np.ndarray (n_samples,) or (n_samples, n_channels)
        LFPデータ
    lfp_times : np.ndarray
        LFPタイムスタンプ
    fs : int
        LFPサンプリング周波数
    window_ms : tuple (pre_ms, post_ms)
        スパイク前後の時間窓 (ms)
    max_spikes : int
        使用する最大スパイク数（ランダムサブサンプリング）

    Returns
    -------
    sta_mean : np.ndarray (n_window,) or (n_window, n_channels)
        STA平均波形
    sta_sem : np.ndarray
        STA標準誤差
    time_axis_ms : np.ndarray
        時間軸 (ms)
    """
    if lfp_data.ndim == 1:
        lfp_data = lfp_data[:, np.newaxis]

    n_samples, n_channels = lfp_data.shape
    pre_samples = int(abs(window_ms[0]) / 1000 * fs)
    post_samples = int(abs(window_ms[1]) / 1000 * fs)
    n_window = pre_samples + post_samples

    # 有効なスパイクを選択（LFP時間範囲内）
    t_start = lfp_times[pre_samples]
    t_end = lfp_times[-post_samples - 1]
    valid_mask = (spike_times >= t_start) & (spike_times <= t_end)
    valid_spikes = spike_times[valid_mask]

    if len(valid_spikes) == 0:
        time_axis = np.linspace(window_ms[0], window_ms[1], n_window)
        return np.zeros((n_window, n_channels)), np.zeros((n_window, n_channels)), time_axis

    # サブサンプリング
    if len(valid_spikes) > max_spikes:
        rng = np.random.default_rng(42)
        valid_spikes = rng.choice(valid_spikes, max_spikes, replace=False)

    # スパイクインデックスに変換
    spike_indices = np.searchsorted(lfp_times, valid_spikes)

    # LFPスニペット収集
    snippets = np.zeros((len(spike_indices), n_window, n_channels))
    for i, idx in enumerate(spike_indices):
        start = idx - pre_samples
        end = idx + post_samples
        if 0 <= start and end <= n_samples:
            snippets[i] = lfp_data[start:end, :]

    sta_mean = np.mean(snippets, axis=0)
    sta_sem = np.std(snippets, axis=0) / np.sqrt(len(spike_indices))
    time_axis = np.linspace(window_ms[0], window_ms[1], n_window)

    return sta_mean, sta_sem, time_axis


# ============================================================
# Phase-Amplitude Coupling (PAC)
# ============================================================

def compute_phase_amplitude_coupling(
    lfp_data: np.ndarray,
    fs: int,
    phase_band: Tuple[float, float] = (4, 12),
    amp_band: Tuple[float, float] = (30, 80),
    n_phase_bins: int = 18
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Phase-Amplitude Coupling (PAC) を計算 — Modulation Index (Tort et al., 2010)

    Parameters
    ----------
    lfp_data : np.ndarray (n_samples,)
        単一チャンネルLFPデータ
    fs : int
        サンプリング周波数
    phase_band : tuple
        位相抽出帯域 (例: theta 4-12Hz)
    amp_band : tuple
        振幅抽出帯域 (例: gamma 30-80Hz)
    n_phase_bins : int
        位相ビン数

    Returns
    -------
    mi : float
        Modulation Index (0 = no coupling, 高い = strong coupling)
    mean_amp_per_bin : np.ndarray (n_phase_bins,)
        位相ビンごとの平均振幅
    phase_bin_centers : np.ndarray (n_phase_bins,)
        位相ビン中心 (rad)
    """
    # 位相信号
    phase, _, _ = extract_instantaneous_phase(lfp_data, fs, phase_band)
    if phase.ndim > 1:
        phase = phase[:, 0]

    # 振幅信号
    _, amplitude, _ = extract_instantaneous_phase(lfp_data, fs, amp_band)
    if amplitude.ndim > 1:
        amplitude = amplitude[:, 0]

    # 位相ビンごとの平均振幅
    phase_bins = np.linspace(-np.pi, np.pi, n_phase_bins + 1)
    phase_bin_centers = (phase_bins[:-1] + phase_bins[1:]) / 2
    mean_amp_per_bin = np.zeros(n_phase_bins)

    for i in range(n_phase_bins):
        mask = (phase >= phase_bins[i]) & (phase < phase_bins[i + 1])
        if np.any(mask):
            mean_amp_per_bin[i] = np.mean(amplitude[mask])

    # 正規化して確率分布にする
    total = np.sum(mean_amp_per_bin)
    if total > 0:
        p = mean_amp_per_bin / total
    else:
        return 0.0, mean_amp_per_bin, phase_bin_centers

    # Modulation Index = KL divergence / log(N)
    uniform = np.ones(n_phase_bins) / n_phase_bins
    # KL divergence: D_KL(P || U) = Σ P(i) * log(P(i) / U(i))
    p_safe = np.where(p > 0, p, 1e-10)
    kl = np.sum(p_safe * np.log(p_safe / uniform))
    mi = kl / np.log(n_phase_bins)

    return float(mi), mean_amp_per_bin, phase_bin_centers


# ============================================================
# CSD (Current Source Density)
# ============================================================

def compute_csd(
    lfp_data: np.ndarray,
    channel_spacing_um: float = 50.0
) -> np.ndarray:
    """
    Current Source Density (CSD) を2次空間微分で算出

    CSD(ch) = -( LFP(ch-1) - 2*LFP(ch) + LFP(ch+1) ) / Δz²

    Parameters
    ----------
    lfp_data : np.ndarray (n_samples, n_channels)
        LFPデータ（チャンネルは深さ順）
    channel_spacing_um : float
        チャンネル間隔 (μm)

    Returns
    -------
    csd : np.ndarray (n_samples, n_channels - 2)
        CSD値。端2チャンネル分は算出不可のため縮む。
        csd[:, 0] は元のch1に対応。
        正 = source（電流湧き出し）、負 = sink（電流吸い込み）
    """
    if lfp_data.ndim == 1:
        raise ValueError("CSD計算には2ch以上が必要")
    if lfp_data.shape[1] < 3:
        raise ValueError(f"CSD計算には3ch以上必要（現在 {lfp_data.shape[1]}ch）")

    dz = channel_spacing_um * 1e-6  # μm → m
    csd = -(lfp_data[:, :-2] - 2 * lfp_data[:, 1:-1] + lfp_data[:, 2:]) / (dz ** 2)
    return csd


def compute_evoked_csd(
    lfp_data: np.ndarray,
    lfp_times: np.ndarray,
    event_times: np.ndarray,
    fs: int,
    channel_spacing_um: float = 50.0,
    window_ms: Tuple[float, float] = (-50.0, 200.0)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    イベント揃えCSD（刺激応答のsink/source分布）

    Parameters
    ----------
    lfp_data : np.ndarray (n_samples, n_channels)
    lfp_times : np.ndarray
    event_times : np.ndarray
        トリガーイベント（刺激時刻など）
    fs : int
    channel_spacing_um : float
    window_ms : tuple (pre_ms, post_ms)

    Returns
    -------
    csd_mean : np.ndarray (n_window, n_channels - 2)
        トライアル平均CSD
    time_axis : np.ndarray (n_window,)
        時間軸 (ms)
    """
    if lfp_data.ndim == 1:
        lfp_data = lfp_data[:, np.newaxis]

    n_samples, n_channels = lfp_data.shape
    pre_samples = int(abs(window_ms[0]) / 1000 * fs)
    post_samples = int(abs(window_ms[1]) / 1000 * fs)
    n_window = pre_samples + post_samples

    # イベントごとにLFPスニペットを収集
    snippets = []
    for t_event in event_times:
        idx = np.searchsorted(lfp_times, t_event)
        start = idx - pre_samples
        end = idx + post_samples
        if 0 <= start and end <= n_samples:
            snippets.append(lfp_data[start:end, :])

    if len(snippets) == 0:
        time_axis = np.linspace(window_ms[0], window_ms[1], n_window)
        return np.zeros((n_window, max(n_channels - 2, 1))), time_axis

    # トライアル平均LFP → CSD
    mean_lfp = np.mean(np.array(snippets), axis=0)
    csd_mean = compute_csd(mean_lfp, channel_spacing_um)
    time_axis = np.linspace(window_ms[0], window_ms[1], n_window)

    return csd_mean, time_axis


# ============================================================
# LFP パワースペクトル
# ============================================================

def compute_lfp_psd_matrix(
    lfp_data: np.ndarray,
    fs: int,
    nperseg: int = 1024,
    freq_range: Tuple[float, float] = (0.5, 200.0)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    全チャンネルのパワースペクトル密度（PSD）を一括計算

    Parameters
    ----------
    lfp_data : np.ndarray (n_samples, n_channels)
    fs : int
    nperseg : int
        Welch法のセグメント長
    freq_range : tuple
        表示周波数範囲 (Hz)

    Returns
    -------
    psd_matrix : np.ndarray (n_freqs, n_channels)
        パワースペクトル (dB: 10*log10)
    freqs : np.ndarray (n_freqs,)
        周波数軸 (Hz)
    """
    if lfp_data.ndim == 1:
        lfp_data = lfp_data[:, np.newaxis]

    n_channels = lfp_data.shape[1]
    freqs_all, psd_ch0 = signal.welch(lfp_data[:, 0], fs=fs, nperseg=nperseg)

    # 周波数範囲でトリミング
    freq_mask = (freqs_all >= freq_range[0]) & (freqs_all <= freq_range[1])
    freqs = freqs_all[freq_mask]

    psd_matrix = np.zeros((len(freqs), n_channels))
    for ch in range(n_channels):
        _, psd = signal.welch(lfp_data[:, ch], fs=fs, nperseg=nperseg)
        psd_db = 10 * np.log10(psd[freq_mask] + 1e-20)
        psd_matrix[:, ch] = psd_db

    return psd_matrix, freqs


# ============================================================
# スパイクのチャンネル集約
# ============================================================

def aggregate_spikes_by_channel(
    spike_data: Dict,
    n_channels: int
) -> Dict[int, np.ndarray]:
    """
    スパイクソーティング結果をチャンネル単位に集約

    同一チャンネルの全ユニットのスパイクをプールし、
    深さ方向の解析（depth×depth マトリクス等）に使う。

    Parameters
    ----------
    spike_data : dict
        {'unit_info': [UnitInfo, ...], 'spike_times': {unit_key: ndarray}}
        phase_gui._convert_sorting_to_spike_data() の出力形式
    n_channels : int
        LFPチャンネル数

    Returns
    -------
    channel_spikes : dict
        {channel_idx: np.ndarray (sorted spike times)}
        スパイクが存在するチャンネルのみ含む
    """
    channel_spikes = {}

    for ui in spike_data['unit_info']:
        ch = ui.channel
        times = spike_data['spike_times'].get(ui.unit_key)
        if times is None or len(times) == 0:
            continue
        if ch not in channel_spikes:
            channel_spikes[ch] = []
        channel_spikes[ch].append(times)

    # 各チャンネルでマージ＆ソート
    for ch in channel_spikes:
        merged = np.concatenate(channel_spikes[ch])
        channel_spikes[ch] = np.sort(merged)

    return channel_spikes
