"""
spike_sorting.py - スパイクソーティング バックエンド処理

機能:
- スパイク検出（閾値ベース、MAD推定）
- 波形切り出し
- PCA特徴抽出
- 自動クラスタリング（GMM + BIC）
- 品質評価（ISI違反率、SNR）
"""

import numpy as np
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import warnings
import os

# sklearn警告を抑制
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
os.environ['OMP_NUM_THREADS'] = '4'  # OpenMP警告対策


# ============================================================
# データクラス
# ============================================================

@dataclass
class SpikeUnit:
    """単一ユニットの情報"""
    unit_id: int
    spike_indices: np.ndarray      # サンプルインデックス
    spike_times: np.ndarray        # 秒単位の時刻
    waveforms: np.ndarray          # 波形 (n_spikes, n_samples)
    pca_features: np.ndarray       # PCA特徴量 (n_spikes, n_components)
    
    # 品質指標
    mean_amplitude: float = 0.0
    snr: float = 0.0
    isi_violation_rate: float = 0.0
    n_spikes: int = 0
    
    # メタデータ
    color: str = 'blue'
    is_mua: bool = False           # マルチユニットとしてマーク
    is_noise: bool = False         # ノイズとしてマーク
    
    def __post_init__(self):
        self.n_spikes = len(self.spike_indices)


@dataclass
class ChannelSortResult:
    """1チャンネルのソーティング結果"""
    channel: int
    fs: float = 0.0
    units: List[SpikeUnit] = field(default_factory=list)
    
    # 生データ参照
    filtered_data: np.ndarray = None
    threshold: float = 0.0
    sigma: float = 0.0
    
    # PCA情報
    pca_model: Any = None
    pca_explained_variance: np.ndarray = None
    
    # 全スパイク情報（クラスタリング前）
    all_spike_indices: np.ndarray = None
    all_waveforms: np.ndarray = None
    all_pca_features: np.ndarray = None
    labels: np.ndarray = None      # クラスターラベル
    
    # 波形時間軸
    waveform_time_ms: np.ndarray = None


@dataclass 
class SortingConfig:
    """ソーティング設定"""
    # フィルタ
    filter_low: float = 300.0      # Hz
    filter_high: float = 3000.0    # Hz
    filter_order: int = 4
    
    # スパイク検出
    threshold_std: float = 4.0     # 閾値（σの倍数）
    min_spike_interval_ms: float = 1.0  # 最小スパイク間隔
    artifact_threshold_std: float = 10.0  # アーティファクト閾値
    
    # 波形切り出し
    pre_spike_ms: float = 0.5      # スパイク前の時間
    post_spike_ms: float = 1.0     # スパイク後の時間
    
    # PCA
    n_pca_components: int = 3
    
    # クラスタリング
    max_clusters: int = 5
    min_cluster_size: int = 20
    
    # 品質基準
    isi_violation_threshold_ms: float = 2.0  # 不応期


# ============================================================
# フィルタリング
# ============================================================

def bandpass_filter(data: np.ndarray, fs: float, 
                    lowcut: float, highcut: float, order: int = 4) -> np.ndarray:
    """
    バンドパスフィルタ（ゼロ位相）
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = signal.butter(order, [low, high], btype='band', output='sos')
    return signal.sosfiltfilt(sos, data, axis=0)


# ============================================================
# スパイク検出
# ============================================================

def estimate_noise_std(data: np.ndarray) -> float:
    """MAD（中央絶対偏差）ベースのノイズ推定"""
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    return mad * 1.4826


def detect_spikes(data: np.ndarray, fs: float, config: SortingConfig
                  ) -> Tuple[np.ndarray, float, float]:
    """スパイク検出（負のピーク）"""
    sigma = estimate_noise_std(data)
    threshold = -config.threshold_std * sigma
    min_distance = int(config.min_spike_interval_ms / 1000 * fs)
    
    peaks, _ = signal.find_peaks(-data, height=-threshold, distance=min_distance)
    
    return peaks, threshold, sigma


def remove_artifacts(waveforms: np.ndarray, spike_indices: np.ndarray,
                     sigma: float, config: SortingConfig
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """アーティファクト除去"""
    peak_amplitudes = np.min(waveforms, axis=1)
    artifact_threshold = -config.artifact_threshold_std * sigma
    valid_mask = peak_amplitudes > artifact_threshold
    
    return waveforms[valid_mask], spike_indices[valid_mask], ~valid_mask


# ============================================================
# 波形処理
# ============================================================

def extract_waveforms(data: np.ndarray, spike_indices: np.ndarray, 
                      fs: float, config: SortingConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """波形切り出し"""
    pre_samples = int(config.pre_spike_ms / 1000 * fs)
    post_samples = int(config.post_spike_ms / 1000 * fs)
    waveform_length = pre_samples + post_samples
    
    waveforms = []
    valid_indices = []
    
    for idx in spike_indices:
        start = idx - pre_samples
        end = idx + post_samples
        
        if start >= 0 and end < len(data):
            waveforms.append(data[start:end])
            valid_indices.append(idx)
    
    time_ms = np.linspace(-config.pre_spike_ms, config.post_spike_ms, waveform_length)
    
    return np.array(waveforms), np.array(valid_indices), time_ms


# ============================================================
# PCA & クラスタリング
# ============================================================

def compute_pca(waveforms: np.ndarray, n_components: int = 3) -> Tuple[np.ndarray, PCA]:
    """PCA特徴抽出"""
    pca = PCA(n_components=n_components)
    features = pca.fit_transform(waveforms)
    return features, pca


def find_optimal_clusters(features: np.ndarray, max_clusters: int = 5,
                          min_cluster_size: int = 20) -> int:
    """BICで最適クラスター数を決定"""
    n_samples = len(features)
    max_possible = min(max_clusters, max(1, n_samples // min_cluster_size))
    
    if max_possible < 1:
        return 1
    
    bic_scores = []
    for n in range(1, max_possible + 1):
        try:
            gmm = GaussianMixture(n_components=n, random_state=42, n_init=3, max_iter=100)
            gmm.fit(features)
            bic_scores.append(gmm.bic(features))
        except:
            bic_scores.append(np.inf)
    
    if not bic_scores:
        return 1
    
    return np.argmin(bic_scores) + 1


def cluster_spikes(features: np.ndarray, n_clusters: int) -> np.ndarray:
    """GMMクラスタリング"""
    gmm = GaussianMixture(n_components=n_clusters, random_state=42, n_init=5)
    return gmm.fit_predict(features)


# ============================================================
# 品質評価
# ============================================================

def compute_isi_violation_rate(spike_indices: np.ndarray, fs: float,
                                threshold_ms: float = 2.0) -> float:
    """ISI違反率を計算"""
    if len(spike_indices) < 2:
        return 0.0
    
    isi_ms = np.diff(np.sort(spike_indices)) / fs * 1000
    n_violations = np.sum(isi_ms < threshold_ms)
    return n_violations / len(isi_ms) * 100


def compute_snr(waveforms: np.ndarray) -> float:
    """SNRを計算"""
    if len(waveforms) == 0:
        return 0.0
    
    mean_wf = np.mean(waveforms, axis=0)
    noise = waveforms - mean_wf
    noise_std = np.std(noise)
    
    if noise_std == 0:
        return 0.0
    
    return np.abs(np.min(mean_wf)) / noise_std


def compute_unit_quality(unit: SpikeUnit, fs: float, 
                         config: SortingConfig) -> SpikeUnit:
    """ユニットの品質指標を計算"""
    unit.mean_amplitude = float(np.mean(np.min(unit.waveforms, axis=1)))
    unit.snr = compute_snr(unit.waveforms)
    unit.isi_violation_rate = compute_isi_violation_rate(
        unit.spike_indices, fs, config.isi_violation_threshold_ms
    )
    return unit


def compute_isi_histogram(spike_times: np.ndarray, max_isi_ms: float = 100.0,
                         bin_size_ms: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """ISIヒストグラム計算"""
    if len(spike_times) < 2:
        return np.array([]), np.array([])
    
    isi_ms = np.diff(np.sort(spike_times)) * 1000
    bins = np.arange(0, max_isi_ms + bin_size_ms, bin_size_ms)
    hist, _ = np.histogram(isi_ms[isi_ms < max_isi_ms], bins=bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    return bin_centers, hist


def compute_autocorrelogram(spike_times: np.ndarray, bin_size_ms: float = 1.0,
                           window_ms: float = 50.0) -> Tuple[np.ndarray, np.ndarray]:
    """自己相関図を計算"""
    n_bins = int(window_ms / bin_size_ms)
    bins = np.arange(-n_bins, n_bins + 1) * bin_size_ms
    
    autocorr = np.zeros(len(bins) - 1)
    spike_times_ms = spike_times * 1000
    
    for t in spike_times_ms:
        diffs = spike_times_ms - t
        hist, _ = np.histogram(diffs, bins=bins)
        autocorr += hist
    
    # 中央のビン（0 lag）を除外
    center = len(autocorr) // 2
    autocorr[center] = 0
    
    bin_centers = (bins[:-1] + bins[1:]) / 2
    return bin_centers, autocorr


# ============================================================
# メイン処理
# ============================================================

def sort_channel(data: np.ndarray, fs: float, channel: int,
                 config: SortingConfig = None,
                 verbose: bool = True) -> ChannelSortResult:
    """1チャンネルのスパイクソーティング"""
    if config is None:
        config = SortingConfig()
    
    result = ChannelSortResult(channel=channel, fs=fs)
    result.filtered_data = data
    
    # 1. スパイク検出
    spike_indices, threshold, sigma = detect_spikes(data, fs, config)
    result.threshold = threshold
    result.sigma = sigma
    
    if verbose:
        print(f"  Ch{channel}: 検出 {len(spike_indices)} スパイク")
    
    if len(spike_indices) < config.min_cluster_size:
        if verbose:
            print(f"  Ch{channel}: スパイク不足、スキップ")
        return result
    
    # 2. 波形切り出し
    waveforms, valid_indices, time_ms = extract_waveforms(data, spike_indices, fs, config)
    result.waveform_time_ms = time_ms
    
    # 3. アーティファクト除去
    waveforms, valid_indices, _ = remove_artifacts(waveforms, valid_indices, sigma, config)
    
    if len(waveforms) < config.min_cluster_size:
        if verbose:
            print(f"  Ch{channel}: 有効スパイク不足、スキップ")
        return result
    
    result.all_spike_indices = valid_indices
    result.all_waveforms = waveforms
    
    # 4. PCA
    pca_features, pca_model = compute_pca(waveforms, config.n_pca_components)
    result.pca_model = pca_model
    result.pca_explained_variance = pca_model.explained_variance_ratio_
    result.all_pca_features = pca_features
    
    if verbose:
        var_str = ', '.join([f'{v:.2f}' for v in pca_model.explained_variance_ratio_])
        print(f"  Ch{channel}: PCA寄与率 [{var_str}]")
    
    # 5. クラスタリング
    n_clusters = find_optimal_clusters(pca_features, config.max_clusters, config.min_cluster_size)
    labels = cluster_spikes(pca_features, n_clusters)
    result.labels = labels
    
    if verbose:
        print(f"  Ch{channel}: {n_clusters} クラスター")
    
    # 6. ユニット作成
    colors = ['#1f77b4', '#d62728', '#ff7f0e', '#2ca02c', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for i in range(n_clusters):
        mask = labels == i
        if np.sum(mask) < config.min_cluster_size:
            continue
        
        unit = SpikeUnit(
            unit_id=i + 1,
            spike_indices=valid_indices[mask],
            spike_times=valid_indices[mask] / fs,
            waveforms=waveforms[mask],
            pca_features=pca_features[mask],
            color=colors[i % len(colors)]
        )
        
        unit = compute_unit_quality(unit, fs, config)
        
        if verbose:
            status = "✓" if unit.isi_violation_rate < 2 else "⚠"
            print(f"    Unit{unit.unit_id}: n={unit.n_spikes:4d}, "
                  f"amp={unit.mean_amplitude:.4f}, "
                  f"SNR={unit.snr:.1f}, "
                  f"ISI={unit.isi_violation_rate:.1f}% {status}")
        
        result.units.append(unit)
    
    return result


def sort_all_channels(wideband_data: np.ndarray, fs: float,
                      config: SortingConfig = None,
                      channels: List[int] = None,
                      verbose: bool = True) -> Dict[int, ChannelSortResult]:
    """全チャンネルのソーティング"""
    if config is None:
        config = SortingConfig()
    
    if channels is None:
        channels = list(range(wideband_data.shape[1]))
    
    if verbose:
        print(f"=== スパイクソーティング ===")
        print(f"チャンネル: {len(channels)}, fs: {fs} Hz")
        print(f"フィルタ: {config.filter_low}-{config.filter_high} Hz")
    
    # フィルタリング
    if verbose:
        print(f"\nフィルタリング...")
    filtered = bandpass_filter(wideband_data, fs, config.filter_low, config.filter_high, config.filter_order)
    
    # 各チャンネル処理
    results = {}
    for ch in channels:
        if verbose:
            print(f"\n--- Channel {ch} ---")
        result = sort_channel(filtered[:, ch], fs, ch, config, verbose)
        results[ch] = result
    
    # サマリー
    if verbose:
        total_units = sum(len(r.units) for r in results.values())
        total_spikes = sum(sum(u.n_spikes for u in r.units) for r in results.values())
        print(f"\n=== 完了 ===")
        print(f"総ユニット数: {total_units}")
        print(f"総スパイク数: {total_spikes}")
    
    return results


# ============================================================
# 編集操作（GUI用）
# ============================================================

def merge_units(result: ChannelSortResult, unit_ids: List[int], 
                config: SortingConfig = None) -> ChannelSortResult:
    """複数ユニットを統合"""
    if config is None:
        config = SortingConfig()
    
    units_to_merge = [u for u in result.units if u.unit_id in unit_ids]
    if len(units_to_merge) < 2:
        return result
    
    # データ統合
    merged_indices = np.concatenate([u.spike_indices for u in units_to_merge])
    merged_times = np.concatenate([u.spike_times for u in units_to_merge])
    merged_waveforms = np.concatenate([u.waveforms for u in units_to_merge])
    merged_pca = np.concatenate([u.pca_features for u in units_to_merge])
    
    # ソート
    sort_order = np.argsort(merged_indices)
    
    new_unit = SpikeUnit(
        unit_id=min(unit_ids),
        spike_indices=merged_indices[sort_order],
        spike_times=merged_times[sort_order],
        waveforms=merged_waveforms[sort_order],
        pca_features=merged_pca[sort_order],
        color=units_to_merge[0].color
    )
    new_unit = compute_unit_quality(new_unit, result.fs, config)
    
    # 元のユニットを削除、新しいユニットを追加
    result.units = [u for u in result.units if u.unit_id not in unit_ids]
    result.units.append(new_unit)
    result.units.sort(key=lambda u: u.unit_id)
    
    return result


def delete_unit(result: ChannelSortResult, unit_id: int) -> ChannelSortResult:
    """ユニットをノイズとしてマーク"""
    for unit in result.units:
        if unit.unit_id == unit_id:
            unit.is_noise = True
            break
    return result


def undelete_unit(result: ChannelSortResult, unit_id: int) -> ChannelSortResult:
    """ユニットの削除を取り消し"""
    for unit in result.units:
        if unit.unit_id == unit_id:
            unit.is_noise = False
            break
    return result


def mark_as_mua(result: ChannelSortResult, unit_id: int) -> ChannelSortResult:
    """ユニットをMUAとしてマーク"""
    for unit in result.units:
        if unit.unit_id == unit_id:
            unit.is_mua = True
            break
    return result


def unmark_mua(result: ChannelSortResult, unit_id: int) -> ChannelSortResult:
    """MUAマークを解除"""
    for unit in result.units:
        if unit.unit_id == unit_id:
            unit.is_mua = False
            break
    return result


def recluster(result: ChannelSortResult, n_clusters: int,
              config: SortingConfig = None) -> ChannelSortResult:
    """指定したクラスター数で再クラスタリング"""
    if config is None:
        config = SortingConfig()
    
    if result.all_pca_features is None or len(result.all_pca_features) == 0:
        return result
    
    # 再クラスタリング
    labels = cluster_spikes(result.all_pca_features, n_clusters)
    result.labels = labels
    
    # ユニット再構築
    result.units = []
    colors = ['#1f77b4', '#d62728', '#ff7f0e', '#2ca02c', '#9467bd']
    
    for i in range(n_clusters):
        mask = labels == i
        if np.sum(mask) < config.min_cluster_size:
            continue
        
        unit = SpikeUnit(
            unit_id=i + 1,
            spike_indices=result.all_spike_indices[mask],
            spike_times=result.all_spike_indices[mask] / result.fs,
            waveforms=result.all_waveforms[mask],
            pca_features=result.all_pca_features[mask],
            color=colors[i % len(colors)]
        )
        unit = compute_unit_quality(unit, result.fs, config)
        result.units.append(unit)
    
    return result


def get_valid_units(result: ChannelSortResult) -> List[SpikeUnit]:
    """有効なユニットを取得"""
    return [u for u in result.units if not u.is_noise]


def get_single_units(result: ChannelSortResult) -> List[SpikeUnit]:
    """シングルユニットを取得"""
    return [u for u in result.units if not u.is_noise and not u.is_mua]


# ============================================================
# 保存
# ============================================================

def save_sorting_results(results: Dict[int, ChannelSortResult], 
                         filepath: str):
    """ソーティング結果をNPZで保存"""
    save_dict = {}
    
    for ch, result in results.items():
        save_dict[f'ch{ch}_fs'] = result.fs
        save_dict[f'ch{ch}_threshold'] = result.threshold
        save_dict[f'ch{ch}_sigma'] = result.sigma
        
        if result.all_spike_indices is not None:
            save_dict[f'ch{ch}_all_indices'] = result.all_spike_indices
            save_dict[f'ch{ch}_all_waveforms'] = result.all_waveforms
            save_dict[f'ch{ch}_all_pca'] = result.all_pca_features
            save_dict[f'ch{ch}_labels'] = result.labels
            save_dict[f'ch{ch}_waveform_time'] = result.waveform_time_ms
        
        for unit in result.units:
            prefix = f'ch{ch}_unit{unit.unit_id}_'
            save_dict[prefix + 'indices'] = unit.spike_indices
            save_dict[prefix + 'times'] = unit.spike_times
            save_dict[prefix + 'waveforms'] = unit.waveforms
            save_dict[prefix + 'pca'] = unit.pca_features
            save_dict[prefix + 'is_noise'] = unit.is_noise
            save_dict[prefix + 'is_mua'] = unit.is_mua
            save_dict[prefix + 'snr'] = unit.snr
            save_dict[prefix + 'isi_viol'] = unit.isi_violation_rate
    
    np.savez(filepath, **save_dict)
    print(f"保存完了: {filepath}")


def export_spike_times_csv(results: Dict[int, ChannelSortResult], 
                           filepath: str):
    """スパイク時刻をCSVでエクスポート"""
    import csv
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['channel', 'unit_id', 'spike_time_sec', 'is_mua', 'is_noise'])
        
        for ch, result in results.items():
            for unit in result.units:
                for t in unit.spike_times:
                    writer.writerow([ch, unit.unit_id, t, unit.is_mua, unit.is_noise])
    
    print(f"CSV保存完了: {filepath}")
