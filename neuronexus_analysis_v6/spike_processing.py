"""
spike_processing.py - スパイクデータの読み込みと前処理

PLXファイルからスパイクデータを読み込み、位相ロック解析用に整形する
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class UnitInfo:
    """ユニット情報を格納するデータクラス"""
    channel: int
    unit_id: int
    n_spikes: int
    t_start: float
    t_stop: float
    unit_key: str
    
    @classmethod
    def from_spike_train(cls, spike_train) -> 'UnitInfo':
        """NeoのSpikeTrainからUnitInfoを生成"""
        channel = spike_train.annotations.get('channel_id', 
                  spike_train.annotations.get('channel_index', -1))
        unit_id = spike_train.annotations.get('unit_id', 
                  spike_train.annotations.get('unit_index', 0))
        
        return cls(
            channel=int(channel),
            unit_id=int(unit_id),
            n_spikes=len(spike_train.times),
            t_start=float(spike_train.t_start),
            t_stop=float(spike_train.t_stop),
            unit_key=get_unit_key(channel, unit_id)
        )


def get_unit_key(channel: int, unit_id: int) -> str:
    """
    ユニット識別子を生成
    
    Parameters
    ----------
    channel : int
        チャンネル番号
    unit_id : int
        ユニットID
    
    Returns
    -------
    str
        'ch{channel}_unit{unit_id}' 形式の識別子
    """
    return f"ch{channel}_unit{unit_id}"


def load_spike_data(
    segment,
    unit_filter: Optional[Dict] = None,
    quality_filter: str = 'all',
    verbose: bool = True
) -> Dict[str, Any]:
    """
    PLXファイル（Neoセグメント）からスパイクデータを読み込む
    
    Parameters
    ----------
    segment : neo.Segment
        Neoライブラリで読み込んだセグメント
    unit_filter : dict or None
        ユニットのフィルタリング条件
        例: {'channel': [1,2,3], 'unit_id': [1,2]}
    quality_filter : str
        'all': 全ユニット
        'sorted_only': ソート済みユニットのみ (unit_id > 0)
        'mua': マルチユニットアクティビティのみ (unit_id == 0)
    verbose : bool
        詳細出力
    
    Returns
    -------
    spike_data : dict
        {
            'spike_trains': list of SpikeTrain objects,
            'unit_info': list of UnitInfo,
            'spike_times': dict {unit_key: np.array of times in seconds}
        }
    """
    spike_trains = segment.spiketrains
    
    if verbose:
        print(f"  スパイクデータ: {len(spike_trains)} ユニット検出")
    
    # フィルタリング
    filtered_trains = []
    unit_info_list = []
    spike_times_dict = {}
    
    for st in spike_trains:
        info = UnitInfo.from_spike_train(st)
        
        # quality_filterによるフィルタリング
        if quality_filter == 'sorted_only' and info.unit_id == 0:
            continue
        elif quality_filter == 'mua' and info.unit_id != 0:
            continue
        
        # unit_filterによるフィルタリング
        if unit_filter is not None:
            if 'channel' in unit_filter:
                if info.channel not in unit_filter['channel']:
                    continue
            if 'unit_id' in unit_filter:
                if info.unit_id not in unit_filter['unit_id']:
                    continue
        
        filtered_trains.append(st)
        unit_info_list.append(info)
        spike_times_dict[info.unit_key] = np.array(st.times.magnitude)
        
        if verbose:
            print(f"    {info.unit_key}: {info.n_spikes} spikes "
                  f"({info.t_start:.1f}s - {info.t_stop:.1f}s)")
    
    if verbose:
        print(f"  フィルタ後: {len(filtered_trains)} ユニット")
        total_spikes = sum(info.n_spikes for info in unit_info_list)
        print(f"  総スパイク数: {total_spikes}")
    
    return {
        'spike_trains': filtered_trains,
        'unit_info': unit_info_list,
        'spike_times': spike_times_dict
    }


def filter_spikes_by_time(
    spike_times: np.ndarray,
    t_start: float,
    t_end: float
) -> np.ndarray:
    """
    時間範囲でスパイクをフィルタリング
    
    Parameters
    ----------
    spike_times : np.ndarray
        スパイクタイムスタンプ（秒）
    t_start, t_end : float
        時間範囲
    
    Returns
    -------
    filtered_spikes : np.ndarray
        時間範囲内のスパイク
    """
    mask = (spike_times >= t_start) & (spike_times <= t_end)
    return spike_times[mask]


def filter_spikes_by_mask(
    spike_times: np.ndarray,
    mask: np.ndarray,
    lfp_times: np.ndarray
) -> np.ndarray:
    """
    LFPマスクでスパイクをフィルタリング
    
    Parameters
    ----------
    spike_times : np.ndarray
        スパイクタイムスタンプ
    mask : np.ndarray (bool)
        LFP時間軸に対応するマスク
    lfp_times : np.ndarray
        LFPタイムスタンプ
    
    Returns
    -------
    filtered_spikes : np.ndarray
        マスク内のスパイク
    """
    # スパイク時刻をLFPインデックスに変換
    spike_indices = np.searchsorted(lfp_times, spike_times)
    
    # 境界チェック
    valid = (spike_indices >= 0) & (spike_indices < len(mask))
    spike_indices = spike_indices[valid]
    spike_times = spike_times[valid]
    
    # マスク適用
    in_mask = mask[spike_indices]
    return spike_times[in_mask]


def get_spike_counts_per_condition(
    spike_times: np.ndarray,
    condition_masks: Dict[str, np.ndarray],
    lfp_times: np.ndarray
) -> Dict[str, int]:
    """
    条件別のスパイク数をカウント
    
    Parameters
    ----------
    spike_times : np.ndarray
        スパイクタイムスタンプ
    condition_masks : dict
        {'baseline': mask, 'stim': mask, 'post': mask}
    lfp_times : np.ndarray
        LFPタイムスタンプ
    
    Returns
    -------
    counts : dict
        各条件でのスパイク数
    """
    counts = {}
    for condition, mask in condition_masks.items():
        filtered = filter_spikes_by_mask(spike_times, mask, lfp_times)
        counts[condition] = len(filtered)
    return counts


def compute_firing_rate(
    spike_times: np.ndarray,
    t_start: float,
    t_end: float
) -> float:
    """
    発火率を計算
    
    Parameters
    ----------
    spike_times : np.ndarray
        スパイクタイムスタンプ（秒）
    t_start, t_end : float
        時間範囲
    
    Returns
    -------
    firing_rate : float
        発火率 (spikes/sec)
    """
    spikes_in_range = filter_spikes_by_time(spike_times, t_start, t_end)
    duration = t_end - t_start
    if duration <= 0:
        return 0.0
    return len(spikes_in_range) / duration


def compute_firing_rate_per_condition(
    spike_times: np.ndarray,
    condition_masks: Dict[str, np.ndarray],
    lfp_times: np.ndarray,
    fs: int
) -> Dict[str, float]:
    """
    条件別の発火率を計算
    
    Parameters
    ----------
    spike_times : np.ndarray
    condition_masks : dict
    lfp_times : np.ndarray
    fs : int
        サンプリング周波数
    
    Returns
    -------
    rates : dict
        各条件での発火率 (spikes/sec)
    """
    rates = {}
    for condition, mask in condition_masks.items():
        n_spikes = len(filter_spikes_by_mask(spike_times, mask, lfp_times))
        duration = np.sum(mask) / fs  # マスク内のサンプル数から秒数を計算
        rates[condition] = n_spikes / duration if duration > 0 else 0.0
    return rates


def exclude_stimulus_artifact(
    spike_times: np.ndarray,
    stim_times: np.ndarray,
    artifact_window: float = 0.005
) -> np.ndarray:
    """
    刺激直後のスパイク（アーティファクト）を除外
    
    Parameters
    ----------
    spike_times : np.ndarray
        スパイクタイムスタンプ
    stim_times : np.ndarray
        刺激タイムスタンプ
    artifact_window : float
        除外する時間窓（秒）、デフォルト5ms
    
    Returns
    -------
    filtered_spikes : np.ndarray
        アーティファクト除外後のスパイク
    """
    mask = np.ones(len(spike_times), dtype=bool)
    
    for stim_t in stim_times:
        artifact_mask = (spike_times >= stim_t) & (spike_times < stim_t + artifact_window)
        mask &= ~artifact_mask
    
    return spike_times[mask]


def summarize_spike_data(
    spike_data: Dict[str, Any],
    lfp_times: np.ndarray,
    condition_masks: Optional[Dict[str, np.ndarray]] = None,
    fs: int = 1000
) -> Dict[str, Dict]:
    """
    スパイクデータのサマリーを生成
    
    Parameters
    ----------
    spike_data : dict
        load_spike_dataの出力
    lfp_times : np.ndarray
    condition_masks : dict or None
    fs : int
    
    Returns
    -------
    summary : dict
        ユニットごとのサマリー情報
    """
    summary = {}
    
    for info in spike_data['unit_info']:
        spike_times = spike_data['spike_times'][info.unit_key]
        
        unit_summary = {
            'channel': info.channel,
            'unit_id': info.unit_id,
            'n_spikes': info.n_spikes,
            't_start': info.t_start,
            't_stop': info.t_stop,
            'mean_firing_rate': info.n_spikes / (info.t_stop - info.t_start)
        }
        
        if condition_masks is not None:
            unit_summary['counts_per_condition'] = get_spike_counts_per_condition(
                spike_times, condition_masks, lfp_times
            )
            unit_summary['rates_per_condition'] = compute_firing_rate_per_condition(
                spike_times, condition_masks, lfp_times, fs
            )
        
        summary[info.unit_key] = unit_summary
    
    return summary
