"""
kilosort_wrapper.py - KiloSort4 ラッパー

KiloSort4 (Python版) を使ったスパイクソーティングを実行し、
既存の ChannelSortResult / SpikeUnit 形式で結果を返す。

使用条件:
  pip install kilosort
  NVIDIA GPU + CUDA が推奨（CPUでも動作するが非常に遅い）
"""

import numpy as np
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import field

from spike_sorting import (
    SortingConfig, ChannelSortResult, SpikeUnit,
    bandpass_filter, compute_pca, compute_unit_quality,
    estimate_noise_std
)


def is_kilosort_available() -> bool:
    """KiloSort4がインストールされているかチェック"""
    try:
        import kilosort
        return True
    except Exception:
        return False


def create_linear_probe(n_channels: int,
                        channel_spacing_um: float = 25.0) -> dict:
    """
    線形プローブ定義を生成

    Parameters
    ----------
    n_channels : int
        チャンネル数
    channel_spacing_um : float
        チャンネル間隔 (マイクロメートル)

    Returns
    -------
    probe : dict
        KiloSort4用プローブ定義
    """
    return {
        'chanMap': np.arange(n_channels),
        'xc': np.zeros(n_channels),
        'yc': np.arange(n_channels) * channel_spacing_um,
        'kcoords': np.zeros(n_channels),
        'n_chan': n_channels,
    }


def _scale_to_int16(data: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    float データを int16 にスケーリング

    Returns
    -------
    data_int16 : np.ndarray (int16)
    scale_factor : float
        元のスケールに戻すための係数
    """
    if data.dtype == np.int16:
        return data, 1.0

    max_val = np.max(np.abs(data))
    if max_val == 0:
        return np.zeros_like(data, dtype=np.int16), 1.0

    # int16 の範囲 (-32768 ~ 32767) の 90% にスケーリング
    scale = 32000.0 / max_val
    data_int16 = (data * scale).astype(np.int16)
    return data_int16, 1.0 / scale


def run_kilosort_sorting(
    wideband_data: np.ndarray,
    fs: float,
    config: SortingConfig = None,
    channels: List[int] = None,
    output_dir: str = "",
    channel_spacing_um: float = 25.0,
    n_templates: int = 6,
    verbose: bool = True
) -> Dict[int, ChannelSortResult]:
    """
    KiloSort4でスパイクソーティングを実行し、既存形式で返す

    Parameters
    ----------
    wideband_data : np.ndarray
        ワイドバンドデータ (n_samples, n_channels)
    fs : float
        サンプリング周波数 (Hz)
    config : SortingConfig
        ソーティング設定（フィルタ、波形切り出しなどに使用）
    channels : list of int or None
        処理対象チャンネル (None=全チャンネル)
    output_dir : str
        KiloSort4の結果保存先
    channel_spacing_um : float
        チャンネル間隔 (マイクロメートル)
    verbose : bool
        詳細出力

    Returns
    -------
    results : Dict[int, ChannelSortResult]
        チャンネルごとのソーティング結果
    """
    from kilosort import run_kilosort

    if config is None:
        config = SortingConfig()

    if channels is None:
        channels = list(range(wideband_data.shape[1]))

    # 対象チャンネルのみ抽出
    data = wideband_data[:, channels]
    n_samples, n_channels = data.shape

    if verbose:
        print(f"=== KiloSort4 スパイクソーティング ===")
        print(f"チャンネル: {n_channels}, サンプル: {n_samples}, fs: {fs} Hz")

    # int16 にスケーリング
    data_int16, inv_scale = _scale_to_int16(data)

    # プローブ定義
    probe = create_linear_probe(n_channels, channel_spacing_um)

    # KiloSort4 設定
    ks_settings = {
        'n_chan_bin': n_channels,
        'fs': fs,
        'batch_size': int(min(fs * 2, n_samples)),  # 2秒 or データ長
        'n_templates': n_templates,
    }

    # 結果保存ディレクトリ（最終コピー先）
    if output_dir:
        final_results_dir = Path(output_dir) / 'kilosort4'
    else:
        final_results_dir = None

    # KiloSort4はローカルパスで動作させる（UNCパス非対応のため）
    local_tmpdir = tempfile.mkdtemp(prefix='ks4_')
    results_dir = Path(local_tmpdir) / 'kilosort4'
    results_dir.mkdir(parents=True, exist_ok=True)

    # バイナリファイルを実際に書き出す（KiloSort4 v4.1+ はファイル存在チェックする）
    bin_filename = results_dir / 'data.bin'
    data_int16.tofile(str(bin_filename))

    if verbose:
        print(f"KiloSort4 実行中...")
        print(f"  作業ディレクトリ: {results_dir}")
        print(f"  バイナリファイル: {bin_filename} ({bin_filename.stat().st_size / 1e6:.1f} MB)")

    # KiloSort4 実行
    try:
        import torch
        device = None  # 自動選択 (GPU利用可能ならGPU)
        if verbose:
            if torch.cuda.is_available():
                print(f"  デバイス: GPU ({torch.cuda.get_device_name(0)})")
            else:
                print(f"  デバイス: CPU (GPU未検出 - 処理が遅くなります)")

        ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate, kept_spikes = \
            run_kilosort(
                settings=ks_settings,
                probe=probe,
                filename=bin_filename,
                file_object=data_int16,
                results_dir=results_dir,
                data_dtype='int16',
                do_CAR=True,
                device=device,
                verbose_console=verbose,
            )
    except Exception as e:
        print(f"KiloSort4 エラー: {e}")
        raise
    finally:
        # 一時バイナリファイルを削除
        if bin_filename.exists():
            bin_filename.unlink()

    # 結果をネットワークドライブにコピー
    if final_results_dir is not None:
        import shutil
        final_results_dir.mkdir(parents=True, exist_ok=True)
        for f in results_dir.iterdir():
            if f.is_file() and f.name != 'data.bin':
                shutil.copy2(str(f), str(final_results_dir / f.name))
        if verbose:
            print(f"  結果コピー先: {final_results_dir}")

    if verbose:
        n_clusters = len(np.unique(clu))
        n_spikes = len(st)
        print(f"KiloSort4 完了: {n_clusters} クラスター, {n_spikes} スパイク")

    # 結果を既存形式に変換
    results = _convert_kilosort_results(
        st, clu, is_ref, est_contam_rate, Wall,
        data, fs, channels, config, verbose
    )

    return results


def _convert_kilosort_results(
    st: np.ndarray,
    clu: np.ndarray,
    is_ref: np.ndarray,
    est_contam_rate: np.ndarray,
    Wall: np.ndarray,
    filtered_data: np.ndarray,
    fs: float,
    channels: List[int],
    config: SortingConfig,
    verbose: bool = True
) -> Dict[int, ChannelSortResult]:
    """
    KiloSort4の出力を ChannelSortResult 形式に変換

    Parameters
    ----------
    st : np.ndarray
        スパイク情報 (n_spikes, 3): [peak_sample, template_id, amplitude]
    clu : np.ndarray
        クラスターラベル (n_spikes,)
    is_ref : np.ndarray
        True=good, False=MUA (n_clusters,)
    est_contam_rate : np.ndarray
        汚染率 (n_clusters,)
    Wall : np.ndarray
        テンプレート (n_clusters, n_channels, n_pcs)
    filtered_data : np.ndarray
        フィルタ済みデータ (n_samples, n_channels)
    fs : float
        サンプリング周波数
    channels : list of int
        チャンネルインデックスのリスト
    config : SortingConfig
    verbose : bool
    """
    n_channels = filtered_data.shape[1]
    unique_clusters = np.unique(clu)

    # バンドパスフィルタ適用（波形切り出し用）
    if verbose:
        print(f"バンドパスフィルタ適用中 ({config.filter_low}-{config.filter_high} Hz)...")
    bp_data = bandpass_filter(
        filtered_data, fs,
        config.filter_low, config.filter_high, config.filter_order
    )

    # 各チャンネルの ChannelSortResult を初期化
    results: Dict[int, ChannelSortResult] = {}
    for i, ch in enumerate(channels):
        sigma = estimate_noise_std(bp_data[:, i])
        results[ch] = ChannelSortResult(
            channel=ch,
            fs=fs,
            filtered_data=bp_data[:, i],
            threshold=-config.threshold_std * sigma,
            sigma=sigma,
        )

    # 波形切り出しパラメータ
    pre_samples = int(config.pre_spike_ms / 1000 * fs)
    post_samples = int(config.post_spike_ms / 1000 * fs)
    waveform_length = pre_samples + post_samples
    waveform_time_ms = np.linspace(-config.pre_spike_ms, config.post_spike_ms, waveform_length)

    colors = ['#1f77b4', '#d62728', '#ff7f0e', '#2ca02c', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # チャンネルごとのユニットIDカウンター
    ch_unit_counters = {ch: 0 for ch in channels}

    if verbose:
        print(f"クラスター → チャンネル割り当て中...")

    for cluster_id in unique_clusters:
        cluster_mask = (clu == cluster_id)
        n_spikes_cluster = np.sum(cluster_mask)

        if n_spikes_cluster < config.min_cluster_size:
            if verbose:
                print(f"  Cluster {cluster_id}: スパイク不足 ({n_spikes_cluster}), スキップ")
            continue

        # スパイクのサンプルインデックス
        spike_samples = st[cluster_mask, 0].astype(int)

        # best channel の決定: テンプレートの振幅が最大のチャンネル
        if Wall is not None and cluster_id < len(Wall):
            template = Wall[cluster_id]  # (n_channels, n_pcs)
            # PCA特徴量の L2 ノルムでチャンネルごとの振幅を推定
            ch_amplitudes = np.sqrt(np.sum(template ** 2, axis=-1))
            best_ch_idx = int(np.argmax(ch_amplitudes))
        else:
            # テンプレートが無い場合: 最初のスパイクの振幅で判定
            best_ch_idx = 0
            if len(spike_samples) > 0:
                sample = spike_samples[0]
                if 0 <= sample < len(bp_data):
                    best_ch_idx = int(np.argmin(bp_data[sample, :]))

        best_channel = channels[best_ch_idx]

        # 波形切り出し
        waveforms = []
        valid_indices = []
        for idx in spike_samples:
            start = idx - pre_samples
            end = idx + post_samples
            if start >= 0 and end < len(bp_data):
                waveforms.append(bp_data[start:end, best_ch_idx])
                valid_indices.append(idx)

        if len(waveforms) < config.min_cluster_size:
            if verbose:
                print(f"  Cluster {cluster_id}: 有効波形不足, スキップ")
            continue

        waveforms = np.array(waveforms)
        valid_indices = np.array(valid_indices)

        # PCA
        n_components = min(config.n_pca_components, waveforms.shape[0] - 1, waveforms.shape[1])
        if n_components < 1:
            n_components = 1
        pca_features, _ = compute_pca(waveforms, n_components)

        # ユニットID割り当て
        ch_unit_counters[best_channel] += 1
        unit_id = ch_unit_counters[best_channel]

        # SpikeUnit 作成
        unit = SpikeUnit(
            unit_id=unit_id,
            spike_indices=valid_indices,
            spike_times=valid_indices / fs,
            waveforms=waveforms,
            pca_features=pca_features,
            color=colors[(unit_id - 1) % len(colors)]
        )

        # 品質指標
        unit = compute_unit_quality(unit, fs, config)

        # MUA 判定: KiloSort4 の is_ref を使用
        if cluster_id < len(is_ref):
            if not is_ref[cluster_id]:
                unit.is_mua = True
        # さらに config の MUA 基準も適用
        if (unit.isi_violation_rate > config.mua_isi_threshold or
                (unit.isi_violation_rate > 2.0 and unit.snr < config.mua_snr_threshold)):
            unit.is_mua = True

        if verbose:
            ks_label = "MUA(KS)" if (cluster_id < len(is_ref) and not is_ref[cluster_id]) else "good(KS)"
            status = "[MUA]" if unit.is_mua else ("✓" if unit.isi_violation_rate < 2 else "⚠")
            contam = est_contam_rate[cluster_id] if cluster_id < len(est_contam_rate) else 0
            print(f"  Cluster{cluster_id} → Ch{best_channel} Unit{unit_id}: "
                  f"n={unit.n_spikes:4d}, SNR={unit.snr:.1f}, "
                  f"ISI={unit.isi_violation_rate:.1f}%, "
                  f"contam={contam:.1f}%, {ks_label} {status}")

        # ChannelSortResult に追加
        results[best_channel].units.append(unit)
        results[best_channel].waveform_time_ms = waveform_time_ms

    # サマリー
    if verbose:
        total_units = sum(len(r.units) for r in results.values())
        total_spikes = sum(sum(u.n_spikes for u in r.units) for r in results.values())
        n_mua = sum(sum(1 for u in r.units if u.is_mua) for r in results.values())
        print(f"\n=== KiloSort4 完了 ===")
        print(f"総ユニット数: {total_units} (MUA: {n_mua})")
        print(f"総スパイク数: {total_spikes}")

    return results
