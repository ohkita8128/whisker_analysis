"""
pipeline.py - Neuronexus解析パイプライン
設定に基づいて処理を実行（フロー制御のみ）
"""
import numpy as np
import os
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any


@dataclass
class PipelineConfig:
    """パイプライン設定"""
    
    # === ファイル設定 ===
    plx_file: str = ""
    output_dir: str = ""
    
    # === チャンネル設定 ===
    channel_order: List[int] = field(default_factory=lambda: [8, 7, 9, 6, 12, 3, 11, 4, 14, 1, 15, 0, 13, 2, 10, 5])
    
    # === フィルタ設定 ===
    filter_enabled: bool = True
    filter_lowcut: float = 0.1
    filter_highcut: float = 100.0
    filter_order: int = 4
    notch_enabled: bool = True
    notch_freq: float = 60.0
    notch_Q: float = 60.0
        # 環境ノイズ除去
    noise_removal_enabled: bool = False
    noise_file: str = ""
    noise_threshold_db: float = 10.0
    noise_q: float = 30.0
    noise_max_peaks: int = 10
    # 高調波ノイズ除去（ピエゾ駆動由来）
    harmonic_removal_enabled: bool = True
    harmonic_fundamental: float = 10.0  # 基本周波数 (Hz)
    harmonic_count: int = 5             # 高調波の数 (5 = 10,30,50,70,90Hz)
    harmonic_q: float = 50.0            # ノッチフィルタのQ値
    # === チャンネル処理 ===
    bad_channel_detection: bool = True
    bad_channel_threshold: float = 3.0
    manual_bad_channels: List[int] = field(default_factory=list)
    
    # === モーション解析 ===
    motion_analysis: bool = True
    motion_roi: Optional[Tuple[int, int, int, int]] = None
    motion_threshold_val: int = 15
    motion_blur: int = 5
    motion_percentile: float = 75.0
    motion_expand_sec: float = 0.1
    
    # === ICA設定 ===
    ica_enabled: bool = True
    ica_noise_ratio_threshold: float = 1.5
    ica_max_remove: int = 4
    
    # === 解析設定 ===
    n_sessions: int = 9
    n_stim_per_session: int = 10
    baseline_pre_sec: float = 3.0
    post_duration_sec: float = 3.0
    stim_margin_sec: float = 0.0
    
    # === 周波数帯域 ===
    bands: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 14),
        'beta': (14, 30), 'gamma': (30, 80)
    })
    
    # === 表示・保存設定 ===
    plot_t_start: Optional[float] = None  # None=最初から
    plot_t_end: Optional[float] = None    # None=最後まで

    save_plots: bool = True

    processing_overview: bool = True
    edge_check: bool = False
    lfp_regions: bool = True           # 全チャンネル+領域プロット
    fft_comparison: bool = True        # FFT比較プロット
    fft_freq_max: float = 300.0             # FFT最大周波数

    ica_components: bool = True
    power_analysis: bool = True
    power_freq_max: float = 100.0
    channel_heatmap: bool = True
    save_summary_csv: bool = True
    save_channel_csv: bool = True
    save_results_npz: bool = True
    save_processed_npz: bool = True
    create_sync_video: bool = False
    sync_video_start: Optional[float] = None
    sync_video_end: Optional[float] = None
    
    # === ウェーブレット設定 ===
    wavelet_enabled: bool = False
    wavelet_freq_min: float = 1.0
    wavelet_freq_max: float = 100.0
    wavelet_n_freqs: int = 50
    wavelet_start: Optional[float] = None
    wavelet_end: Optional[float] = None
    wavelet_single: bool = True  # 単一チャンネル
    wavelet_all: bool = True     # 全チャンネル
    wavelet_channel: int = 0          # 単一チャンネル用
    
    # === 追加プロット設定 ===
    
    # === 同期動画設定 ===
    sync_video_t_start: Optional[float] = None  # 動画開始時刻（None=最初から）
    
    # === 表示設定 ===
    show_plots: bool = True
    verbose: bool = True


def run_pipeline(config: PipelineConfig) -> Dict[str, Any]:
    """パイプライン実行"""
    import neo
    import cv2
    
    # 関数インポート
    from processing import (
        bandpass_notch_filter, get_frame_times, detect_bad_channels,
        analyze_video_motion, create_noise_mask, remove_artifact_ica,
        get_stim_events, create_stim_mask, compute_psd, compute_band_power,
        compute_cwt, load_noise_reference, remove_environmental_noise, remove_known_harmonics
    )
    from plotting import (
        plot_processing_overview, plot_edge_check, plot_ica_components,
        plot_power_analysis, plot_channel_heatmap, create_sync_video,
        plot_wavelet_single, plot_wavelet_all, plot_all_channels_with_regions,
        plot_fft_comparison
    )
    from saving import (
        save_summary_csv, save_channel_csv, save_results_npz, save_processed_npz
    )
    
    results = {}
    log = print if config.verbose else lambda x: None
    
    # =========================================================================
    # 1. ファイル設定
    # =========================================================================
    if not config.plx_file:
        from get_path import get_path
        config.plx_file = get_path(mode='file', file_type='plx')
    
    plx_file = config.plx_file
    video_file = os.path.splitext(plx_file)[0] + '.mp4'
    output_dir = config.output_dir or os.path.dirname(plx_file)
    basename = os.path.splitext(os.path.basename(plx_file))[0]
    
    log(f'{plx_file} を解析します。')
    log(f"=== パイプライン開始: {basename} ===")
    
    # =========================================================================
    # 2. データ読み込み
    # =========================================================================
    log("\n[1/6] データ読み込み...")
    
    plx = neo.io.PlexonIO(filename=plx_file)
    seg = plx.read()[0].segments[0]
    
    lfp_raw = np.array(seg.analogsignals[1][:, config.channel_order])
    fs = int(seg.analogsignals[1].sampling_rate)
    lfp_times_full = np.arange(len(lfp_raw)) / fs
    
    log(f"  LFP: {lfp_raw.shape}, fs={fs}Hz")
    
    # フレーム同期
    frame_times = get_frame_times(seg.events, verbose=config.verbose)
    
    cap = cv2.VideoCapture(video_file)
    n_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    n_sync = min(len(frame_times), n_video_frames)
    frame_times = frame_times[:n_sync]
    TRIM_START, TRIM_END = float(frame_times[0]), float(frame_times[-1])
    
    log(f"  Trim範囲: {TRIM_START:.2f}s ~ {TRIM_END:.2f}s")
    

    # pipeline.py の読み込み後に確認
    log(f"PLX全長: {lfp_times_full[-1]:.1f}秒")
    log(f"動画フレーム数: {n_video_frames}")
    log(f"同期イベント数: {len(frame_times)}")
    log(f"Trim範囲: {TRIM_START:.1f}〜{TRIM_END:.1f}秒 (= {TRIM_END - TRIM_START:.1f}秒)")










    # =========================================================================
    # 3. フィルタリング & Trim
    # =========================================================================
    log("\n[2/6] フィルタリング...")
    
    if config.filter_enabled:
        notch = config.notch_freq if config.notch_enabled else None
        lfp_filtered_full = bandpass_notch_filter(
            lfp_raw, config.filter_lowcut, config.filter_highcut, fs,
            order=config.filter_order, notch_freq=notch, notch_Q=config.notch_Q
        )
        log(f"  バンドパス: {config.filter_lowcut}-{config.filter_highcut}Hz (order={config.filter_order})")
        if config.notch_enabled:
            log(f"  ノッチフィルタ: {config.notch_freq}Hz (Q={config.notch_Q})")
    else:
        lfp_filtered_full = lfp_raw.copy()
        log("  フィルタ: スキップ")
    
    # Trim
    idx_start, idx_end = int(TRIM_START * fs), int(TRIM_END * fs)
    lfp_trimmed = lfp_raw[idx_start:idx_end, :]
    lfp_filtered = lfp_filtered_full[idx_start:idx_end, :]
    lfp_times = np.arange(len(lfp_filtered)) / fs + TRIM_START
    log(f"  Trim: {TRIM_START:.1f}〜{TRIM_END:.1f}秒 ({len(lfp_filtered)/fs:.1f}秒間)")
    
    # FFT比較（フィルタ効果確認）
    if config.fft_comparison:
        plot_fft_comparison(
            lfp_trimmed, lfp_filtered, fs, output_dir, basename,
            freq_max=config.fft_freq_max, show=config.show_plots, save=config.save_plots
        )
        log("  → FFT比較プロット出力")

    # 環境ノイズ除去
    if config.noise_removal_enabled and config.noise_file:
        log("\n[2.5/6] 環境ノイズ除去...")
        log(f"  ノイズファイル: {os.path.basename(config.noise_file)}")
        
        # ノイズファイル読み込み
        noise_raw, _ = load_noise_reference(config.noise_file, config.channel_order, fs)
        log(f"  ノイズデータ: {noise_raw.shape[0]/fs:.1f}秒, {noise_raw.shape[1]}ch")
        
        # ノイズにも同じフィルタを適用
        noise_filtered = bandpass_notch_filter(
            noise_raw, config.filter_lowcut, config.filter_highcut, fs,
            notch_freq=config.notch_freq if config.notch_enabled else None
        )
        
        # ノイズ除去
        lfp_filtered, noise_peaks, noise_psd = remove_environmental_noise(
            lfp_filtered, noise_filtered, fs,
            threshold_db=config.noise_threshold_db,
            Q=config.noise_q,
            max_peaks=config.noise_max_peaks
        )
        
        if noise_peaks:
            log(f"  検出ピーク: {[f'{f:.1f}Hz' for f in noise_peaks]}")
            log(f"  ノッチ除去: Q={config.noise_q}")
        else:
            log("  ⚠ ノイズピーク検出されず（スキップ）")

    # 高調波ノイズ除去（ピエゾ由来）
    if config.harmonic_removal_enabled:
        log("\n[2.6/6] 高調波ノイズ除去（ピエゾ由来）...")
        
        # 除去前のパワーを保存（検証用）
        from scipy.signal import welch
        freqs_check, psd_before = welch(lfp_filtered.mean(axis=1), fs, nperseg=fs*2)
        
        lfp_filtered, harmonics = remove_known_harmonics(
            lfp_filtered, fs,
            fundamental=config.harmonic_fundamental,
            n_harmonics=config.harmonic_count,
            Q=config.harmonic_q
        )
        
        log(f"  基本周波数: {config.harmonic_fundamental}Hz（矩形波）")
        log(f"  除去対象: {[f'{h:.0f}Hz' for h in harmonics]}（奇数倍高調波）")
        log(f"  ノッチQ値: {config.harmonic_q}（帯域幅 ≈ {config.harmonic_fundamental/config.harmonic_q:.2f}Hz）")
        
        # === 検証: 神経信号帯域への影響を確認 ===
        freqs_check, psd_after = welch(lfp_filtered.mean(axis=1), fs, nperseg=fs*2)
        
        # 各帯域での変化を計算
        bands_check = {'theta': (4, 8), 'alpha': (8, 14), 'beta': (14, 30), 'gamma': (30, 80)}
        log("  --- 神経信号帯域への影響 ---")
        for band_name, (f_low, f_high) in bands_check.items():
            mask = (freqs_check >= f_low) & (freqs_check < f_high)
            power_before = psd_before[mask].mean()
            power_after = psd_after[mask].mean()
            change = (power_after / power_before - 1) * 100
            status = "✓" if abs(change) < 5 else "⚠"
            log(f"    {band_name:6s} ({f_low:2.0f}-{f_high:2.0f}Hz): {change:+.1f}% {status}")
            
    # =========================================================================
    # 4. チャンネル処理
    # =========================================================================
    log("\n[3/6] チャンネル処理...")
    
    bad_channels = []
    if config.bad_channel_detection:
        bad_channels, _ = detect_bad_channels(
            lfp_filtered, config.channel_order,
            threshold=config.bad_channel_threshold, verbose=config.verbose
        )
    bad_channels = list(set(bad_channels + config.manual_bad_channels))
    
    good_channels = [i for i in range(lfp_filtered.shape[1]) if i not in bad_channels]
    original_ch_numbers = [config.channel_order[i] for i in good_channels]
    
    lfp_trimmed = lfp_trimmed[:, good_channels]
    lfp_filtered = lfp_filtered[:, good_channels]
    n_channels = len(good_channels)
    
    log(f"  除外: {bad_channels}, 残り: {n_channels}ch")
    
    # =========================================================================
    # 5. モーション解析 & ICA
    # =========================================================================
    log("\n[4/6] モーション解析...")
    
    if config.motion_analysis:
        motion_values, roi = analyze_video_motion(
            video_file, roi=config.motion_roi,
            threshold=config.motion_threshold_val, blur=config.motion_blur
        )
        noise_mask, motion_resampled, motion_threshold = create_noise_mask(
            motion_values, frame_times, lfp_times, fs,
            percentile=config.motion_percentile, expand_sec=config.motion_expand_sec
        )
        log(f"  ノイズ区間: {100*np.sum(noise_mask)/len(noise_mask):.1f}%")
    else:
        noise_mask = np.zeros(len(lfp_times), dtype=bool)
        motion_resampled = np.zeros(len(lfp_times))
        motion_threshold = 0.0
        roi = (0, 0, 0, 0)
        log("  スキップ")
    


    log("\n[5/6] ICA...")
    
    if config.ica_enabled and np.sum(noise_mask) > 0:
        lfp_cleaned, removed_ics, noise_ratios, ica_sources = remove_artifact_ica(
            lfp_filtered, noise_mask,
            noise_ratio_threshold=config.ica_noise_ratio_threshold,
            max_remove=config.ica_max_remove, verbose=config.verbose
        )

        log(f"\n成分のノイズ比率 (閾値: {config.ica_noise_ratio_threshold}):")
            # ratioが高い順にソート
        sorted_idx = np.argsort(noise_ratios)[::-1]
        remove_idx = []
        for i in sorted_idx:
            status = ""
            if noise_ratios[i] > config.ica_noise_ratio_threshold and len(remove_idx) < config.ica_max_remove:
                remove_idx.append(i)
                status = "→ 除去"
            log(f"  IC{i}: ratio={noise_ratios[i]:.2f} {status}")

        log(f"  除去成分: {len(removed_ics)}個")
    else:
        lfp_cleaned = lfp_filtered.copy()
        removed_ics, noise_ratios = [], []
        ica_sources = np.zeros((len(lfp_filtered), n_channels))
        log("  スキップ")
    
    # =========================================================================
    # 6. パワー解析
    # =========================================================================
    log("\n[6/6] パワー解析...")
    
    _, stim_times = get_stim_events(seg.events, verbose=config.verbose)
    stim_sessions = stim_times.reshape(config.n_sessions, config.n_stim_per_session)
    session_ranges = [(s[0], s[-1]) for s in stim_sessions]
    
    # マスク作成
    stim_mask = create_stim_mask(lfp_times, session_ranges, margin=config.stim_margin_sec)
    baseline_ranges = [(s - config.baseline_pre_sec, s) for s, _ in session_ranges]
    baseline_mask = create_stim_mask(lfp_times, baseline_ranges, margin=0)
    post_ranges = [(e, e + config.post_duration_sec) for _, e in session_ranges]
    post_mask = create_stim_mask(lfp_times, post_ranges, margin=0)
    
    clean_baseline = baseline_mask & ~noise_mask
    clean_stim = stim_mask & ~noise_mask
    clean_post = post_mask & ~noise_mask


    # === ここに追加 ===
    log("=== マスク確認 ===")
    log(f"clean_baseline sum: {np.sum(clean_baseline)}")
    log(f"clean_stim sum: {np.sum(clean_stim)}")
    log(f"clean_post sum: {np.sum(clean_post)}")
    log(f"\nlfp_cleaned[clean_baseline] mean: {np.mean(lfp_cleaned[clean_baseline]):.6f}")
    log(f"lfp_cleaned[clean_stim] mean: {np.mean(lfp_cleaned[clean_stim]):.6f}")
    log(f"lfp_cleaned[clean_post] mean: {np.mean(lfp_cleaned[clean_post]):.6f}")
    # === ここまで ===


    
    
    # PSD
    freqs, psd_baseline, _ = compute_psd(lfp_cleaned, clean_baseline, fs)
    freqs, psd_stim, _ = compute_psd(lfp_cleaned, clean_stim, fs)
    freqs, psd_post, _ = compute_psd(lfp_cleaned, clean_post, fs)
    
    # バンドパワー
    bands = list(config.bands.keys())
    baseline_power, stim_power, post_power = [], [], []
    change_stim_list, change_post_list = [], []
    
    for name, (low, high) in config.bands.items():
        p_base = compute_band_power(psd_baseline, freqs, (low, high))
        p_stim = compute_band_power(psd_stim, freqs, (low, high))
        p_post = compute_band_power(psd_post, freqs, (low, high))
        baseline_power.append(p_base)
        stim_power.append(p_stim)
        post_power.append(p_post)
        change_stim_list.append((p_stim - p_base) / max(p_base, 1e-10) * 100)
        change_post_list.append((p_post - p_base) / max(p_base, 1e-10) * 100)
    
    # チャンネル別パワー
    channel_band_power = np.zeros((n_channels, len(bands), 3))
    for ch in range(n_channels):
        for b, (name, (low, high)) in enumerate(config.bands.items()):
            f, psd_b, _ = compute_psd(lfp_cleaned[:, ch:ch+1], clean_baseline, fs)
            f, psd_s, _ = compute_psd(lfp_cleaned[:, ch:ch+1], clean_stim, fs)
            f, psd_p, _ = compute_psd(lfp_cleaned[:, ch:ch+1], clean_post, fs)
            channel_band_power[ch, b, 0] = compute_band_power(psd_b, f, (low, high))
            channel_band_power[ch, b, 1] = compute_band_power(psd_s, f, (low, high))
            channel_band_power[ch, b, 2] = compute_band_power(psd_p, f, (low, high))
    
    change_stim_ch = (channel_band_power[:,:,1] - channel_band_power[:,:,0]) / np.maximum(channel_band_power[:,:,0], 1e-10) * 100
    change_post_ch = (channel_band_power[:,:,2] - channel_band_power[:,:,0]) / np.maximum(channel_band_power[:,:,0], 1e-10) * 100
    
    log("  完了")
    
    # =========================================================================
    # 結果格納
    # =========================================================================
    results.update({
        'fs': fs, 'lfp_times': lfp_times, 'lfp_times_full': lfp_times_full,
        'lfp_trimmed': lfp_trimmed, 'lfp_filtered': lfp_filtered, 'lfp_cleaned': lfp_cleaned,
        'good_channels': good_channels, 'bad_channels': bad_channels,
        'original_ch_numbers': original_ch_numbers, 'n_channels': n_channels,
        'noise_mask': noise_mask, 'motion_resampled': motion_resampled,
        'motion_threshold': motion_threshold, 'roi': roi,
        'removed_ics': removed_ics, 'noise_ratios': noise_ratios, 'ica_sources': ica_sources,
        'frame_times': frame_times, 'TRIM_START': TRIM_START, 'TRIM_END': TRIM_END,
        'n_video_frames': n_video_frames, 'n_sync': n_sync,
        'stim_times': stim_times, 'session_ranges': session_ranges,
        'stim_mask': stim_mask, 'baseline_mask': baseline_mask, 'post_mask': post_mask,
        'freqs': freqs, 'psd_baseline': psd_baseline, 'psd_stim': psd_stim, 'psd_post': psd_post,
        'bands': bands, 'baseline_power': baseline_power, 'stim_power': stim_power,
        'post_power': post_power, 'change_stim_list': change_stim_list,
        'change_post_list': change_post_list, 'channel_band_power': channel_band_power,
        'change_stim_ch': change_stim_ch, 'change_post_ch': change_post_ch,
    })
    
    # =========================================================================
    # 保存 & プロット
    # =========================================================================
    log("\n=== 保存 ===")

    # 全チャンネル領域プロット（解析プロットの最初）
    if config.lfp_regions:
        plot_all_channels_with_regions(
            lfp_cleaned, lfp_times,
            stim_mask, baseline_mask, post_mask, noise_mask,
            original_ch_numbers, output_dir, basename,
            t_start=config.plot_t_start,
            t_end=config.plot_t_end,
            show=config.show_plots, save=config.save_plots
        )
        log("  全チャンネル領域プロット")
    
    if config.processing_overview:
        plot_processing_overview(
            lfp_trimmed, lfp_filtered, lfp_cleaned, lfp_times,
            motion_resampled, motion_threshold, noise_mask,
            removed_ics, original_ch_numbers, output_dir, basename, config.show_plots, save=config.save_plots
        )
        log("  処理概要プロット")
    
    if config.edge_check:
        plot_edge_check(lfp_filtered, lfp_cleaned, lfp_times, fs, output_dir, basename, config.show_plots, save=config.save_plots)
        log("  端部効果確認")
    
    if config.ica_components and config.ica_enabled:
        plot_ica_components(ica_sources, noise_ratios, noise_mask, lfp_times, removed_ics, output_dir, 
                            basename, t_start=config.plot_t_start, t_end=config.plot_t_end, show=config.show_plots, save=config.save_plots)
        log("  ICA成分プロット")
    
    if config.power_analysis:
        plot_power_analysis(freqs, psd_baseline, psd_stim, psd_post, bands, baseline_power, stim_power, post_power, change_stim_list, 
                            change_post_list, output_dir, basename, config.power_freq_max, config.show_plots, save=config.save_plots)
        log("  パワー解析プロット")
    
    if config.channel_heatmap:
        plot_channel_heatmap(change_stim_ch, change_post_ch, bands, original_ch_numbers, output_dir, basename, config.show_plots, save=config.save_plots)
        log("  チャンネルヒートマップ")
    
    if config.save_summary_csv:
        save_summary_csv(basename, fs, lfp_times_full, TRIM_START, TRIM_END, n_video_frames, n_sync, roi, bad_channels, good_channels,
                          motion_threshold, noise_mask, removed_ics, noise_ratios, config.n_sessions, config.n_stim_per_session, stim_times,
                            bands, baseline_power, stim_power, post_power, change_stim_list, change_post_list, output_dir)
        log("  サマリーCSV")
    
    if config.save_channel_csv:
        save_channel_csv(basename, n_channels, original_ch_numbers, bands, channel_band_power, change_stim_ch, change_post_ch, output_dir)
        log("  チャンネル別CSV")
    
    if config.save_results_npz:
        save_results_npz(results, output_dir, basename)
        log("  結果NPZ")
    
    if config.save_processed_npz:
        save_processed_npz(lfp_cleaned, lfp_filtered, lfp_trimmed, motion_resampled, ica_sources, noise_ratios, output_dir, basename)
        log("  処理済みNPZ")
    
    if config.create_sync_video:
        create_sync_video(
            video_file, lfp_cleaned, lfp_times, motion_resampled,
            noise_mask, frame_times, roi, motion_threshold,
            output_dir, basename, config.sync_video_start, config.sync_video_end
        )
        log("  同期動画")
    
    # ウェーブレット解析
    if config.wavelet_enabled:
        log("\n=== ウェーブレット解析 ===")
        cwt_power, cwt_freqs = compute_cwt(
            lfp_cleaned, fs,
            freq_min=config.wavelet_freq_min,
            freq_max=config.wavelet_freq_max,
            n_freqs=config.wavelet_n_freqs
        )
        log(f"  CWT計算完了: {cwt_power.shape}")
        
        results['cwt_power'] = cwt_power
        results['cwt_freqs'] = cwt_freqs
        
        if config.wavelet_single:
            plot_wavelet_single(
                cwt_power, cwt_freqs, lfp_times, stim_times,
                original_ch_numbers, config.wavelet_channel,
                output_dir, basename, 
                t_start=config.wavelet_start, t_end=config.wavelet_end,
                show=config.show_plots, save=config.save_plots
            )
            log(f"  単一チャンネル (ch{config.wavelet_channel})")
        
        if config.wavelet_all:
            plot_wavelet_all(
                cwt_power, cwt_freqs, lfp_times, original_ch_numbers,
                output_dir, basename, t_start=config.wavelet_start, t_end=config.wavelet_end,
                show=config.show_plots, save=config.save_plots
            )
            log("  全チャンネル")
    
    log("\n=== 完了 ===")
    return results


if __name__ == "__main__":
    config = PipelineConfig()
    results = run_pipeline(config)
