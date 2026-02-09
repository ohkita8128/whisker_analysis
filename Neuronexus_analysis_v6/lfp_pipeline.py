"""
lfp_pipeline.py - LFP処理パイプライン

LFP Filter GUI からの設定を受け取り、
フィルタリング → Trim → チャンネル処理 → モーション → ICA → パワー解析 → プロット
を一貫して実行する。
"""
import numpy as np
import os
from typing import Dict, Any, Optional
from lfp_filter_gui import LfpConfig
from data_loader import PlxData


def run_lfp_pipeline(config: LfpConfig, plx_data: PlxData,
                     verbose: bool = True) -> Dict[str, Any]:
    """
    LFP処理パイプラインを実行

    Parameters
    ----------
    config : LfpConfig
    plx_data : PlxData
    verbose : bool

    Returns
    -------
    results : dict
    """
    from lfp_processing import (
        bandpass_notch_filter, detect_bad_channels,
        analyze_video_motion, create_noise_mask, remove_artifact_ica,
        compute_psd, compute_band_power, remove_known_harmonics,
        load_noise_reference, remove_environmental_noise
    )
    from lfp_plotting import (
        plot_processing_overview, plot_fft_comparison,
        plot_power_analysis, plot_channel_heatmap,
        plot_ica_components, plot_all_channels_with_regions
    )
    from saving import save_summary_csv, save_channel_csv

    log = print if verbose else lambda *a: None
    results = {}

    # ファイルパス
    output_dir = config.output_dir or plx_data.output_dir
    basename = plx_data.basename
    lfp_raw = plx_data.lfp_raw
    fs = plx_data.lfp_fs
    lfp_times_full = plx_data.lfp_times

    TRIM_START = plx_data.trim_start
    TRIM_END = plx_data.trim_end

    log(f"=== LFPパイプライン開始: {basename} ===")
    log(f"  LFP: {lfp_raw.shape}, fs={fs}Hz")
    log(f"  Trim: {TRIM_START:.1f}~{TRIM_END:.1f}s")

    # =========================================================================
    # 1. フィルタリング
    # =========================================================================
    log("\n[1/6] フィルタリング...")

    if config.filter_enabled:
        notch = config.notch_freq if config.notch_enabled else None
        numtaps = config.filter_fir_numtaps if config.filter_fir_numtaps > 0 else None
        lfp_filtered_full, actual_taps = bandpass_notch_filter(
            lfp_raw, config.filter_lowcut, config.filter_highcut, fs,
            filter_type=config.filter_type,
            order=config.filter_order,
            fir_numtaps=numtaps,
            notch_freq=notch, notch_Q=config.notch_Q
        )
        log(f"  バンドパス: {config.filter_lowcut}-{config.filter_highcut}Hz "
            f"[{config.filter_type.upper()}]")
    else:
        lfp_filtered_full = lfp_raw.copy()
        actual_taps = None
        log("  フィルタ: スキップ")

    # Trim
    idx_s, idx_e = int(TRIM_START * fs), int(TRIM_END * fs)
    lfp_trimmed = lfp_raw[idx_s:idx_e, :]
    lfp_filtered = lfp_filtered_full[idx_s:idx_e, :]
    lfp_times = np.arange(len(lfp_filtered)) / fs + TRIM_START

    # FFT比較
    if config.save_plots:
        plot_fft_comparison(lfp_trimmed, lfp_filtered, fs, output_dir, basename,
                            freq_max=config.fft_freq_max,
                            show=config.show_plots, save=True)

    # 高調波除去
    if config.harmonic_removal_enabled:
        log("\n  高調波ノイズ除去...")
        lfp_filtered, harmonics = remove_known_harmonics(
            lfp_filtered, fs,
            fundamental=config.harmonic_fundamental,
            n_harmonics=config.harmonic_count,
            Q=config.harmonic_q)
        log(f"  除去: {[f'{h:.0f}Hz' for h in harmonics]}")

    # 環境ノイズ除去
    if config.noise_removal_enabled and config.noise_file:
        log("\n  環境ノイズ除去...")
        noise_raw, _ = load_noise_reference(config.noise_file,
                                             plx_data.channel_order, fs)
        noise_filt, _ = bandpass_notch_filter(
            noise_raw, config.filter_lowcut, config.filter_highcut, fs,
            notch_freq=config.notch_freq if config.notch_enabled else None)
        lfp_filtered, noise_peaks, _ = remove_environmental_noise(
            lfp_filtered, noise_filt, fs,
            threshold_db=config.noise_threshold_db)

    # =========================================================================
    # 2. チャンネル処理
    # =========================================================================
    log("\n[2/6] チャンネル処理...")
    bad_channels = []
    if config.bad_channel_detection:
        bad_channels, _ = detect_bad_channels(
            lfp_filtered, plx_data.channel_order,
            threshold=config.bad_channel_threshold, verbose=verbose)

    manual = []
    if config.manual_bad_channels:
        try:
            manual = [int(x.strip()) for x in config.manual_bad_channels.split(',') if x.strip()]
        except:
            pass
    bad_channels = list(set(bad_channels + manual))

    good_channels = [i for i in range(lfp_filtered.shape[1]) if i not in bad_channels]
    original_ch_numbers = [plx_data.channel_order[i] for i in good_channels]

    lfp_trimmed = lfp_trimmed[:, good_channels]
    lfp_filtered = lfp_filtered[:, good_channels]
    n_channels = len(good_channels)
    log(f"  除外: {bad_channels}, 残り: {n_channels}ch")

    # =========================================================================
    # 3. モーション解析
    # =========================================================================
    log("\n[3/6] モーション解析...")
    if config.motion_analysis and plx_data.video_file:
        from lfp_processing import analyze_video_motion, create_noise_mask
        roi = None
        if config.motion_roi:
            try:
                parts = [int(x.strip()) for x in config.motion_roi.split(',')]
                roi = tuple(parts) if len(parts) == 4 else None
            except:
                pass
        motion_values, roi = analyze_video_motion(plx_data.video_file, roi=roi)
        noise_mask, motion_resampled, motion_threshold = create_noise_mask(
            motion_values, plx_data.frame_times, lfp_times, fs,
            percentile=config.motion_percentile,
            expand_sec=config.motion_expand_sec)
        log(f"  ノイズ区間: {100 * np.sum(noise_mask) / len(noise_mask):.1f}%")
    else:
        noise_mask = np.zeros(len(lfp_times), dtype=bool)
        motion_resampled = np.zeros(len(lfp_times))
        motion_threshold = 0.0
        roi = (0, 0, 0, 0)
        log("  スキップ")

    # =========================================================================
    # 4. ICA
    # =========================================================================
    log("\n[4/6] ICA...")
    if config.ica_enabled and np.sum(noise_mask) > 0:
        lfp_cleaned, removed_ics, noise_ratios, ica_sources = remove_artifact_ica(
            lfp_filtered, noise_mask,
            noise_ratio_threshold=config.ica_noise_ratio_threshold,
            max_remove=config.ica_max_remove, verbose=verbose)
        log(f"  除去: {len(removed_ics)} 成分")
    else:
        lfp_cleaned = lfp_filtered.copy()
        removed_ics, noise_ratios = [], []
        ica_sources = np.zeros((len(lfp_filtered), n_channels))
        log("  スキップ")

    # =========================================================================
    # 5. パワー解析
    # =========================================================================
    log("\n[5/6] パワー解析...")

    from lfp_processing import get_stim_events, create_stim_mask
    _, stim_times = get_stim_events(plx_data.segment.events, verbose=verbose)

    if stim_times is None or len(stim_times) == 0:
        log("  警告: 刺激イベントが見つかりません。パワー解析をスキップします。")
        stim_times = np.array([])
        session_ranges = []
    else:
        expected = config.n_sessions * config.n_stim_per_session
        if len(stim_times) != expected:
            log(f"  警告: 刺激数 ({len(stim_times)}) != n_sessions*n_stim ({expected})")
            log(f"  n_sessions={config.n_sessions}, n_stim_per_session={config.n_stim_per_session}")
            # 実際の刺激数からセッション数を再推定
            if config.n_stim_per_session > 0:
                config.n_sessions = len(stim_times) // config.n_stim_per_session
                stim_times = stim_times[:config.n_sessions * config.n_stim_per_session]
                log(f"  再推定: n_sessions={config.n_sessions}, 使用刺激数={len(stim_times)}")
        stim_sessions = stim_times.reshape(config.n_sessions, config.n_stim_per_session)
        session_ranges = [(s[0], s[-1]) for s in stim_sessions]

    stim_mask = create_stim_mask(lfp_times, session_ranges) if session_ranges else np.zeros(len(lfp_times), dtype=bool)
    baseline_ranges = [(s - config.baseline_pre_sec, s) for s, _ in session_ranges]
    baseline_mask = create_stim_mask(lfp_times, baseline_ranges, margin=0) if baseline_ranges else np.zeros(len(lfp_times), dtype=bool)
    post_ranges = [(e, e + config.post_duration_sec) for _, e in session_ranges]
    post_mask = create_stim_mask(lfp_times, post_ranges, margin=0) if post_ranges else np.zeros(len(lfp_times), dtype=bool)

    clean_baseline = baseline_mask & ~noise_mask
    clean_stim = stim_mask & ~noise_mask
    clean_post = post_mask & ~noise_mask

    # PSD
    freqs, psd_baseline, _ = compute_psd(lfp_cleaned, clean_baseline, fs)
    freqs, psd_stim, _ = compute_psd(lfp_cleaned, clean_stim, fs)
    freqs, psd_post, _ = compute_psd(lfp_cleaned, clean_post, fs)

    # バンドパワー
    band_names = list(config.bands.keys())
    baseline_power, stim_power, post_power = [], [], []
    for name, (lo, hi) in config.bands.items():
        baseline_power.append(compute_band_power(psd_baseline, freqs, (lo, hi)))
        stim_power.append(compute_band_power(psd_stim, freqs, (lo, hi)))
        post_power.append(compute_band_power(psd_post, freqs, (lo, hi)))

    # チャンネル別パワー
    channel_band_power = np.zeros((n_channels, len(band_names), 3))
    for ch in range(n_channels):
        for b, (name, (lo, hi)) in enumerate(config.bands.items()):
            f, pb, _ = compute_psd(lfp_cleaned[:, ch:ch + 1], clean_baseline, fs)
            f, ps, _ = compute_psd(lfp_cleaned[:, ch:ch + 1], clean_stim, fs)
            f, pp, _ = compute_psd(lfp_cleaned[:, ch:ch + 1], clean_post, fs)
            channel_band_power[ch, b, 0] = compute_band_power(pb, f, (lo, hi))
            channel_band_power[ch, b, 1] = compute_band_power(ps, f, (lo, hi))
            channel_band_power[ch, b, 2] = compute_band_power(pp, f, (lo, hi))

    change_stim_ch = ((channel_band_power[:, :, 1] - channel_band_power[:, :, 0]) /
                      np.maximum(channel_band_power[:, :, 0], 1e-10) * 100)
    change_post_ch = ((channel_band_power[:, :, 2] - channel_band_power[:, :, 0]) /
                      np.maximum(channel_band_power[:, :, 0], 1e-10) * 100)

    # =========================================================================
    # 6. プロット
    # =========================================================================
    log("\n[6/6] プロット出力...")

    t_s = config.plot_t_start if config.plot_t_start > 0 else None
    t_e = config.plot_t_end if config.plot_t_end > 0 else None

    plot_processing_overview(
        lfp_trimmed, lfp_filtered, lfp_cleaned, lfp_times,
        motion_resampled, motion_threshold, noise_mask,
        removed_ics, original_ch_numbers, output_dir, basename,
        t_start=t_s, t_end=t_e, show=config.show_plots, save=config.save_plots)

    if config.ica_enabled and len(removed_ics) > 0:
        plot_ica_components(
            ica_sources, noise_ratios, noise_mask, lfp_times,
            removed_ics, output_dir, basename,
            t_start=t_s, t_end=t_e, show=config.show_plots, save=config.save_plots)

    # パワー解析 (v4.6スタイル: PSD + 変化率曲線 + 棒グラフ)
    change_stim_list = [((s - b) / max(b, 1e-10) * 100)
                        for b, s in zip(baseline_power, stim_power)]
    change_post_list = [((p - b) / max(b, 1e-10) * 100)
                        for b, p in zip(baseline_power, post_power)]

    plot_power_analysis(
        freqs, psd_baseline, psd_stim, psd_post,
        config.bands, baseline_power, stim_power, post_power,
        change_stim_list, change_post_list,
        output_dir, basename,
        power_freq_min=config.power_freq_min,
        power_freq_max=config.power_freq_max,
        show=config.show_plots, save=config.save_plots)

    # チャンネル別ヒートマップ (v4.6スタイル: seaborn + 数値注釈)
    plot_channel_heatmap(
        change_stim_ch, change_post_ch, config.bands,
        original_ch_numbers, output_dir, basename,
        show=config.show_plots, save=config.save_plots)

    plot_all_channels_with_regions(
        lfp_cleaned, lfp_times, stim_mask, baseline_mask, post_mask,
        noise_mask, original_ch_numbers, output_dir, basename,
        t_start=t_s, t_end=t_e, show=config.show_plots, save=config.save_plots)

    # =========================================================================
    # 結果格納
    # =========================================================================
    results = {
        'fs': fs, 'lfp_times': lfp_times,
        'lfp_trimmed': lfp_trimmed, 'lfp_filtered': lfp_filtered,
        'lfp_cleaned': lfp_cleaned,
        'good_channels': good_channels, 'bad_channels': bad_channels,
        'original_ch_numbers': original_ch_numbers, 'n_channels': n_channels,
        'noise_mask': noise_mask, 'motion_resampled': motion_resampled,
        'motion_threshold': motion_threshold, 'roi': roi,
        'removed_ics': removed_ics, 'noise_ratios': noise_ratios,
        'ica_sources': ica_sources,
        'stim_times': stim_times, 'session_ranges': session_ranges,
        'stim_mask': stim_mask, 'baseline_mask': baseline_mask,
        'post_mask': post_mask,
        'clean_baseline': clean_baseline, 'clean_stim': clean_stim,
        'clean_post': clean_post,
        'freqs': freqs, 'psd_baseline': psd_baseline,
        'psd_stim': psd_stim, 'psd_post': psd_post,
        'bands': config.bands,
        'channel_band_power': channel_band_power,
        'change_stim_ch': change_stim_ch,
        'change_post_ch': change_post_ch,
        'output_dir': output_dir, 'basename': basename,
    }

    # CSV保存
    log("\n  CSV保存...")
    n_sync = min(len(plx_data.frame_times), plx_data.n_video_frames) if plx_data.frame_times is not None and len(plx_data.frame_times) > 0 else 0

    save_summary_csv(
        basename, fs, lfp_times_full, TRIM_START, TRIM_END,
        plx_data.n_video_frames, n_sync, roi,
        bad_channels, good_channels,
        motion_threshold, noise_mask, removed_ics, noise_ratios,
        config.n_sessions, config.n_stim_per_session, stim_times,
        band_names, baseline_power, stim_power, post_power,
        change_stim_list, change_post_list, output_dir)

    save_channel_csv(
        basename, n_channels, original_ch_numbers, band_names,
        channel_band_power, change_stim_ch, change_post_ch, output_dir)

    log("\n=== LFPパイプライン完了 ===")
    return results
