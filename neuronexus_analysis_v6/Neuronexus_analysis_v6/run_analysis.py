"""
run_analysis.py - 統合解析ワークフロー

PLXファイルから位相ロック解析までの完全なパイプライン。
Jupyter Notebook でセルごとに実行、またはスクリプトとして一括実行。

=== 全体フロー ===

  PLX File
    │
    ├─→ [Step 1] データ読み込み (data_loader.py)
    │     └─→ RecordingSession
    │
    ├─→ [Step 2] LFP処理 (pipeline.py / processing.py)
    │     ├─→ バンドパスフィルタ
    │     ├─→ ノッチフィルタ
    │     ├─→ 環境ノイズ除去
    │     ├─→ モーションアーティファクト除去 (ICA)
    │     └─→ lfp_cleaned
    │
    ├─→ [Step 3] スパイクソーティング (spike_sorting.py)
    │     ├─→ 自動ソーティング (GMM + BIC)
    │     ├─→ GUI手動キュレーション (spike_sorting_gui.py)
    │     └─→ sorting_results {ch: ChannelSortResult}
    │
    ├─→ [Step 4] 刺激応答解析 (stimulus.py)
    │     ├─→ PSTH
    │     ├─→ Trial別ラスター
    │     ├─→ 適応解析
    │     └─→ 条件マスク
    │
    ├─→ [Step 5] 位相ロック解析 (spike_lfp_analysis.py)
    │     ├─→ 周波数帯域別 × チャンネル別
    │     ├─→ 条件別（Baseline/Stim/Post）
    │     ├─→ STA (Spike Triggered Average)
    │     └─→ 統合サマリー
    │
    └─→ [Step 6] 保存 & 可視化
          ├─→ CSV (位相ロック結果)
          ├─→ NPZ (ソーティング結果)
          └─→ PNG (サマリープロット)

使い方:
    # === Jupyter Notebook ===
    # 各ステップをセルごとに実行

    # === スクリプト ===
    python run_analysis.py --plx data.plx --output results/
"""

import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# Step 1: データ読み込み
# ============================================================

def step1_load_data(plx_file, channel_order=None, verbose=True):
    """
    PLXファイルを読み込み、RecordingSessionを返す
    
    Parameters
    ----------
    plx_file : str
        PLXファイルパス
    channel_order : list or None
        チャンネル順序。Noneでデフォルト。
    
    Returns
    -------
    session : RecordingSession
    """
    from data_loader import load_plx_session
    
    session = load_plx_session(plx_file, channel_order=channel_order, verbose=verbose)
    return session


# ============================================================
# Step 2: LFP処理
# ============================================================

def step2_process_lfp(session, config=None, use_gui=False, verbose=True):
    """
    LFPの前処理（フィルタ、ノイズ除去、ICA）
    
    Parameters
    ----------
    session : RecordingSession
    config : PipelineConfig or None
        LFPパイプライン設定。Noneの場合はデフォルト。
    use_gui : bool
        Trueの場合、config_gui.pyのGUIで設定
    
    Returns
    -------
    lfp_result : dict
        {
            'lfp_cleaned': ndarray,
            'lfp_filtered': ndarray,
            'lfp_times': ndarray,
            'noise_mask': ndarray,
            'good_channels': list,
            'original_ch_numbers': list,
        }
    """
    from processing import (
        bandpass_notch_filter, detect_bad_channels,
        remove_known_harmonics,
    )
    
    log = print if verbose else lambda *a, **kw: None
    log("\n=== Step 2: LFP Processing ===")
    
    lfp_raw = session.lfp_raw
    fs = session.fs_lfp
    
    # デフォルトパラメータ
    if config is None:
        from pipeline import PipelineConfig
        config = PipelineConfig()
    
    # --- バンドパス + ノッチ ---
    log(f"  Bandpass: {config.filter_lowcut}-{config.filter_highcut} Hz")
    notch = config.notch_freq if config.notch_enabled else None
    lfp_filtered, _ = bandpass_notch_filter(
        lfp_raw, config.filter_lowcut, config.filter_highcut, fs,
        filter_type=config.filter_type,
        order=config.filter_order,
        notch_freq=notch,
        notch_Q=config.notch_Q
    )
    
    # --- 高調波除去 ---
    if config.harmonic_removal_enabled:
        log(f"  Harmonic removal: {config.harmonic_fundamental}Hz fundamental")
        lfp_filtered, harmonics = remove_known_harmonics(
            lfp_filtered, fs,
            fundamental=config.harmonic_fundamental,
            n_harmonics=config.harmonic_count,
            Q=config.harmonic_q
        )
    
    # --- Bad channel detection ---
    bad_channels = []
    if config.bad_channel_detection:
        bad_channels, _ = detect_bad_channels(
            lfp_filtered, session.channel_order,
            threshold=config.bad_channel_threshold, verbose=verbose
        )
    bad_channels = list(set(bad_channels + config.manual_bad_channels))
    
    good_channels = [i for i in range(lfp_filtered.shape[1]) if i not in bad_channels]
    original_ch_numbers = [session.channel_order[i] for i in good_channels]
    
    lfp_filtered = lfp_filtered[:, good_channels]
    
    log(f"  Bad channels: {bad_channels}, remaining: {len(good_channels)}")
    
    # --- 時間軸 ---
    lfp_times = np.arange(len(lfp_filtered)) / fs
    
    # --- ノイズマスク（モーション解析） ---
    noise_mask = np.zeros(len(lfp_times), dtype=bool)
    
    video_file = session.get_video_path()
    if config.motion_analysis and os.path.exists(video_file) and len(session.frame_times) > 0:
        from processing import analyze_video_motion, create_noise_mask, remove_artifact_ica
        
        log("  Motion analysis...")
        motion_values, roi = analyze_video_motion(
            video_file, roi=config.motion_roi,
            threshold=config.motion_threshold_val
        )
        noise_mask, motion_resampled, motion_threshold = create_noise_mask(
            motion_values, session.frame_times, lfp_times, fs,
            percentile=config.motion_percentile
        )
        log(f"  Noise: {100*np.sum(noise_mask)/len(noise_mask):.1f}%")
        
        # ICA
        if config.ica_enabled and np.sum(noise_mask) > 0:
            log("  ICA artifact removal...")
            lfp_cleaned, removed_ics, _, _ = remove_artifact_ica(
                lfp_filtered, noise_mask,
                noise_ratio_threshold=config.ica_noise_ratio_threshold,
                max_remove=config.ica_max_remove, verbose=verbose
            )
            log(f"  Removed {len(removed_ics)} ICA components")
        else:
            lfp_cleaned = lfp_filtered.copy()
    else:
        lfp_cleaned = lfp_filtered.copy()
        log("  Motion analysis: skipped (no video)")
    
    log(f"  Final LFP: {lfp_cleaned.shape}")
    
    return {
        'lfp_cleaned': lfp_cleaned,
        'lfp_filtered': lfp_filtered,
        'lfp_times': lfp_times,
        'noise_mask': noise_mask,
        'good_channels': good_channels,
        'original_ch_numbers': original_ch_numbers,
        'fs': fs,
    }


# ============================================================
# Step 3: スパイクソーティング
# ============================================================

def step3_spike_sorting(session, channels=None, config=None, 
                        use_gui=True, verbose=True):
    """
    スパイクソーティング
    
    Parameters
    ----------
    session : RecordingSession
    channels : list or None
        ソーティングするチャンネル。Noneで全チャンネル。
    config : SortingConfig or None
    use_gui : bool
        TrueでGUIを起動
    
    Returns
    -------
    sorting_results : dict {channel: ChannelSortResult}
    """
    from spike_sorting import sort_channel, sort_all_channels, SortingConfig, bandpass_filter
    
    log = print if verbose else lambda *a, **kw: None
    log("\n=== Step 3: Spike Sorting ===")
    
    if config is None:
        config = SortingConfig()
    
    wideband = session.wideband
    fs = session.fs_wideband
    
    if channels is None:
        channels = list(range(wideband.shape[1]))
    
    # フィルタリング
    log(f"  Bandpass: {config.filter_low}-{config.filter_high} Hz")
    filtered = bandpass_filter(wideband, fs, config.filter_low, config.filter_high, config.filter_order)
    
    # 自動ソーティング
    sorting_results = {}
    for ch in channels:
        log(f"\n  --- Channel {ch} ---")
        result = sort_channel(filtered[:, ch], fs, ch, config, verbose)
        sorting_results[ch] = result
    
    total_units = sum(len(r.units) for r in sorting_results.values())
    total_spikes = sum(sum(u.n_spikes for u in r.units) for r in sorting_results.values())
    log(f"\n  Total: {total_units} units, {total_spikes} spikes")
    
    # GUI
    if use_gui:
        log("\n  Opening GUI for manual curation...")
        from spike_sorting_gui import SpikeSortingGUI
        gui = SpikeSortingGUI(sorting_results)
        gui.run()
        # GUIで編集された結果を取得
        sorting_results = gui.results
    
    return sorting_results


def step3_load_sorting(filepath, verbose=True):
    """
    保存済みソーティング結果を読み込む
    
    Parameters
    ----------
    filepath : str
        NPZファイルパス
    
    Returns
    -------
    sorting_results : dict {channel: ChannelSortResult}
    """
    from data_loader import load_sorting_results
    
    if verbose:
        print(f"\n=== Step 3: Load Sorting Results ===")
    
    return load_sorting_results(filepath)


# ============================================================
# Step 4: 刺激応答解析
# ============================================================

def step4_stimulus_analysis(session, sorting_results, lfp_result=None,
                             verbose=True):
    """
    刺激応答解析
    
    Parameters
    ----------
    session : RecordingSession
    sorting_results : dict
    lfp_result : dict or None
        step2の出力（STAに使用）
    
    Returns
    -------
    protocol : StimulusProtocol
    stim_results : dict {unit_key: {psth, adaptation, ...}}
    """
    from stimulus import StimulusProtocol
    
    log = print if verbose else lambda *a, **kw: None
    log("\n=== Step 4: Stimulus Response Analysis ===")
    
    # プロトコル作成
    protocol = StimulusProtocol(
        stim_times=session.stim_times,
        trial_starts=session.trial_starts,
        n_stim_per_trial=session.n_stim_per_trial,
        stim_freq=session.stim_freq,
        iti=session.iti
    )
    log(f"  {protocol}")
    
    # 各ユニットの解析
    stim_results = {}
    
    for ch, result in sorting_results.items():
        for unit in result.units:
            if unit.is_noise:
                continue
            
            unit_key = f"ch{ch}_unit{unit.unit_id}"
            log(f"\n  {unit_key} (n={unit.n_spikes}):")
            
            # PSTH
            psth = protocol.compute_psth(unit.spike_times)
            log(f"    PSTH: peak={psth.peak_latency_ms:.1f}ms, "
                f"rate={psth.peak_rate:.1f}Hz")
            
            # Adaptation
            adapt = protocol.compute_adaptation(unit.spike_times)
            log(f"    Adaptation ratio: {adapt.adaptation_ratio:.2f}")
            
            stim_results[unit_key] = {
                'psth': psth,
                'adaptation': adapt,
                'spike_times': unit.spike_times,
            }
    
    return protocol, stim_results


# ============================================================
# Step 5: 位相ロック解析
# ============================================================

def step5_phase_locking(sorting_results, lfp_result, protocol=None,
                         freq_bands=None, min_spikes=50, verbose=True):
    """
    位相ロック解析
    
    Parameters
    ----------
    sorting_results : dict
    lfp_result : dict
        step2の出力
    protocol : StimulusProtocol or None
    freq_bands : dict or None
    min_spikes : int
    
    Returns
    -------
    analyzer : SpikeLFPAnalyzer
    """
    from spike_lfp_analysis import SpikeLFPAnalyzer
    
    log = print if verbose else lambda *a, **kw: None
    log("\n=== Step 5: Phase-Locking Analysis ===")
    
    analyzer = SpikeLFPAnalyzer(
        lfp_data=lfp_result['lfp_cleaned'],
        lfp_times=lfp_result['lfp_times'],
        fs_lfp=lfp_result['fs'],
        sorting_results=sorting_results,
        protocol=protocol,
        noise_mask=lfp_result.get('noise_mask'),
        freq_bands=freq_bands,
    )
    
    # 全ユニット解析
    results = analyzer.analyze_all(min_spikes=min_spikes, verbose=verbose)
    
    return analyzer


# ============================================================
# Step 6: 保存 & 可視化
# ============================================================

def step6_save_and_plot(analyzer, sorting_results, protocol, 
                         output_dir, basename=None, verbose=True):
    """
    結果の保存と可視化
    
    Parameters
    ----------
    analyzer : SpikeLFPAnalyzer
    sorting_results : dict
    protocol : StimulusProtocol
    output_dir : str
    basename : str or None
    """
    from spike_sorting import save_sorting_results, export_spike_times_csv
    
    log = print if verbose else lambda *a, **kw: None
    log("\n=== Step 6: Save & Visualize ===")
    
    os.makedirs(output_dir, exist_ok=True)
    basename = basename or "analysis"
    
    # --- ソーティング結果 ---
    sorting_path = os.path.join(output_dir, f'{basename}_sorting.npz')
    save_sorting_results(sorting_results, sorting_path)
    log(f"  Sorting: {sorting_path}")
    
    spikes_csv = os.path.join(output_dir, f'{basename}_spike_times.csv')
    export_spike_times_csv(sorting_results, spikes_csv)
    log(f"  Spike times: {spikes_csv}")
    
    # --- 位相ロック結果 ---
    analyzer.save_results_csv(output_dir, basename)
    
    # --- ユニットサマリープロット ---
    log("\n  Generating summary plots...")
    
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    for unit_key, pl_result in analyzer.unit_results.items():
        save_path = os.path.join(plot_dir, f'{unit_key}_summary.png')
        try:
            analyzer.plot_unit_summary(
                pl_result.channel, pl_result.unit_id,
                save_path=save_path
            )
            import matplotlib.pyplot as plt
            plt.close()
        except Exception as e:
            log(f"    Warning: {unit_key} plot failed: {e}")
    
    # --- 集団サマリー ---
    pop_path = os.path.join(plot_dir, f'{basename}_population_summary.png')
    try:
        analyzer.plot_population_summary(save_path=pop_path)
        import matplotlib.pyplot as plt
        plt.close()
    except Exception as e:
        log(f"    Warning: population plot failed: {e}")
    
    # --- 刺激応答プロット ---
    if protocol is not None:
        for ch, result in sorting_results.items():
            for unit in result.units:
                if unit.is_noise:
                    continue
                unit_key = f"ch{ch}_unit{unit.unit_id}"
                stim_path = os.path.join(plot_dir, f'{unit_key}_stimulus.png')
                try:
                    protocol.plot_summary(
                        unit.spike_times,
                        lfp_data=analyzer.lfp_data,
                        lfp_times=analyzer.lfp_times,
                        fs=analyzer.fs_lfp,
                        save_path=stim_path
                    )
                    import matplotlib.pyplot as plt
                    plt.close()
                except Exception as e:
                    log(f"    Warning: {unit_key} stimulus plot failed: {e}")
    
    log(f"\n  All results saved to: {output_dir}")
    return output_dir


# ============================================================
# 完全パイプライン
# ============================================================

def run_full_pipeline(plx_file, output_dir=None, channel_order=None,
                      sorting_file=None, sort_channels=None,
                      use_gui=False, freq_bands=None,
                      verbose=True):
    """
    全ステップを一括実行
    
    Parameters
    ----------
    plx_file : str
        PLXファイルパス
    output_dir : str or None
        出力ディレクトリ。NoneでPLXと同じ場所。
    channel_order : list or None
        チャンネル順序
    sorting_file : str or None
        保存済みソーティング結果。Noneで新規ソーティング。
    sort_channels : list or None
        ソーティングするチャンネル
    use_gui : bool
        GUIを使うか
    freq_bands : dict or None
        位相ロック解析の周波数帯域
    verbose : bool
    
    Returns
    -------
    results : dict
        全ステップの結果
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(plx_file), 'analysis_output')
    
    basename = os.path.splitext(os.path.basename(plx_file))[0]
    
    # Step 1
    session = step1_load_data(plx_file, channel_order, verbose)
    
    # Step 2
    lfp_result = step2_process_lfp(session, verbose=verbose)
    
    # Step 3
    if sorting_file and os.path.exists(sorting_file):
        sorting_results = step3_load_sorting(sorting_file, verbose)
    else:
        sorting_results = step3_spike_sorting(
            session, channels=sort_channels, use_gui=use_gui, verbose=verbose
        )
    
    # Step 4
    protocol, stim_results = step4_stimulus_analysis(
        session, sorting_results, lfp_result, verbose
    )
    
    # Step 5
    analyzer = step5_phase_locking(
        sorting_results, lfp_result, protocol, freq_bands, verbose=verbose
    )
    
    # Step 6
    step6_save_and_plot(analyzer, sorting_results, protocol, output_dir, basename, verbose)
    
    return {
        'session': session,
        'lfp_result': lfp_result,
        'sorting_results': sorting_results,
        'protocol': protocol,
        'stim_results': stim_results,
        'analyzer': analyzer,
    }


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Neuronexus Analysis Pipeline")
    parser.add_argument("--plx", required=True, help="PLX file path")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--sorting", default=None, help="Pre-sorted NPZ file")
    parser.add_argument("--channels", nargs='+', type=int, default=None,
                       help="Channels to sort")
    parser.add_argument("--gui", action="store_true", help="Use sorting GUI")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    
    args = parser.parse_args()
    
    results = run_full_pipeline(
        plx_file=args.plx,
        output_dir=args.output,
        sorting_file=args.sorting,
        sort_channels=args.channels,
        use_gui=args.gui,
        verbose=not args.quiet,
    )
    
    print("\nDone!")
