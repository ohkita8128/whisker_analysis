"""
saving.py - データ保存関数
CSV, NPZ出力
"""
import numpy as np
import pandas as pd
import os


def save_summary_csv(basename, fs, lfp_times_full, TRIM_START, TRIM_END,
                     n_video_frames, n_sync, roi, bad_channels, good_channels,
                     motion_threshold, noise_mask, removed_ics, noise_ratios,
                     n_sessions, n_stim_per_session, stim_times,
                     bands, baseline_power, stim_power, post_power,
                     change_stim_list, change_post_list, output_dir):
    """解析サマリーCSV"""
    summary = {
        'file_name': basename,
        'sampling_rate_hz': fs,
        'total_recording_sec': lfp_times_full[-1],
        'trim_start_sec': TRIM_START,
        'trim_end_sec': TRIM_END,
        'video_frames': n_video_frames,
        'sync_frames': n_sync,
        'roi': str(roi),
        'bad_channels': str(bad_channels),
        'n_good_channels': len(good_channels),
        'noise_threshold': motion_threshold,
        'noise_ratio_percent': 100 * np.sum(noise_mask) / len(noise_mask),
        'ica_removed': str(removed_ics),
        'n_sessions': n_sessions,
        'total_stim': len(stim_times),
    }
    for i, name in enumerate(bands):
        summary[f'baseline_{name}'] = baseline_power[i]
        summary[f'stim_{name}'] = stim_power[i]
        summary[f'post_{name}'] = post_power[i]
        summary[f'stim_{name}_change'] = change_stim_list[i]
        summary[f'post_{name}_change'] = change_post_list[i]
    
    df = pd.DataFrame([summary])
    df.to_csv(os.path.join(output_dir, f'{basename}_summary.csv'), index=False)


def save_channel_csv(basename, n_channels, original_ch_numbers, bands,
                     channel_band_power, change_stim_ch, change_post_ch, output_dir):
    """チャンネル別パワーCSV"""
    rows = []
    for ch in range(n_channels):
        for b, band in enumerate(bands):
            rows.append({
                'file': basename,
                'depth_idx': ch,
                'original_ch': original_ch_numbers[ch],
                'band': band,
                'baseline': channel_band_power[ch, b, 0],
                'stim': channel_band_power[ch, b, 1],
                'post': channel_band_power[ch, b, 2],
                'stim_change': change_stim_ch[ch, b],
                'post_change': change_post_ch[ch, b],
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, f'{basename}_channel_power.csv'), index=False)


def save_results_npz(results, output_dir, basename):
    """解析結果NPZ保存"""
    save_keys = ['fs', 'good_channels', 'original_ch_numbers', 'lfp_times',
                 'stim_times', 'stim_mask', 'baseline_mask', 'post_mask', 'noise_mask',
                 'freqs', 'psd_baseline', 'psd_stim', 'psd_post',
                 'bands', 'channel_band_power', 'change_stim_ch', 'change_post_ch']
    save_data = {k: np.array(results[k]) for k in save_keys if k in results}
    np.savez(os.path.join(output_dir, f'{basename}_results.npz'), **save_data)


def save_processed_npz(lfp_cleaned, lfp_filtered, lfp_trimmed,
                       motion_resampled, ica_sources, noise_ratios,
                       output_dir, basename):
    """処理済みデータNPZ保存"""
    np.savez_compressed(os.path.join(output_dir, f'{basename}_processed.npz'),
                        lfp_cleaned=lfp_cleaned,
                        lfp_filtered=lfp_filtered,
                        lfp_trimmed=lfp_trimmed,
                        motion_resampled=motion_resampled,
                        ica_sources=ica_sources,
                        noise_ratios=np.array(noise_ratios))
