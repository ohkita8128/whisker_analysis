"""
lfp_plotting.py - LFP関連プロットの統合モジュール

v4.6スタイルを基に、以下のプロットを生成:
1. 処理概要 (Raw→Filtered→ICA cleaned, モーション)
2. FFT比較 (フィルタ前後)
3. パワー解析 (PSD + 変化率 + バンドパワー棒グラフ + バンドパワー変化率)
4. チャンネル別ヒートマップ (seaborn)
5. ICA成分
6. 全チャンネル波形 + 領域マーク
7. 端部効果確認
"""
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.family'] = 'MS Gothic'

# 共通フォント設定
_FONT_PARAMS = {
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
}

# 帯域色 (GUIのBandEditorFrameと同じ順番)
BAND_COLORS = ['#3b82f6', '#22c55e', '#f59e0b', '#a855f7', '#ef4444',
               '#06b6d4', '#ec4899', '#84cc16']


# ============================================================
# 1. 処理概要プロット (v4.6スタイル: 生振幅 + 固定スペーシング)
# ============================================================
def plot_processing_overview(lfp_trimmed, lfp_filtered, lfp_cleaned, lfp_times,
                              motion_resampled, motion_threshold, noise_mask,
                              removed_ics, original_ch_numbers,
                              output_dir, basename, t_start=None, t_end=None,
                              show=True, save=True):
    """処理概要 (4段: モーション, Raw, Filtered, Cleaned)"""
    n_channels = lfp_cleaned.shape[1]
    if t_start is None:
        t_start = lfp_times[0]
    if t_end is None:
        t_end = lfp_times[-1]
    t_mask = (lfp_times >= t_start) & (lfp_times <= t_end)
    tp = lfp_times[t_mask]

    plt.rcParams.update(_FONT_PARAMS)
    fig, axes = plt.subplots(4, 1, figsize=(15, 8), sharex=True)
    fig.suptitle(f'{basename} - Processing Overview', fontsize=18)

    # モーション
    axes[0].plot(tp, motion_resampled[t_mask], 'b-', lw=0.5)
    axes[0].axhline(motion_threshold, color='r', ls='--')
    axes[0].fill_between(tp, 0, motion_resampled[t_mask].max(),
                         where=noise_mask[t_mask], alpha=0.3, color='purple')
    axes[0].set_ylabel('Motion')
    axes[0].set_title('Motion Detection')

    # Raw / Filtered / Cleaned (データ振幅に応じた自動スペーシング)
    for row_idx, (data, ylabel, title) in enumerate([
        (lfp_trimmed, 'LFP', 'Raw LFP'),
        (lfp_filtered, 'LFP', 'Filtered LFP'),
        (lfp_cleaned, 'LFP', f'After ICA (removed {len(removed_ics)} artifacts)')
    ], start=1):
        ax = axes[row_idx]
        d = data[t_mask]
        # 全チャンネルの振幅からスペーシングを自動決定
        amp = np.percentile(np.abs(d), 95)
        spacing = amp * 2.5 if amp > 0 else 1.0
        for idx in range(n_channels):
            ax.plot(tp, d[:, idx] - idx * spacing, lw=0.5)
        if row_idx == 3:
            y_bot = -n_channels * spacing
            ax.fill_between(tp, y_bot, spacing,
                            where=noise_mask[t_mask], alpha=0.3, color='purple')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_yticks([])

    for ax in axes:
        ax.set_xlim(t_start, t_end)

    axes[3].set_xlabel('Time (s)')

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(output_dir, f'{basename}_processing_overview.png'),
                    dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


# ============================================================
# 2. FFT比較 (フィルタ前後)
# ============================================================
def plot_fft_comparison(lfp_raw_trim, lfp_filtered, fs, output_dir, basename,
                        freq_max=300.0, show=True, save=True):
    """フィルタ前後のFFT比較"""
    from scipy.fft import fft, fftfreq

    N = len(lfp_raw_trim)
    freqs_fft = fftfreq(N, 1 / fs)[:N // 2]

    # チャンネル平均
    fft_raw = np.abs(fft(lfp_raw_trim.mean(axis=1)))[:N // 2]
    fft_filt = np.abs(fft(lfp_filtered.mean(axis=1)))[:N // 2]

    # 正規化
    fft_raw = fft_raw / (fft_raw.max() + 1e-30)
    fft_filt = fft_filt / (fft_filt.max() + 1e-30)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.semilogy(freqs_fft, fft_raw, alpha=0.7, label='Raw', lw=0.8)
    ax.semilogy(freqs_fft, fft_filt, alpha=0.7, label='Filtered', lw=0.8)
    ax.set_xlim(0, freq_max)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude (normalized)')
    ax.set_title('FFT Comparison: Raw vs Filtered')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(output_dir, f'{basename}_fft_comparison.png'),
                    dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


# ============================================================
# 3. パワー解析 (v4.6スタイル: PSD + 変化率 + 棒グラフ + 変化率棒グラフ)
# ============================================================
def plot_power_analysis(freqs, psd_baseline, psd_stim, psd_post,
                        bands, baseline_power, stim_power, post_power,
                        change_stim_list, change_post_list,
                        output_dir, basename,
                        power_freq_min=0.5, power_freq_max=100.0,
                        show=True, save=True):
    """
    パワースペクトル解析プロット (v4.6スタイル, 2x2)
    左上: PSD (帯域背景付き)
    右上: 変化率 (帯域背景付き)
    左下: バンドパワー棒グラフ
    右下: バンドパワー変化率
    """
    # --- 安全なデフォルト値 ---
    if power_freq_min is None or power_freq_min <= 0:
        power_freq_min = 0.5
    if power_freq_max is None or power_freq_max <= 0:
        power_freq_max = 100.0
    if power_freq_min >= power_freq_max:
        power_freq_min = 0.5
        power_freq_max = 100.0

    # bandsがリストの場合は辞書に変換
    if isinstance(bands, list):
        DEFAULT_BANDS = {
            'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 14),
            'beta': (14, 30), 'gamma': (30, 80),
            'low_gamma': (30, 60), 'high_gamma': (60, 120),
            'low': (1, 30), 'high': (30, 100)
        }
        bands_dict = {}
        for name in bands:
            if name in DEFAULT_BANDS:
                bands_dict[name] = DEFAULT_BANDS[name]
            else:
                bands_dict[name] = (0, 100)
        bands = bands_dict

    band_names = list(bands.keys())

    plt.rcParams.update(_FONT_PARAMS)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # ===================
    # 左上: PSD (帯域背景付き)
    # ===================
    ax = axes[0, 0]
    for i, (band_name, (f_low, f_high)) in enumerate(bands.items()):
        color = BAND_COLORS[i % len(BAND_COLORS)]
        ax.axvspan(f_low, f_high, alpha=0.25, color=color)
    ax.semilogy(freqs, psd_baseline, 'b-', lw=2, label='Baseline')
    ax.semilogy(freqs, psd_stim, 'r-', lw=2, label='Stim')
    ax.semilogy(freqs, psd_post, 'g-', lw=2, label='Post')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power')
    ax.set_title('Power Spectrum')
    ax.set_xlim(power_freq_min, power_freq_max)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # ===================
    # 右上: 変化率 + 帯域背景
    # ===================
    ax = axes[0, 1]
    change_stim_psd = (psd_stim - psd_baseline) / np.maximum(psd_baseline, 1e-10) * 100
    change_post_psd = (psd_post - psd_baseline) / np.maximum(psd_baseline, 1e-10) * 100

    freq_mask = (freqs >= power_freq_min) & (freqs <= power_freq_max)
    freqs_plot = freqs[freq_mask]
    change_stim_plot = change_stim_psd[freq_mask]
    change_post_plot = change_post_psd[freq_mask]

    for i, (band_name, (f_low, f_high)) in enumerate(bands.items()):
        color = BAND_COLORS[i % len(BAND_COLORS)]
        ax.axvspan(f_low, f_high, alpha=0.25, color=color)

    ax.plot(freqs_plot, change_stim_plot, 'r-', lw=2, label='Stim')
    ax.plot(freqs_plot, change_post_plot, 'g-', lw=2, label='Post')
    ax.axhline(0, color='k', ls='--')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Change (%)')
    ax.set_title('Power Change')
    ax.set_xlim(power_freq_min, power_freq_max)

    # Y軸範囲を自動調整
    if len(change_stim_plot) > 0 and len(change_post_plot) > 0:
        y_max = min(300, max(np.max(change_stim_plot), np.max(change_post_plot)) * 1.2)
        y_min = max(-100, min(np.min(change_stim_plot), np.min(change_post_plot)) * 1.2)
        ax.set_ylim(y_min, y_max)
    else:
        ax.set_ylim(-100, 300)

    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # ===================
    # 左下: バンドパワー棒グラフ
    # ===================
    ax = axes[1, 0]
    x = np.arange(len(band_names))
    width = 0.25
    ax.bar(x - width, baseline_power, width, label='Baseline', color='blue', alpha=0.7)
    ax.bar(x, stim_power, width, label='Stim', color='red', alpha=0.7)
    ax.bar(x + width, post_power, width, label='Post', color='green', alpha=0.7)
    ax.set_ylabel('Power')
    ax.set_title('Band Power')
    ax.set_xticks(x)
    ax.set_xticklabels(band_names, rotation=45, ha='right')
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)

    # ===================
    # 右下: バンドパワー変化率
    # ===================
    ax = axes[1, 1]
    ax.bar(x - width / 2, change_stim_list, width, label='Stim', color='red', alpha=0.7)
    ax.bar(x + width / 2, change_post_list, width, label='Post', color='green', alpha=0.7)
    ax.axhline(0, color='k', lw=1)
    ax.set_ylabel('Change (%)')
    ax.set_title('Band Power Change')
    ax.set_xticks(x)
    ax.set_xticklabels(band_names, rotation=45, ha='right')
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(output_dir, f'{basename}_power_analysis.png'),
                    dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


# ============================================================
# 4. チャンネル別ヒートマップ (v4.6スタイル: seaborn + 注釈)
# ============================================================
def plot_channel_heatmap(change_stim_ch, change_post_ch, bands,
                         original_ch_numbers,
                         output_dir, basename, show=True, save=True):
    """チャンネル×バンドのヒートマップ (seaborn, 数値注釈付き)"""
    try:
        import seaborn as sns
    except ImportError:
        # seabornがない場合はimshowにフォールバック
        _plot_channel_heatmap_fallback(change_stim_ch, change_post_ch, bands,
                                       original_ch_numbers, output_dir, basename,
                                       show, save)
        return

    n_channels = len(original_ch_numbers)
    if isinstance(bands, dict):
        band_names = list(bands.keys())
    else:
        band_names = bands

    ch_labels = [f'D{i}(Ch{original_ch_numbers[i]})' for i in range(n_channels)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    sns.heatmap(change_stim_ch, ax=axes[0], cmap='RdBu_r', center=0,
                xticklabels=band_names, yticklabels=ch_labels,
                annot=True, fmt='.0f', cbar_kws={'label': '%'})
    axes[0].set_title('Stim vs Baseline')

    sns.heatmap(change_post_ch, ax=axes[1], cmap='RdBu_r', center=0,
                xticklabels=band_names, yticklabels=ch_labels,
                annot=True, fmt='.0f', cbar_kws={'label': '%'})
    axes[1].set_title('Post vs Baseline')

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(output_dir, f'{basename}_channel_heatmap.png'),
                    dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def _plot_channel_heatmap_fallback(change_stim_ch, change_post_ch, bands,
                                    original_ch_numbers,
                                    output_dir, basename, show=True, save=True):
    """seabornがない場合のフォールバック (imshow)"""
    n_channels = len(original_ch_numbers)
    if isinstance(bands, dict):
        band_names = list(bands.keys())
    else:
        band_names = bands

    ch_labels = [f'D{i}(Ch{original_ch_numbers[i]})' for i in range(n_channels)]
    vmax = max(np.abs(change_stim_ch).max(), np.abs(change_post_ch).max(), 1)
    vmax = min(vmax, 200)

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    im0 = axes[0].imshow(change_stim_ch, aspect='auto', cmap='RdBu_r',
                          vmin=-vmax, vmax=vmax, interpolation='nearest')
    axes[0].set_yticks(range(n_channels))
    axes[0].set_yticklabels(ch_labels, fontsize=8)
    axes[0].set_xticks(range(len(band_names)))
    axes[0].set_xticklabels(band_names)
    axes[0].set_title('Stim vs Baseline')
    plt.colorbar(im0, ax=axes[0], label='%')
    # 数値注釈
    for i in range(n_channels):
        for j in range(len(band_names)):
            axes[0].text(j, i, f'{change_stim_ch[i, j]:.0f}',
                         ha='center', va='center', fontsize=7)

    im1 = axes[1].imshow(change_post_ch, aspect='auto', cmap='RdBu_r',
                          vmin=-vmax, vmax=vmax, interpolation='nearest')
    axes[1].set_yticks(range(n_channels))
    axes[1].set_yticklabels(ch_labels, fontsize=8)
    axes[1].set_xticks(range(len(band_names)))
    axes[1].set_xticklabels(band_names)
    axes[1].set_title('Post vs Baseline')
    plt.colorbar(im1, ax=axes[1], label='%')
    for i in range(n_channels):
        for j in range(len(band_names)):
            axes[1].text(j, i, f'{change_post_ch[i, j]:.0f}',
                         ha='center', va='center', fontsize=7)

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(output_dir, f'{basename}_channel_heatmap.png'),
                    dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


# ============================================================
# 5. ICA成分プロット (v4.6スタイル: グリッド表示)
# ============================================================
def plot_ica_components(ica_sources, noise_ratios, noise_mask, lfp_times,
                        removed_ics, output_dir, basename,
                        t_start=None, t_end=None, show=True, save=True):
    """ICA成分 (グリッド表示, v4.6スタイル)"""
    if t_start is None:
        t_start = lfp_times[0]
    if t_end is None:
        t_end = lfp_times[-1]
    t_mask = (lfp_times >= t_start) & (lfp_times <= t_end)

    n_components = ica_sources.shape[1]
    n_cols = 4
    n_rows = (n_components + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 2.5 * n_rows), sharex=True)
    axes = axes.flatten()

    for i in range(n_components):
        ax = axes[i]
        ax.plot(lfp_times[t_mask], ica_sources[t_mask, i], 'k-', lw=0.5)
        ax.fill_between(lfp_times[t_mask], ica_sources[t_mask, i].min(),
                        ica_sources[t_mask, i].max(), where=noise_mask[t_mask],
                        alpha=0.3, color='purple')
        is_removed = i in removed_ics
        color = 'red' if is_removed else 'black'
        ax.set_title(f'IC{i} ({noise_ratios[i]:.2f})' + (' *' if is_removed else ''),
                     color=color)

    for i in range(n_components, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(output_dir, f'{basename}_ica_components.png'),
                    dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


# ============================================================
# 6. 全チャンネル波形 + 領域マーク (v4.6スタイル)
# ============================================================
def plot_all_channels_with_regions(lfp_cleaned, lfp_times,
                                   stim_mask, baseline_mask, post_mask,
                                   noise_mask, original_ch_numbers,
                                   output_dir, basename,
                                   t_start=None, t_end=None,
                                   show=True, save=True):
    """全チャンネルLFP + baseline/stim/post領域を色分け (v4.6スタイル)"""
    n_ch = lfp_cleaned.shape[1]
    if t_start is None:
        t_start = lfp_times[0]
    if t_end is None:
        t_end = lfp_times[-1]
    t_mask = (lfp_times >= t_start) & (lfp_times <= t_end)
    tp = lfp_times[t_mask]

    # データ振幅に応じた自動スペーシング
    d = lfp_cleaned[t_mask]
    amp = np.percentile(np.abs(d), 95)
    spacing = amp * 2.5 if amp > 0 else 1.0
    ymin = -n_ch * spacing - spacing * 0.5
    ymax = spacing * 0.5

    fig, ax = plt.subplots(figsize=(16, 10))

    # 背景色
    ax.fill_between(tp, ymin, ymax, where=baseline_mask[t_mask],
                    color='blue', alpha=0.2, label='Baseline')
    ax.fill_between(tp, ymin, ymax, where=stim_mask[t_mask],
                    color='red', alpha=0.3, label='Stim')
    ax.fill_between(tp, ymin, ymax, where=post_mask[t_mask],
                    color='green', alpha=0.2, label='Post')
    ax.fill_between(tp, ymin, ymax, where=noise_mask[t_mask],
                    color='gray', alpha=0.5, label='Noise')

    # LFP波形 (全チャンネル, 生振幅)
    for idx in range(n_ch):
        ax.plot(tp, lfp_cleaned[t_mask, idx] - idx * spacing, 'k-', lw=0.5)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Channel')
    ax.set_ylim(ymin, ymax)
    ax.set_yticks([-idx * spacing for idx in range(n_ch)])
    ax.set_yticklabels([f'D{idx}(Ch{original_ch_numbers[idx]})' for idx in range(n_ch)])
    ax.set_title('All Channels LFP with Region Markers')
    ax.legend(loc='upper right', ncol=4)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(output_dir, f'{basename}_all_channels.png'),
                    dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


# ============================================================
# 7. 端部効果確認
# ============================================================
def plot_edge_check(lfp_filtered, lfp_cleaned, lfp_times, fs,
                    output_dir, basename, t_start=None, t_end=None,
                    show=True, save=True):
    """端部効果の確認 (v4.6スタイル)"""
    if t_start is None:
        t_start = lfp_times[0]
    if t_end is None:
        t_end = lfp_times[-1]
    t_mask = (lfp_times >= t_start) & (lfp_times <= t_end)
    times_plot = lfp_times[t_mask]
    filtered_plot = lfp_filtered[t_mask]
    cleaned_plot = lfp_cleaned[t_mask]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    t_check = 2.0

    for i, (ax, title) in enumerate(zip(axes[0], ['Start (2s)', 'End (2s)'])):
        if i == 0:
            edge_mask = times_plot < times_plot[0] + t_check
        else:
            edge_mask = times_plot > times_plot[-1] - t_check
        ax.plot(times_plot[edge_mask], filtered_plot[edge_mask, 0], 'b-', lw=1, label='Filtered')
        ax.plot(times_plot[edge_mask], cleaned_plot[edge_mask, 0], 'r-', lw=1, alpha=0.7, label='ICA')
        ax.set_title(title)
        ax.legend()

    axes[1, 0].plot(times_plot, filtered_plot[:, 0], 'b-', lw=0.3)
    axes[1, 0].set_title('Filtered - Full')
    axes[1, 0].set_xlim(t_start, t_end)
    axes[1, 1].plot(times_plot, cleaned_plot[:, 0], 'r-', lw=0.3)
    axes[1, 1].set_title('ICA Cleaned - Full')
    axes[1, 1].set_xlim(t_start, t_end)

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(output_dir, f'{basename}_edge_check.png'),
                    dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
