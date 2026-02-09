"""
lfp_plotting.py - LFP関連プロットの統合モジュール

枚数を抑えてまとめるため、以下のプロットを生成:
1. 処理概要 (Raw→Filtered→ICA cleaned, モーション)
2. FFT比較 (フィルタ前後)
3. パワー解析サマリー (PSD + バンドパワー + ヒートマップ を1枚に統合)
4. ICA成分
5. 全チャンネル波形 + 領域マーク
"""
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']


# ============================================================
# 1. 処理概要プロット
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

    fig, axes = plt.subplots(4, 1, figsize=(15, 8), sharex=True)
    fig.suptitle(f'{basename} - Processing Overview', fontsize=13)

    axes[0].plot(tp, motion_resampled[t_mask], 'b-', lw=0.5)
    axes[0].axhline(motion_threshold, color='r', ls='--')
    axes[0].fill_between(tp, 0, motion_resampled[t_mask].max(),
                         where=noise_mask[t_mask], alpha=0.3, color='purple')
    axes[0].set_ylabel('Motion')
    axes[0].set_title('Motion Detection')

    for idx in range(n_channels):
        axes[1].plot(tp, lfp_trimmed[t_mask, idx] - idx * 0.3, lw=0.3)
    axes[1].set_ylabel('Raw LFP')

    for idx in range(n_channels):
        axes[2].plot(tp, lfp_filtered[t_mask, idx] - idx * 0.3, lw=0.3)
    axes[2].set_ylabel('Filtered LFP')

    for idx in range(n_channels):
        axes[3].plot(tp, lfp_cleaned[t_mask, idx] - idx * 0.3, lw=0.3)
    axes[3].fill_between(tp, axes[3].get_ylim()[0], axes[3].get_ylim()[1],
                         where=noise_mask[t_mask], alpha=0.2, color='purple')
    axes[3].set_ylabel('ICA Cleaned')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_title(f'Cleaned (removed {len(removed_ics)} ICs)')

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
    from scipy.signal import welch

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{basename} - FFT Comparison (Filter Effect)', fontsize=12)

    nperseg = min(fs * 4, len(lfp_raw_trim) // 2)
    f_raw, psd_raw = welch(lfp_raw_trim.mean(axis=1), fs, nperseg=nperseg)
    f_flt, psd_flt = welch(lfp_filtered.mean(axis=1), fs, nperseg=nperseg)

    mask = f_raw <= freq_max
    axes[0].semilogy(f_raw[mask], psd_raw[mask], 'k-', lw=0.8, alpha=0.7, label='Before')
    axes[0].semilogy(f_flt[mask], psd_flt[mask], 'b-', lw=0.8, label='After')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('PSD (V²/Hz)')
    axes[0].set_title('PSD Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # dB差
    ratio = psd_flt[mask] / (psd_raw[mask] + 1e-30)
    db_change = 10 * np.log10(ratio + 1e-30)
    axes[1].plot(f_raw[mask], db_change, 'r-', lw=0.8)
    axes[1].axhline(0, color='gray', ls='--')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Change (dB)')
    axes[1].set_title('Attenuation')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(output_dir, f'{basename}_fft_comparison.png'),
                    dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


# ============================================================
# 3. パワー解析サマリー (PSD + バンドパワー + ヒートマップ 統合)
# ============================================================
def plot_power_summary(freqs, psd_baseline, psd_stim, psd_post,
                       bands, baseline_power, stim_power, post_power,
                       change_stim_ch, change_post_ch,
                       original_ch_numbers,
                       output_dir, basename,
                       freq_min=0.5, freq_max=100.0,
                       show=True, save=True):
    """
    パワー解析を1枚にまとめた統合プロット
    左上: PSD (条件別), 右上: バンドパワー棒グラフ
    左下: Stim変化ヒートマップ, 右下: Post変化ヒートマップ
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    fig.suptitle(f'{basename} - Power Analysis Summary', fontsize=13)

    band_names = list(bands.keys())
    n_bands = len(band_names)
    COLORS = ['#3b82f6', '#22c55e', '#f59e0b', '#a855f7', '#ef4444',
              '#06b6d4', '#ec4899', '#84cc16']

    # --- 左上: PSD ---
    ax1 = fig.add_subplot(gs[0, 0])
    fmask = (freqs >= freq_min) & (freqs <= freq_max)
    ax1.semilogy(freqs[fmask], psd_baseline[fmask], 'k-', lw=1.2, label='Baseline')
    ax1.semilogy(freqs[fmask], psd_stim[fmask], 'r-', lw=1.2, label='Stim', alpha=0.8)
    ax1.semilogy(freqs[fmask], psd_post[fmask], 'b-', lw=1.2, label='Post', alpha=0.8)
    for i, (name, (lo, hi)) in enumerate(bands.items()):
        ax1.axvspan(lo, hi, alpha=0.1, color=COLORS[i % len(COLORS)])
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('PSD')
    ax1.set_title('Power Spectral Density')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- 右上: バンドパワー棒グラフ ---
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(n_bands)
    w = 0.25
    bars_b = ax2.bar(x - w, baseline_power, w, label='Baseline', color='gray')
    bars_s = ax2.bar(x, stim_power, w, label='Stim', color='red', alpha=0.7)
    bars_p = ax2.bar(x + w, post_power, w, label='Post', color='blue', alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(band_names, rotation=30)
    ax2.set_ylabel('Mean Band Power')
    ax2.set_title('Band Power Comparison')
    ax2.legend(fontsize=8)

    # --- 下段: ヒートマップ ---
    ch_labels = [f'Ch{ch}' for ch in original_ch_numbers]
    vmax = max(np.abs(change_stim_ch).max(), np.abs(change_post_ch).max(), 1)
    vmax = min(vmax, 200)  # 上限

    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.imshow(change_stim_ch.T, aspect='auto', cmap='RdBu_r',
                     vmin=-vmax, vmax=vmax, interpolation='nearest')
    ax3.set_xticks(range(len(ch_labels)))
    ax3.set_xticklabels(ch_labels, rotation=45, fontsize=8)
    ax3.set_yticks(range(n_bands))
    ax3.set_yticklabels(band_names)
    ax3.set_xlabel('Channel')
    ax3.set_ylabel('Band')
    ax3.set_title('Stim Change (%)')
    plt.colorbar(im3, ax=ax3, label='%')

    ax4 = fig.add_subplot(gs[1, 1])
    im4 = ax4.imshow(change_post_ch.T, aspect='auto', cmap='RdBu_r',
                     vmin=-vmax, vmax=vmax, interpolation='nearest')
    ax4.set_xticks(range(len(ch_labels)))
    ax4.set_xticklabels(ch_labels, rotation=45, fontsize=8)
    ax4.set_yticks(range(n_bands))
    ax4.set_yticklabels(band_names)
    ax4.set_xlabel('Channel')
    ax4.set_title('Post Change (%)')
    plt.colorbar(im4, ax=ax4, label='%')

    if save:
        plt.savefig(os.path.join(output_dir, f'{basename}_power_summary.png'),
                    dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


# ============================================================
# 4. ICA成分プロット
# ============================================================
def plot_ica_components(ica_sources, noise_ratios, noise_mask, lfp_times,
                        removed_ics, output_dir, basename,
                        t_start=None, t_end=None, show=True, save=True):
    """ICA成分 (上位成分のみ表示)"""
    n_ics = ica_sources.shape[1]
    n_show = min(n_ics, 8)

    if t_start is None:
        t_start = lfp_times[0]
    if t_end is None:
        t_end = lfp_times[-1]
    t_mask = (lfp_times >= t_start) & (lfp_times <= t_end)
    tp = lfp_times[t_mask]

    sorted_idx = np.argsort(noise_ratios)[::-1][:n_show]

    fig, axes = plt.subplots(n_show, 1, figsize=(14, n_show * 1.2), sharex=True)
    fig.suptitle(f'{basename} - ICA Components', fontsize=12)
    if n_show == 1:
        axes = [axes]

    for i, ic_idx in enumerate(sorted_idx):
        color = 'red' if ic_idx in removed_ics else 'black'
        axes[i].plot(tp, ica_sources[t_mask, ic_idx], color=color, lw=0.3)
        axes[i].fill_between(tp, axes[i].get_ylim()[0], axes[i].get_ylim()[1],
                             where=noise_mask[t_mask], alpha=0.2, color='purple')
        label = f'IC{ic_idx} (ratio={noise_ratios[ic_idx]:.2f})'
        if ic_idx in removed_ics:
            label += ' ★REMOVED'
        axes[i].set_ylabel(label, fontsize=8)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(output_dir, f'{basename}_ica_components.png'),
                    dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


# ============================================================
# 5. 全チャンネル波形 + 領域マーク
# ============================================================
def plot_all_channels_with_regions(lfp_cleaned, lfp_times,
                                   stim_mask, baseline_mask, post_mask,
                                   noise_mask, original_ch_numbers,
                                   output_dir, basename,
                                   t_start=None, t_end=None,
                                   show=True, save=True):
    """全チャンネルLFP + baseline/stim/post領域を色分け"""
    n_ch = lfp_cleaned.shape[1]
    if t_start is None:
        t_start = lfp_times[0]
    if t_end is None:
        t_end = lfp_times[-1]
    t_mask = (lfp_times >= t_start) & (lfp_times <= t_end)
    tp = lfp_times[t_mask]

    fig, ax = plt.subplots(figsize=(16, max(6, n_ch * 0.5)))
    fig.suptitle(f'{basename} - All Channels with Regions', fontsize=12)

    # 領域の背景色
    ax.fill_between(tp, -n_ch, 1, where=baseline_mask[t_mask],
                    alpha=0.08, color='green', label='Baseline')
    ax.fill_between(tp, -n_ch, 1, where=stim_mask[t_mask],
                    alpha=0.08, color='red', label='Stim')
    ax.fill_between(tp, -n_ch, 1, where=post_mask[t_mask],
                    alpha=0.08, color='blue', label='Post')
    ax.fill_between(tp, -n_ch, 1, where=noise_mask[t_mask],
                    alpha=0.15, color='purple', label='Noise')

    # 波形
    spacing = 0.3
    for idx in range(n_ch):
        offset = -idx * spacing
        sig = lfp_cleaned[t_mask, idx]
        sig_norm = sig / (np.std(sig) * 5 + 1e-10)  # スケール正規化
        ax.plot(tp, sig_norm + offset, lw=0.3, color='black')
        ax.text(tp[0] - (tp[-1] - tp[0]) * 0.02, offset,
                f'Ch{original_ch_numbers[idx]}', fontsize=7, va='center', ha='right')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Channel')
    ax.set_yticks([])
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(output_dir, f'{basename}_all_channels.png'),
                    dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


# ============================================================
# 端部効果確認
# ============================================================
def plot_edge_check(lfp_filtered, lfp_cleaned, lfp_times, fs,
                    output_dir, basename, t_start=None, t_end=None,
                    show=True, save=True):
    """端部効果の確認"""
    n_edge = int(2 * fs)

    fig, axes = plt.subplots(2, 2, figsize=(14, 6))
    fig.suptitle(f'{basename} - Edge Effect Check', fontsize=12)

    for row, (data, label) in enumerate([(lfp_filtered, 'Filtered'), (lfp_cleaned, 'Cleaned')]):
        # 先頭
        axes[row, 0].plot(lfp_times[:n_edge], data[:n_edge, :3], lw=0.5)
        axes[row, 0].set_title(f'{label} - Start')
        axes[row, 0].set_xlabel('Time (s)')
        # 末尾
        axes[row, 1].plot(lfp_times[-n_edge:], data[-n_edge:, :3], lw=0.5)
        axes[row, 1].set_title(f'{label} - End')
        axes[row, 1].set_xlabel('Time (s)')

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(output_dir, f'{basename}_edge_check.png'),
                    dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
