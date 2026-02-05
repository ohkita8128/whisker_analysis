"""
plotting.py - プロット関数
処理確認、パワー解析、ヒートマップなど
"""
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.family'] = 'MS Gothic'  # または 'Yu Gothic', 'Meiryo'

def plot_processing_overview(lfp_trimmed, lfp_filtered, lfp_cleaned, lfp_times,
                              motion_resampled, motion_threshold, noise_mask,
                              removed_ics, original_ch_numbers,
                              output_dir, basename, show=True, save=True):
    """処理概要プロット"""
    n_channels = lfp_cleaned.shape[1]
    
    plt.rcParams.update({
    'font.size': 15,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 12,
})
    fig, axes = plt.subplots(4, 1, figsize=(14,8), sharex=True)
    
    # モーション
    axes[0].plot(lfp_times, motion_resampled, 'b-', lw=0.5)
    axes[0].axhline(motion_threshold, color='r', ls='--')
    axes[0].fill_between(lfp_times, 0, motion_resampled.max(), where=noise_mask, alpha=0.3, color='purple')
    axes[0].set_ylabel('Motion')
    axes[0].set_title('Motion Detection')
   
    # Raw LFP
    for idx in range(n_channels):
        axes[1].plot(lfp_times, lfp_trimmed[:, idx] - idx*0.3, lw=0.5)
    axes[1].set_ylabel('LFP')
    axes[1].set_title('Raw LFP')
    
    # Filtered LFP
    for idx in range(n_channels):
        axes[2].plot(lfp_times, lfp_filtered[:, idx] - idx*0.3, lw=0.5)
    axes[2].set_ylabel('LFP')
    axes[2].set_title('Filtered LFP')
    
    # ICA Cleaned LFP
    for idx in range(n_channels):
        axes[3].plot(lfp_times, lfp_cleaned[:, idx] - idx*0.3, lw=0.5)
    axes[3].fill_between(lfp_times, -5, 0.5, where=noise_mask, alpha=0.3, color='purple')
    axes[3].set_ylabel('LFP')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_title(f'After ICA (removed noise artifacts)')
    
    for ax in axes:
        ax.set_yticks([])
    
    plt.tight_layout()
    if save: plt.savefig(os.path.join(output_dir, f'{basename}_processing_overview.png'), dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


def plot_edge_check(lfp_filtered, lfp_cleaned, lfp_times, fs, 
                    output_dir, basename, show=True, save=True):
    """端部効果確認"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    t_check = 2.0
    
    for i, (ax, title) in enumerate(zip(axes[0], ['最初 2秒', '最後 2秒'])):
        if i == 0:
            t_mask = lfp_times < lfp_times[0] + t_check
        else:
            t_mask = lfp_times > lfp_times[-1] - t_check
        ax.plot(lfp_times[t_mask], lfp_filtered[t_mask, 0], 'b-', lw=1, label='Filtered')
        ax.plot(lfp_times[t_mask], lfp_cleaned[t_mask, 0], 'r-', lw=1, alpha=0.7, label='ICA')
        ax.set_title(title)
        ax.legend()
    
    axes[1, 0].plot(lfp_times, lfp_filtered[:, 0], 'b-', lw=0.3)
    axes[1, 0].set_title('Filtered - Full')
    axes[1, 1].plot(lfp_times, lfp_cleaned[:, 0], 'r-', lw=0.3)
    axes[1, 1].set_title('ICA Cleaned - Full')
    
    plt.tight_layout()
    if save: plt.savefig(os.path.join(output_dir, f'{basename}_edge_check.png'), dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


def plot_ica_components(ica_sources, noise_ratios, noise_mask, lfp_times, removed_ics,
                        output_dir, basename, t_start=None, t_end=None, show=True, save=True):

    
    # ... 以下既存コード
    if t_start is None:
        t_start = lfp_times[0]
    if t_end is None:
        t_end = lfp_times[-1]
    t_mask = (lfp_times >= t_start) & (lfp_times <= t_end)

    """ICA成分の可視化"""
    n_components = ica_sources.shape[1]
    n_cols = 4
    n_rows = (n_components + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 2.5 * n_rows), sharex=True)
    axes = axes.flatten()
    
    for i in range(n_components):
        ax = axes[i]
        ax.plot(lfp_times[t_mask], ica_sources[t_mask, i], 'k-', lw=0.5)
        ax.fill_between(lfp_times[t_mask], ica_sources[t_mask, i].min(),
                        ica_sources[t_mask, i].max(), where=noise_mask[t_mask], alpha=0.3, color='purple')
        is_removed = i in removed_ics
        color = 'red' if is_removed else 'black'
        ax.set_title(f'IC{i} ({noise_ratios[i]:.2f})' + (' *' if is_removed else ''), color=color)
    
    for i in range(n_components, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    if save: plt.savefig(os.path.join(output_dir, f'{basename}_ica_components.png'), dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


# plotting.pyのplot_power_analysis関数を置き換え

def plot_power_analysis(freqs, psd_baseline, psd_stim, psd_post,
                        bands, baseline_power, stim_power, post_power,
                        change_stim_list, change_post_list,
                        output_dir, basename, power_freq_max, n_peaks=3, show=True, save=True):
    """
    パワースペクトル解析プロット（上段のみ帯域背景色付き）
    
    Parameters
    ----------
    bands : dict or list
        帯域定義 {'delta': (1, 4), ...} または ['delta', 'theta', ...]
    """
    from scipy.signal import find_peaks
    
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

    
    # 帯域の色定義
    BAND_COLORS = {
        'delta': '#3b82f6',      # 青
        'theta': '#22c55e',      # 緑
        'alpha': '#f59e0b',      # オレンジ
        'beta': '#a855f7',       # 紫
        'gamma': '#ef4444',      # 赤
        'low_gamma': '#f97316',
        'high_gamma': '#dc2626',
        'low': '#6366f1',
        'high': '#ec4899',
    }
    DEFAULT_COLORS = ['#06b6d4', '#84cc16', '#f43f5e', '#8b5cf6', '#14b8a6', '#eab308']
    
    fig, axes = plt.subplots(2, 2, figsize=(8,6))
    plt.rcParams.update({
    'font.size': 15,
    'axes.titlesize': 18,
    'axes.labelsize': 15,
    'xtick.labelsize': 14,
    'ytick.labelsize': 15,
    'legend.fontsize': 13,
})
    
    # ===================
    # 左上: PSD（帯域背景付き）
    # ===================
    ax = axes[0, 0]
    
    for i, (band_name, (f_low, f_high)) in enumerate(bands.items()):
        color = BAND_COLORS.get(band_name, DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
        ax.axvspan(f_low, f_high, alpha=0.15, color=color)
    
    ax.semilogy(freqs, psd_baseline, 'b-', lw=2, label='Baseline')
    ax.semilogy(freqs, psd_stim, 'r-', lw=2, label='Stim')
    ax.semilogy(freqs, psd_post, 'g-', lw=2, label='Post')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power')
    ax.set_title('Power Spectrum')
    ax.set_xlim(0, power_freq_max)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # ===================
    # 右上: 変化率 + ピーク + 帯域背景
    # ===================
    ax = axes[0, 1]
    
    change_stim_psd = (psd_stim - psd_baseline) / np.maximum(psd_baseline, 1e-10) * 100
    change_post_psd = (psd_post - psd_baseline) / np.maximum(psd_baseline, 1e-10) * 100
    
    freq_mask = freqs <= power_freq_max
    freqs_plot = freqs[freq_mask]
    change_stim_plot = change_stim_psd[freq_mask]
    change_post_plot = change_post_psd[freq_mask]
    
    # 帯域の背景色
    for i, (band_name, (f_low, f_high)) in enumerate(bands.items()):
        color = BAND_COLORS.get(band_name, DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
        ax.axvspan(f_low, f_high, alpha=0.15, color=color)
        f_center = (f_low + f_high) / 2
        if f_center <= power_freq_max:
            ax.text(f_center, 280, band_name, ha='center', va='top', 
                    fontsize=8, fontweight='bold', color=color, alpha=0.8)
    
    ax.plot(freqs_plot, change_stim_plot, 'r-', lw=2, label='Stim')
    ax.plot(freqs_plot, change_post_plot, 'g-', lw=2, label='Post')
    ax.axhline(0, color='k', ls='--')
    print("ピーク検出はコメントアウトされています。")
    
    # # ピーク検出
    # peaks, _ = find_peaks(change_stim_plot, prominence=5)
    # if len(peaks) > 0:
    #     peak_values = change_stim_plot[peaks]
    #     top_indices = np.argsort(peak_values)[::-1][:n_peaks]
    #     top_peaks = peaks[top_indices]
        
    #     for i, peak_idx in enumerate(top_peaks):
    #         freq_val = freqs_plot[peak_idx]
    #         change_val = change_stim_plot[peak_idx]
            
    #         # どの帯域に属するか
    #         peak_band = None
    #         peak_color = 'darkred'
    #         for band_name, (f_low, f_high) in bands.items():
    #             if f_low <= freq_val <= f_high:
    #                 peak_band = band_name
    #                 peak_color = BAND_COLORS.get(band_name, 'darkred')
    #                 break
            
    #         ax.plot(freq_val, change_val, 'o', markersize=12, 
    #                 color=peak_color, markeredgecolor='black', markeredgewidth=1.5)
            
    #         y_offset = 15 + i * 25
    #         label_text = f'{freq_val:.1f}Hz ({change_val:+.0f}%)'
    #         if peak_band:
    #             label_text = f'{peak_band}\n{freq_val:.1f}Hz ({change_val:+.0f}%)'
            
    #         # ax.annotate(label_text,
    #         #             xy=(freq_val, change_val),
    #         #             xytext=(0, y_offset), textcoords='offset points',
            #             fontsize=9, color=peak_color, fontweight='bold',
            #             ha='center',
            #             arrowprops=dict(arrowstyle='->', color=peak_color, lw=1),
            #             bbox=dict(boxstyle='round,pad=0.3', 
            #                       facecolor='white', 
            #                       edgecolor=peak_color, alpha=0.9))
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Change (%)')
    ax.set_title('Power Change (top peaks labeled)')
    ax.set_xlim(0, power_freq_max)
    ax.set_ylim(-100, 300)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # ===================
    # 左下: バンドパワー棒グラフ（そのまま）
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
    ax.set_xticklabels(band_names)
    ax.legend()
    ax.grid(True, axis='y')
    
    # ===================
    # 右下: バンドパワー変化率（そのまま）
    # ===================
    ax = axes[1, 1]
    
    ax.bar(x - width/2, change_stim_list, width, label='Stim', color='red', alpha=0.7)
    ax.bar(x + width/2, change_post_list, width, label='Post', color='green', alpha=0.7)
    ax.axhline(0, color='k', lw=1)
    ax.set_ylabel('Change (%)')
    ax.set_title('Band Power Change')
    ax.set_xticks(x)
    ax.set_xticklabels(band_names)
    ax.legend()
    ax.grid(True, axis='y')
    
    plt.tight_layout()
    if save: 
        plt.savefig(os.path.join(output_dir, f'{basename}_power_analysis.png'), dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


import matplotlib.pyplot as plt
import numpy as np
import os

def plot_power_analysis_monocro(freqs, psd_baseline, psd_stim, psd_post,
                        bands, baseline_power, stim_power, post_power,
                        change_stim_list, change_post_list,
                        output_dir, basename, power_freq_max, n_peaks=3, show=True, save=True):
    """
    パワースペクトル解析プロット（白黒印刷・小サイズ対応版）
    """
    from scipy.signal import find_peaks
    
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

    # 白黒対応: 明度で区別
    STYLES = {
        'baseline': {'color': 'black'},
        'stim':     {'color': '#555555'},
        'post':     {'color': '#999999'},
    }
    
    BAR_STYLES = {
        'baseline': {'color': 'black'},
        'stim':     {'color': '#666666'},
        'post':     {'color': 'white'},
    }
    
    # ========================================
    # フォント設定（小サイズ用に最適化）
    # ========================================
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 19,
        'axes.labelsize': 17,
        'xtick.labelsize': 14,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'axes.linewidth': 0.6,
        'lines.linewidth': 1.2,
        'axes.titlepad': 4,      # タイトルと軸の間隔
        'axes.labelpad': 2,      # ラベルと軸の間隔
    })
    
    # ========================================
    # Legend共通設定
    # ========================================
    LEGEND_OPTS = {
        'framealpha': 0.9,
        'edgecolor': 'none',       # 枠なしでスッキリ
        'handlelength': 1.0,       # 線の長さ
        'handletextpad': 0.4,      # 線とテキストの間隔
        'labelspacing': 0.3,       # 項目間の間隔
        'borderpad': 0.3,          # legend内の余白
        'borderaxespad': 0.3,      # legendと軸の間隔
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    
    # ===================
    # 左上: PSD
    # ===================
    ax = axes[0, 0]
    
    for key, psd, label in [('baseline', psd_baseline, 'Baseline'),
                             ('stim', psd_stim, 'Stim'),
                             ('post', psd_post, 'Post')]:
        s = STYLES[key]
        ax.semilogy(freqs, psd, '-', color=s['color'], lw=1.2, label=label)
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power')
    ax.set_title('Power Spectrum')
    ax.set_xlim(0, power_freq_max)
    ax.legend(loc='lower left', **LEGEND_OPTS)
    ax.grid(True, alpha=0.3, linewidth=0.4)
    
    # ===================
    # 右上: 変化率
    # ===================
    ax = axes[0, 1]
    
    change_stim_psd = (psd_stim - psd_baseline) / np.maximum(psd_baseline, 1e-10) * 100
    change_post_psd = (psd_post - psd_baseline) / np.maximum(psd_baseline, 1e-10) * 100
    
    freq_mask = freqs <= power_freq_max
    freqs_plot = freqs[freq_mask]
    change_stim_plot = change_stim_psd[freq_mask]
    change_post_plot = change_post_psd[freq_mask]
    
    for key, data, label in [('stim', change_stim_plot, 'Stim'),
                              ('post', change_post_plot, 'Post')]:
        s = STYLES[key]
        ax.plot(freqs_plot, data, '-', color=s['color'], lw=1.2, label=label)
    
    ax.axhline(0, color='black', ls='-', lw=0.6)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Change (%)')
    ax.set_title('Power Change')
    ax.set_xlim(0, power_freq_max)
    ax.set_ylim(-100, 300)
    ax.legend(loc='upper left', **LEGEND_OPTS)
    ax.grid(True, alpha=0.3, linewidth=0.4)
    
    # ===================
    # 左下: バンドパワー棒グラフ
    # ===================
    ax = axes[1, 0]
    x = np.arange(len(band_names))
    width = 0.25
    
    for i, (key, data, label) in enumerate([('baseline', baseline_power, 'Baseline'),
                                             ('stim', stim_power, 'Stim'),
                                             ('post', post_power, 'Post')]):
        b = BAR_STYLES[key]
        offset = (i - 1) * width
        ax.bar(x + offset, data, width, label=label,
               color=b['color'], edgecolor='black', linewidth=0.6)
    
    ax.set_ylabel('Power')
    ax.set_title('Band Power')
    ax.set_xticks(x)
    ax.set_xticklabels(band_names, fontsize=16)
    ax.legend(**LEGEND_OPTS)
    ax.grid(True, axis='y', alpha=0.3, linewidth=0.4)
    
    # ===================
    # 右下: バンドパワー変化率
    # ===================
    ax = axes[1, 1]
    
    for i, (key, data, label) in enumerate([('stim', change_stim_list, 'Stim'),
                                             ('post', change_post_list, 'Post')]):
        b = BAR_STYLES[key]
        offset = (i - 0.5) * width
        ax.bar(x + offset, data, width, label=label,
               color=b['color'], edgecolor='black', linewidth=0.6)
    
    ax.axhline(0, color='black', lw=0.6)
    ax.set_ylabel('Change (%)')
    ax.set_title('Band Power Change')
    ax.set_xticks(x)
    ax.set_xticklabels(band_names, fontsize=16)
    ax.legend(**LEGEND_OPTS)
    ax.grid(True, axis='y', alpha=0.3, linewidth=0.4)
    
    # ========================================
    # 余白を詰める
    # ========================================
    plt.tight_layout(pad=0.5, h_pad=0.8, w_pad=0.8)
    
    if save: 
        plt.savefig(os.path.join(output_dir, f'{basename}_power_analysis.png'), 
                    dpi=300, bbox_inches='tight', pad_inches=0.05, facecolor='white')
    if show:
        plt.show()
    else:
        plt.close()

def plot_channel_heatmap(change_stim_ch, change_post_ch, bands, original_ch_numbers,
                         output_dir, basename, show=True, save=True):
    """チャンネル×バンドのヒートマップ"""
    import seaborn as sns
    
    n_channels = len(original_ch_numbers)
    ch_labels = [f'D{i}(Ch{original_ch_numbers[i]})' for i in range(n_channels)]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    
    sns.heatmap(change_stim_ch, ax=axes[0], cmap='RdBu_r', center=0,
                xticklabels=bands, yticklabels=ch_labels,
                annot=True, fmt='.0f', cbar_kws={'label': '%'})
    axes[0].set_title('Stim vs Baseline')
    
    sns.heatmap(change_post_ch, ax=axes[1], cmap='RdBu_r', center=0,
                xticklabels=bands, yticklabels=ch_labels,
                annot=True, fmt='.0f', cbar_kws={'label': '%'})
    axes[1].set_title('Post vs Baseline')
    
    plt.tight_layout()
    if save: plt.savefig(os.path.join(output_dir, f'{basename}_channel_heatmap.png'), dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


def create_sync_video(video_file, lfp_data, lfp_times, motion_resampled,
                      noise_mask, frame_times, roi, motion_threshold,
                      output_dir, basename, t_start=None, t_end=None,
                      channel=0, fps_out=None):
    """
    LFPと動画の同期確認動画を生成
    
    Parameters
    ----------
    t_start : float, optional
        開始時刻。Noneならlfp_times[0]を使用
    t_end : float, optional
        終了時刻。Noneならlfp_times[-1]を使用
    """
    import cv2
    
    if t_start is None:
        t_start = lfp_times[0]
    if t_end is None:
        t_end = lfp_times[-1]

    
    cap = cv2.VideoCapture(video_file)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    #.plxから実際のfpsを計算
    if fps_out is None:
        # PLXの同期イベントから実際のフレームレートを計算
        actual_fps = len(frame_times) / (frame_times[-1] - frame_times[0])
        fps_out = actual_fps

    print(f"実際のfps（PLX基準）: {fps_out:.2f}")

    start_idx = np.searchsorted(frame_times, t_start)
    end_idx = np.searchsorted(frame_times, t_end)
    total_frames = end_idx - start_idx
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    
    fig_w, fig_h, dpi = 10, 5, 80
    plot_w, plot_h = int(fig_w * dpi), int(fig_h * dpi)
    total_w = frame_width + plot_w
    total_h = max(frame_height, plot_h)
    
    output_path = os.path.join(output_dir, f'{basename}_sync.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps_out, (total_w, total_h))
    
    x, y, w, h = roi
    lfp_window = 5.0
    
    plt.ioff()
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h), dpi=dpi)
    
    for i, frame_idx in enumerate(range(start_idx, end_idx)):
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = frame_times[frame_idx]
        frame_display = cv2.convertScaleAbs(frame, alpha=1.5, beta=30)
        
        lfp_idx = np.searchsorted(lfp_times, current_time)
        is_noise = noise_mask[lfp_idx] if lfp_idx < len(noise_mask) else False
        color = (0, 0, 255) if is_noise else (0, 255, 0)
        cv2.rectangle(frame_display, (x, y), (x+w, y+h), color, 3)
        
        axes[0].cla()
        axes[1].cla()
        t_mask = (lfp_times >= current_time - lfp_window) & (lfp_times <= current_time + lfp_window)
        if np.sum(t_mask) > 0:
            axes[0].plot(lfp_times[t_mask], motion_resampled[t_mask], 'b-')
            axes[0].axhline(motion_threshold, color='r', ls='--')
            axes[0].axvline(current_time, color='k')
            axes[0].set_xlim(current_time - lfp_window, current_time + lfp_window) 
            axes[1].plot(lfp_times[t_mask], lfp_data[t_mask, channel], 'k-', lw=0.5)
            axes[1].axvline(current_time, color='k')
            axes[1].set_xlim(current_time - lfp_window, current_time + lfp_window) 
        
        fig.canvas.draw()
        plot_img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)
        
        combined = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        y_off = (total_h - frame_height) // 2
        combined[y_off:y_off+frame_height, :frame_width] = frame_display
        y_off = (total_h - plot_h) // 2
        combined[y_off:y_off+plot_h, frame_width:] = cv2.resize(plot_img, (plot_w, plot_h))
        
        out.write(combined)
        if i % (total_frames // 10 + 1) == 0:
            print(f"    {100*i/total_frames:.0f}%")
    
    plt.close(fig)
    plt.ion()
    cap.release()
    out.release()


def plot_wavelet_single(cwt_power, cwt_freqs, lfp_times, stim_times,
                        original_ch_numbers, channel, output_dir, basename,
                        t_start=None, t_end=None, show=True, save=True):
    """単一チャンネルのウェーブレットスペクトログラム"""
    if t_start is None:
        t_start = lfp_times[0]
    if t_end is None:
        t_end = lfp_times[-1]
    t_mask = (lfp_times >= t_start) & (lfp_times <= t_end)

    idx_start = np.where(t_mask)[0][0]
    idx_end = np.where(t_mask)[0][-1]
    
    times_plot = lfp_times[t_mask]
    cwt_plot = cwt_power[:, channel, idx_start:idx_end+1]
    
    ch_label = f'D{channel}(Ch{original_ch_numbers[channel]})'
    
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(cwt_plot, aspect='auto', origin='lower', cmap='jet',
                   extent=[times_plot[0], times_plot[-1], cwt_freqs[0], cwt_freqs[-1]],
                   vmin=0, vmax=np.percentile(cwt_plot, 95))
    
    # 刺激タイミング
    stim_in_range = stim_times[(stim_times >= t_start) & (stim_times <= t_end)]
    for stim_t in stim_in_range:
        ax.axvline(stim_t, color='white', alpha=0.7, lw=1, ls='--')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(f'Wavelet Spectrogram ({ch_label})')
    plt.colorbar(im, ax=ax, label='Power')
    
    plt.tight_layout()
    if save: plt.savefig(os.path.join(output_dir, f'{basename}_wavelet_ch{channel}.png'), dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


def plot_wavelet_all(cwt_power, cwt_freqs, lfp_times, original_ch_numbers,
                     output_dir, basename, t_start=None, t_end=None, show=True, save=True):
    """全チャンネルのウェーブレットスペクトログラム（タイル表示）"""
    n_channels = cwt_power.shape[1]
    n_cols = 4
    n_rows = (n_channels + n_cols - 1) // n_cols
    
    if t_start is None:
        t_start = lfp_times[0]
    if t_end is None:
        t_end = lfp_times[-1]

    t_mask = (lfp_times >= t_start) & (lfp_times <= t_end)

    idx_start = np.where(t_mask)[0][0]
    idx_end = np.where(t_mask)[0][-1]
    times_plot = lfp_times[t_mask]
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
    axes = axes.flatten()
    
    for ch_idx in range(n_channels):
        ax = axes[ch_idx]
        cwt_plot = cwt_power[:, ch_idx, idx_start:idx_end+1]
        ch_label = f'D{ch_idx}(Ch{original_ch_numbers[ch_idx]})'
        
        im = ax.imshow(cwt_plot, aspect='auto', origin='lower', cmap='jet',
                       extent=[times_plot[0], times_plot[-1], cwt_freqs[0], cwt_freqs[-1]],
                       vmin=0, vmax=np.percentile(cwt_plot, 95))
        ax.set_title(ch_label)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Freq (Hz)')
    
    for idx in range(n_channels, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Wavelet Spectrogram - All Channels', fontsize=12)
    plt.tight_layout()
    if save: plt.savefig(os.path.join(output_dir, f'{basename}_wavelet_all.png'), dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


def plot_all_channels_with_regions(lfp_data, lfp_times, 
                                    stim_mask, baseline_mask, post_mask, noise_mask,
                                    original_ch_numbers, output_dir, basename,
                                    t_start=None, t_end=None, show=True, save=True):
    """
    全チャンネルLFPをマスク領域とともにプロット
    
    Parameters
    ----------
    lfp_data : ndarray (n_samples, n_channels)
    lfp_times : ndarray
    stim_mask, baseline_mask, post_mask, noise_mask : ndarray (bool)
    original_ch_numbers : list
    output_dir : str
    basename : str
    t_start : float, optional
        表示開始時刻（None=最初から）
    t_end : float
        表示終了時刻（None=最後まで）
    show : bool
    """
    if t_start is None:
        t_start = lfp_times[0]
    if t_end is None:
        t_end = lfp_times[-1]
    
    t_mask = (lfp_times >= t_start) & (lfp_times <= t_end)
    
    times = lfp_times[t_mask]
    n_channels = lfp_data.shape[1]
    
    # マスクも時間範囲で切り出し
    stim_plot = stim_mask[t_mask]
    base_plot = baseline_mask[t_mask]
    post_plot = post_mask[t_mask]
    noise_plot = noise_mask[t_mask]
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # 背景色
    spacing = 0.3
    ymin = -n_channels * spacing - 0.5
    ymax = 0.5
    
    ax.fill_between(times, ymin, ymax, where=base_plot, color='blue', alpha=0.2, label='Baseline')
    ax.fill_between(times, ymin, ymax, where=stim_plot, color='red', alpha=0.3, label='Stim')
    ax.fill_between(times, ymin, ymax, where=post_plot, color='green', alpha=0.2, label='Post')
    ax.fill_between(times, ymin, ymax, where=noise_plot, color='gray', alpha=0.5, label='Noise')
    
    # LFP波形(全チャンネル)
    for idx in range(n_channels):
        ax.plot(times, lfp_data[t_mask, idx] - idx * spacing, 'k-', lw=0.5)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Channel')
    ax.set_ylim(ymin, ymax)
    ax.set_yticks([-idx * spacing for idx in range(n_channels)])
    ax.set_yticklabels([f'D{idx}(Ch{original_ch_numbers[idx]})' for idx in range(n_channels)])
    ax.set_title('All Channels LFP with Region Markers')
    ax.legend(loc='upper right', ncol=4)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    if save: plt.savefig(os.path.join(output_dir, f'{basename}_lfp_all_channels_regions.png'), dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


def plot_fft_comparison(lfp_raw, lfp_filtered, fs, output_dir, basename,
                        freq_max=300, show=True, save=True):
    """
    FFTでフィルタ効果を確認
    
    Parameters
    ----------
    lfp_raw : ndarray (n_samples, n_channels) - フィルタ前
    lfp_filtered : ndarray (n_samples, n_channels) - フィルタ後
    fs : int
    output_dir : str
    basename : str
    freq_max : float
        表示する最大周波数
    show : bool
    """
    from scipy.fft import fft, fftfreq
    
    N = len(lfp_raw)
    freqs_fft = fftfreq(N, 1/fs)[:N//2]
    
    # チャンネル平均
    fft_raw = np.abs(fft(lfp_raw.mean(axis=1)))[:N//2]
    fft_filt = np.abs(fft(lfp_filtered.mean(axis=1)))[:N//2]
    
    # 正規化
    fft_raw = fft_raw / fft_raw.max()
    fft_filt = fft_filt / fft_filt.max()
    
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
    if save: plt.savefig(os.path.join(output_dir, f'{basename}_fft_comparison.png'), dpi=150)
    if show:
        plt.show()
    else:
        plt.close()
