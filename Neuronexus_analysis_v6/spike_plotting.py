"""
spike_plotting.py - スパイク関連プロットの統合モジュール

枚数を抑える方針:
1. グランドサマリー: 全チャンネル×全ユニットを1枚に概観
2. チャンネル詳細: 波形+PCA+ISI+ACGを1チャンネル1枚 (必要時のみ)
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List
from spike_sorting import (ChannelSortResult, SpikeUnit,
                            compute_isi_histogram, compute_autocorrelogram)


# ============================================================
# 1. グランドサマリー (全チャンネル一覧を1枚)
# ============================================================
def plot_spike_grand_summary(results: Dict[int, ChannelSortResult],
                              output_dir: str, basename: str,
                              max_ch_display: int = 16,
                              show: bool = True, save: bool = True):
    """
    全チャンネルのスパイクソーティング結果を1枚にまとめる

    各チャンネル: 平均波形 (1セル) + PCA散布図 (1セル) = 2列 × n_ch行
    """
    channels = sorted(results.keys())[:max_ch_display]
    n_ch = len(channels)
    if n_ch == 0:
        return

    fig, axes = plt.subplots(n_ch, 2, figsize=(12, max(4, n_ch * 1.5)))
    fig.suptitle(f'{basename} - Spike Sorting Grand Summary', fontsize=13, y=1.0)

    if n_ch == 1:
        axes = axes.reshape(1, -1)

    for row, ch in enumerate(channels):
        result = results[ch]
        ax_wf = axes[row, 0]
        ax_pca = axes[row, 1]

        valid_units = [u for u in result.units if not u.is_noise]

        if not valid_units:
            ax_wf.text(0.5, 0.5, 'No units', ha='center', va='center', fontsize=9)
            ax_wf.set_ylabel(f'Ch{ch}', fontsize=9)
            ax_pca.text(0.5, 0.5, '-', ha='center', va='center', fontsize=9)
            ax_wf.set_xticks([])
            ax_wf.set_yticks([])
            ax_pca.set_xticks([])
            ax_pca.set_yticks([])
            continue

        time_ms = result.waveform_time_ms if result.waveform_time_ms is not None \
            else np.linspace(-0.5, 1.0, 60)

        for unit in valid_units:
            c = unit.color
            a = 0.5 if unit.is_mua else 0.8

            # 波形
            if len(unit.waveforms) > 0:
                mean_wf = np.mean(unit.waveforms, axis=0)
                std_wf = np.std(unit.waveforms, axis=0)
                ax_wf.plot(time_ms, mean_wf, color=c, lw=1.5, alpha=a)
                ax_wf.fill_between(time_ms, mean_wf - std_wf, mean_wf + std_wf,
                                   color=c, alpha=0.15)

            # PCA
            if len(unit.pca_features) > 0:
                ax_pca.scatter(unit.pca_features[:, 0], unit.pca_features[:, 1],
                               c=c, s=3, alpha=0.3)

        # 装飾
        ax_wf.axhline(0, color='gray', ls='--', lw=0.5, alpha=0.4)
        ax_wf.axvline(0, color='gray', ls='--', lw=0.5, alpha=0.4)
        info_parts = [f'U{u.unit_id}:{u.n_spikes}' for u in valid_units]
        ax_wf.set_ylabel(f'Ch{ch}', fontsize=9)
        ax_wf.set_title(', '.join(info_parts), fontsize=7, pad=1)
        ax_wf.tick_params(labelsize=6)
        ax_pca.tick_params(labelsize=6)

        if row < n_ch - 1:
            ax_wf.set_xticklabels([])
            ax_pca.set_xticklabels([])

    axes[-1, 0].set_xlabel('Time (ms)', fontsize=8)
    axes[-1, 1].set_xlabel('PC1', fontsize=8)
    axes[0, 0].set_title('Mean Waveform ± SD', fontsize=9)
    axes[0, 1].set_title('PCA (PC1 vs PC2)', fontsize=9)

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(output_dir, f'{basename}_spike_grand_summary.png'),
                    dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


# ============================================================
# 2. チャンネル詳細 (1チャンネル1枚, 波形+PCA+ISI+ACG)
# ============================================================
def plot_channel_detail(result: ChannelSortResult,
                        output_dir: str, basename: str,
                        show: bool = True, save: bool = True):
    """
    1チャンネルの詳細プロット (2×2: 波形, PCA, ISI, ACG)
    """
    ch = result.channel
    valid_units = [u for u in result.units if not u.is_noise]
    if not valid_units:
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f'{basename} - Ch{ch} Detail (σ={result.sigma:.4f})', fontsize=11)

    time_ms = result.waveform_time_ms if result.waveform_time_ms is not None \
        else np.linspace(-0.5, 1.0, 60)

    for unit in valid_units:
        c = unit.color
        label = f'U{unit.unit_id} (n={unit.n_spikes})'

        # 波形
        if len(unit.waveforms) > 0:
            mean_wf = np.mean(unit.waveforms, axis=0)
            std_wf = np.std(unit.waveforms, axis=0)
            axes[0, 0].plot(time_ms, mean_wf, color=c, lw=2, label=label)
            axes[0, 0].fill_between(time_ms, mean_wf - std_wf, mean_wf + std_wf,
                                    color=c, alpha=0.2)

        # PCA
        if len(unit.pca_features) > 0:
            axes[0, 1].scatter(unit.pca_features[:, 0], unit.pca_features[:, 1],
                               c=c, s=8, alpha=0.4, label=label)

        # ISI
        if len(unit.spike_times) > 1:
            bins, hist = compute_isi_histogram(unit.spike_times)
            if len(bins) > 0:
                axes[1, 0].bar(bins, hist, width=1.0, color=c, alpha=0.6, label=label)

        # Autocorrelogram
        if len(unit.spike_times) > 1:
            bins_ac, acg = compute_autocorrelogram(unit.spike_times)
            if len(bins_ac) > 0:
                axes[1, 1].bar(bins_ac, acg, width=1.0, color=c, alpha=0.6, label=label)

    axes[0, 0].axhline(0, color='gray', ls='--', alpha=0.4)
    axes[0, 0].axvline(0, color='gray', ls='--', alpha=0.4)
    axes[0, 0].set_xlabel('Time (ms)')
    axes[0, 0].set_title('Mean Waveform')
    axes[0, 0].legend(fontsize=7)

    axes[0, 1].set_xlabel('PC1')
    axes[0, 1].set_ylabel('PC2')
    axes[0, 1].set_title('PCA')
    axes[0, 1].legend(fontsize=7)

    axes[1, 0].axvline(2, color='red', ls='--', alpha=0.5, label='2ms')
    axes[1, 0].set_xlabel('ISI (ms)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('ISI Histogram')
    axes[1, 0].set_xlim(0, 100)
    axes[1, 0].legend(fontsize=7)

    axes[1, 1].set_xlabel('Lag (ms)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Autocorrelogram')
    axes[1, 1].legend(fontsize=7)

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(output_dir, f'{basename}_ch{ch}_spike_detail.png'),
                    dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


# ============================================================
# 3. 品質サマリーテーブルプロット
# ============================================================
def plot_quality_table(results: Dict[int, ChannelSortResult],
                       output_dir: str, basename: str,
                       show: bool = True, save: bool = True):
    """
    全ユニットの品質指標をテーブルとして表示
    """
    rows = []
    for ch in sorted(results.keys()):
        for u in results[ch].units:
            if u.is_noise:
                continue
            status = 'MUA' if u.is_mua else ('Good' if u.isi_violation_rate < 2 else
                     'Fair' if u.isi_violation_rate < 5 else 'Poor')
            rows.append([
                f'Ch{ch}', f'U{u.unit_id}', u.n_spikes,
                f'{u.mean_amplitude:.4f}', f'{u.snr:.1f}',
                f'{u.isi_violation_rate:.1f}%', status
            ])

    if not rows:
        return

    fig, ax = plt.subplots(figsize=(10, max(3, len(rows) * 0.4)))
    ax.axis('off')
    ax.set_title(f'{basename} - Spike Unit Quality Summary', fontsize=12, pad=15)

    headers = ['Channel', 'Unit', 'N Spikes', 'Amplitude', 'SNR', 'ISI Viol', 'Status']
    table = ax.table(cellText=rows, colLabels=headers,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.3)

    # 色分け
    for i, row in enumerate(rows):
        status = row[-1]
        color = ('#d4edda' if status == 'Good' else
                 '#fff3cd' if status in ('Fair', 'MUA') else '#f8d7da')
        for j in range(len(headers)):
            table[i + 1, j].set_facecolor(color)

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(output_dir, f'{basename}_spike_quality.png'),
                    dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


# ============================================================
# 4. ユーティリティ: 全チャンネル詳細を選択的に出力
# ============================================================
def plot_selected_channel_details(results: Dict[int, ChannelSortResult],
                                  output_dir: str, basename: str,
                                  channels: List[int] = None,
                                  show: bool = True, save: bool = True):
    """指定チャンネルの詳細プロットを出力（指定なし＝ユニットを持つ全ch）"""
    if channels is None:
        channels = [ch for ch, r in results.items()
                    if any(not u.is_noise for u in r.units)]

    for ch in channels:
        if ch in results:
            plot_channel_detail(results[ch], output_dir, basename, show=show, save=save)
