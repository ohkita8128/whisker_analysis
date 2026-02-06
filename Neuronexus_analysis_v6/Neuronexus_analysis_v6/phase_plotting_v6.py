"""
phase_plotting_v6.py - 位相ロック解析の可視化 (v6)

グランドサマリー機能を追加:
1. 全ユニット×全帯域のMRL/PPC/有意性を1枚に
2. チャンネル別 preferred phase の極座標プロット
3. 条件別比較のコンパクトまとめ
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Optional, Any


# ============================================================
# 1. グランドサマリー (全ユニット一覧)
# ============================================================
def plot_phase_grand_summary(
    all_phase_results: Dict[str, Dict],
    all_condition_results: Dict[str, Dict],
    bands: Dict[str, tuple],
    original_ch_numbers: list,
    spike_data: Dict,
    output_dir: str,
    basename: str,
    show: bool = True,
    save: bool = True
):
    """
    位相ロック解析のグランドサマリー

    上段: ユニット × 帯域のMRLヒートマップ (参照LFPチャンネル=0)
    中段: 代表ユニットの位相分布 (極座標ヒストグラム)
    下段: 条件別MRL比較 (棒グラフ)
    """
    unit_keys = list(all_phase_results.keys())
    band_names = list(bands.keys())
    n_units = len(unit_keys)
    n_bands = len(band_names)

    if n_units == 0:
        print("  グランドサマリー: ユニットなし")
        return

    # レイアウト計算
    n_polar_cols = min(n_units, 4)
    n_polar_rows = 1

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, max(n_polar_cols, 2), hspace=0.45, wspace=0.4)
    fig.suptitle(f'{basename} - Phase Locking Grand Summary', fontsize=13, y=0.98)

    # --- 上段: MRLヒートマップ (ユニット × 帯域, ch=0のみ) ---
    ax_heat = fig.add_subplot(gs[0, :])
    mrl_matrix = np.zeros((n_units, n_bands))
    sig_matrix = np.ones((n_units, n_bands))  # p-values

    for i, uk in enumerate(unit_keys):
        for j, band in enumerate(band_names):
            if band in all_phase_results[uk]:
                # チャンネル0の結果を使用
                r = all_phase_results[uk][band].get(0, None)
                if r is not None:
                    mrl_matrix[i, j] = r.mrl
                    sig_matrix[i, j] = r.p_value

    im = ax_heat.imshow(mrl_matrix, aspect='auto', cmap='YlOrRd',
                         vmin=0, vmax=max(0.3, mrl_matrix.max()))
    ax_heat.set_xticks(range(n_bands))
    ax_heat.set_xticklabels(band_names)
    ax_heat.set_yticks(range(n_units))
    ax_heat.set_yticklabels(unit_keys, fontsize=8)
    ax_heat.set_xlabel('Frequency Band')
    ax_heat.set_ylabel('Unit')
    ax_heat.set_title('MRL (ch=0) — * p<0.05, ** p<0.01')
    plt.colorbar(im, ax=ax_heat, label='MRL', shrink=0.6)

    for i in range(n_units):
        for j in range(n_bands):
            if sig_matrix[i, j] < 0.01:
                ax_heat.text(j, i, '**', ha='center', va='center',
                             fontsize=10, color='white', fontweight='bold')
            elif sig_matrix[i, j] < 0.05:
                ax_heat.text(j, i, '*', ha='center', va='center',
                             fontsize=12, color='white', fontweight='bold')

    # --- 中段: 極座標ヒストグラム (代表ユニット, 最初の帯域) ---
    first_band = band_names[0] if band_names else 'theta'
    for i, uk in enumerate(unit_keys[:n_polar_cols]):
        ax_polar = fig.add_subplot(gs[1, i], projection='polar')

        if first_band in all_phase_results[uk]:
            r = all_phase_results[uk][first_band].get(0, None)
            if r is not None:
                bins = np.linspace(-np.pi, np.pi, 37)
                counts, _ = np.histogram(r.spike_phases, bins=bins)
                bin_width = 2 * np.pi / 36
                counts_norm = counts / (len(r.spike_phases) * bin_width)
                bin_centers = (bins[:-1] + bins[1:]) / 2

                ax_polar.bar(bin_centers, counts_norm, width=bin_width * 0.9,
                             color='steelblue', alpha=0.7, edgecolor='white', lw=0.3)

                max_c = counts_norm.max() if len(counts_norm) > 0 else 1
                ax_polar.annotate('', xy=(r.preferred_phase, r.mrl * max_c * 1.2),
                                  xytext=(0, 0),
                                  arrowprops=dict(arrowstyle='->', color='red', lw=2))

                sig_str = '*' if r.significant else ''
                ax_polar.set_title(f'{uk}\nMRL={r.mrl:.3f}{sig_str}',
                                   fontsize=8, pad=12)
            else:
                ax_polar.set_title(f'{uk}\n(no data)', fontsize=8)
        else:
            ax_polar.set_title(f'{uk}', fontsize=8)

    # --- 下段: 条件別MRL比較 ---
    if all_condition_results:
        ax_cond = fig.add_subplot(gs[2, :])

        conditions = ['baseline', 'stim', 'post']
        cond_colors = {'baseline': '#888888', 'stim': '#e74c3c', 'post': '#3498db'}
        x = np.arange(n_units)
        width = 0.25

        for ci, cond in enumerate(conditions):
            vals = []
            for uk in unit_keys:
                v = 0.0
                if uk in all_condition_results:
                    # 最初の帯域の結果を使用
                    for band_name, cond_results in all_condition_results[uk].items():
                        if isinstance(cond_results, dict) and cond in cond_results:
                            r = cond_results[cond]
                            if r is not None:
                                v = r.mrl
                                break
                vals.append(v)

            ax_cond.bar(x + ci * width - width, vals, width,
                        color=cond_colors[cond], alpha=0.7, label=cond)

        ax_cond.set_xticks(x)
        ax_cond.set_xticklabels(unit_keys, rotation=30, fontsize=8)
        ax_cond.set_ylabel('MRL')
        ax_cond.set_title('Condition Comparison (MRL)')
        ax_cond.legend(fontsize=8)
    else:
        ax_blank = fig.add_subplot(gs[2, :])
        ax_blank.text(0.5, 0.5, '条件別解析なし', ha='center', va='center')
        ax_blank.axis('off')

    if save:
        fp = os.path.join(output_dir, f'{basename}_phase_grand_summary.png')
        plt.savefig(fp, dpi=150, bbox_inches='tight')
        print(f"  保存: {fp}")

    if show:
        plt.show()
    else:
        plt.close()


# ============================================================
# 2. チャンネル別 MRL ヒートマップ (1ユニット)
# ============================================================
def plot_unit_phase_heatmap(
    results: Dict[str, Dict],
    band_names: list,
    channel_labels: list,
    unit_key: str,
    output_dir: str,
    basename: str,
    show: bool = True,
    save: bool = True
):
    """1ユニットの位相ロック: 帯域 × チャンネル ヒートマップ"""
    n_bands = len(band_names)
    n_ch = len(channel_labels)

    mrl_mat = np.zeros((n_bands, n_ch))
    for i, band in enumerate(band_names):
        if band in results:
            for j in range(n_ch):
                r = results[band].get(j, None)
                if r is not None:
                    mrl_mat[i, j] = r.mrl

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(mrl_mat, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.5)
    ax.set_xticks(range(n_ch))
    ax.set_xticklabels(channel_labels, rotation=45, fontsize=7)
    ax.set_yticks(range(n_bands))
    ax.set_yticklabels(band_names)
    ax.set_title(f'{unit_key} - Phase Locking (MRL)')
    plt.colorbar(im, ax=ax, label='MRL')

    plt.tight_layout()
    if save:
        fp = os.path.join(output_dir, f'{basename}_{unit_key}_phase_heatmap.png')
        plt.savefig(fp, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


# ============================================================
# 3. 条件別位相分布比較 (1ユニット, 1帯域)
# ============================================================
def plot_condition_polar(
    condition_results: Dict[str, Any],
    unit_key: str,
    band_name: str,
    output_dir: str,
    basename: str,
    show: bool = True,
    save: bool = True
):
    """条件別の位相分布を横並び極座標で表示"""
    conditions = ['baseline', 'stim', 'post']
    colors = {'baseline': 'gray', 'stim': 'red', 'post': 'blue'}

    fig = plt.figure(figsize=(14, 4))

    for i, cond in enumerate(conditions):
        ax = fig.add_subplot(1, 4, i + 1, projection='polar')
        r = condition_results.get(cond, None)
        if r is not None:
            bins = np.linspace(-np.pi, np.pi, 37)
            counts, _ = np.histogram(r.spike_phases, bins=bins)
            bw = 2 * np.pi / 36
            counts_norm = counts / (len(r.spike_phases) * bw)
            bc = (bins[:-1] + bins[1:]) / 2
            ax.bar(bc, counts_norm, width=bw * 0.9, color=colors[cond],
                   alpha=0.7, edgecolor='white', lw=0.3)

            mc = counts_norm.max() if len(counts_norm) > 0 else 1
            ax.annotate('', xy=(r.preferred_phase, r.mrl * mc * 1.2),
                        xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2))
            sig = '*' if r.significant else ''
            ax.set_title(f'{cond}\nMRL={r.mrl:.3f}{sig}\nn={r.n_spikes}', fontsize=9)
        else:
            ax.set_title(f'{cond}\n(insufficient)', fontsize=9)

    # 右端: MRL比較棒グラフ
    ax_bar = fig.add_subplot(1, 4, 4)
    mrl_vals = []
    for cond in conditions:
        r = condition_results.get(cond, None)
        mrl_vals.append(r.mrl if r is not None else 0)
    ax_bar.bar(conditions, mrl_vals, color=[colors[c] for c in conditions], alpha=0.7)
    ax_bar.set_ylabel('MRL')
    ax_bar.set_title('MRL Comparison')

    fig.suptitle(f'{unit_key} - {band_name}', fontsize=11)
    plt.tight_layout()

    if save:
        fp = os.path.join(output_dir, f'{basename}_{unit_key}_{band_name}_condition.png')
        plt.savefig(fp, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
