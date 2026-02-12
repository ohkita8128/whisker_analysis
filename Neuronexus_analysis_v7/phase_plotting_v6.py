"""
phase_plotting_v6.py - 位相ロック解析の可視化 (v6)

6ページ構成のグランドサマリー（16ch リニアプローブ、層構造解析対応）:
  Page 1: LFPパワー×深さ + CSD + スパイクラスタ+LFP
  Page 2: PPC 深さ×帯域ヒートマップ（チャンネル集約）+ preferred phase 深さプロファイル
  Page 3: 極座標マトリクス（LFP ch × 帯域、スパイクchごと）
  Page 4: 深さ×深さ マトリクス（帯域ごと、チャンネル集約）
  Page 5: ローカル vs ディスタル + 条件比較
  Page 6: PAC + STA 深さプロファイル
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Optional, Tuple
from matplotlib.backends.backend_pdf import PdfPages


def _ch_label(idx, original_ch_numbers):
    """チャンネルインデックス→ラベル"""
    if original_ch_numbers and idx < len(original_ch_numbers):
        return f'Ch{original_ch_numbers[idx]}'
    return f'Ch{idx}'


def _ch_labels(n_ch, original_ch_numbers):
    return [_ch_label(i, original_ch_numbers) for i in range(n_ch)]


def _aggregate_by_channel(phase_results, spike_data, n_lfp_ch, band_names):
    """
    phase_results (unit単位) をスパイクチャンネル単位に集約。

    Returns
    -------
    agg : dict  {spike_ch: {band: {lfp_ch: PhaseLockingResult or None}}}
        各スパイクchで最良のユニットの結果を採用（MRL最大）
    ch_spike_counts : dict  {spike_ch: int}
    """
    from phase_locking import aggregate_spikes_by_channel
    ch_spikes = aggregate_spikes_by_channel(spike_data, n_lfp_ch)

    # 各ユニットのスパイクchを取得
    unit_to_ch = {}
    for ui in spike_data['unit_info']:
        unit_to_ch[ui.unit_key] = ui.channel

    # チャンネル集約: 同一スパイクchのユニットからMRL最大を採用
    agg = {}
    for unit_key, band_results in phase_results.items():
        sp_ch = unit_to_ch.get(unit_key)
        if sp_ch is None:
            continue
        if sp_ch not in agg:
            agg[sp_ch] = {}
        for band in band_names:
            if band not in band_results:
                continue
            if band not in agg[sp_ch]:
                agg[sp_ch][band] = {}
            for lfp_ch, result in band_results[band].items():
                existing = agg[sp_ch][band].get(lfp_ch)
                if result is not None:
                    if existing is None or result.mrl > existing.mrl:
                        agg[sp_ch][band][lfp_ch] = result

    ch_spike_counts = {ch: len(times) for ch, times in ch_spikes.items()}
    return agg, ch_spike_counts


# ============================================================
# Page 1: 記録の文脈
# ============================================================

def _plot_page1(fig, lfp_cleaned, lfp_times, fs, stim_times,
                channel_spacing_um, original_ch_numbers, spike_data, bands):
    """LFPパワー×深さ + CSD + スパイクラスタ+LFP"""
    from phase_locking import compute_lfp_psd_matrix, compute_evoked_csd

    n_ch = lfp_cleaned.shape[1] if lfp_cleaned.ndim > 1 else 1
    ch_labs = _ch_labels(n_ch, original_ch_numbers)

    has_stim = stim_times is not None and len(stim_times) > 0

    if has_stim:
        gs = fig.add_gridspec(1, 3, wspace=0.35, width_ratios=[1, 1, 1.2])
    else:
        gs = fig.add_gridspec(1, 2, wspace=0.35, width_ratios=[1, 1.2])

    fig.suptitle('Page 1: Recording Context', fontsize=13, y=0.98)

    # --- LFP Power Spectrum × Depth ---
    ax_psd = fig.add_subplot(gs[0, 0])
    psd_matrix, freqs = compute_lfp_psd_matrix(lfp_cleaned, fs)
    # pcolormeshでlog周波数軸を正しく表示（imshowはlog軸非対応）
    freq_edges = np.zeros(len(freqs) + 1)
    freq_edges[1:-1] = (freqs[:-1] + freqs[1:]) / 2
    freq_edges[0] = max(freqs[0] - (freqs[1] - freqs[0]) / 2, 0.1)
    freq_edges[-1] = freqs[-1] + (freqs[-1] - freqs[-2]) / 2
    ch_edges = np.arange(n_ch + 1) - 0.5
    im = ax_psd.pcolormesh(freq_edges, ch_edges, psd_matrix.T,
                            cmap='inferno', shading='flat')
    ax_psd.set_xscale('log')
    ax_psd.set_xlabel('Frequency (Hz)', fontsize=8)
    ax_psd.set_ylabel('Channel (depth)', fontsize=8)
    ax_psd.set_yticks(range(n_ch))
    ax_psd.set_yticklabels(ch_labs, fontsize=6)
    ax_psd.invert_yaxis()
    ax_psd.set_title('LFP Power Spectrum', fontsize=9)
    fig.colorbar(im, ax=ax_psd, label='Power (dB)', shrink=0.7, pad=0.02)

    # 帯域範囲を半透明で表示
    band_colors = plt.cm.Set2(np.linspace(0, 1, len(bands)))
    for bi, (bname, (lo, hi)) in enumerate(bands.items()):
        ax_psd.axvspan(lo, hi, alpha=0.1, color=band_colors[bi])
        ax_psd.text(np.sqrt(lo * hi), -0.8, bname, fontsize=5,
                    ha='center', color=band_colors[bi])

    # --- CSD (イベント揃え、stim_timesがある場合) ---
    if has_stim:
        ax_csd = fig.add_subplot(gs[0, 1])
        try:
            csd_mean, csd_time = compute_evoked_csd(
                lfp_cleaned, lfp_times, stim_times, fs,
                channel_spacing_um, window_ms=(-50, 200))
            n_csd_ch = csd_mean.shape[1]
            vmax_csd = np.percentile(np.abs(csd_mean), 95) if np.any(csd_mean) else 1
            im2 = ax_csd.imshow(csd_mean.T, aspect='auto', cmap='RdBu_r',
                                 extent=[csd_time[0], csd_time[-1],
                                         n_csd_ch - 0.5, -0.5],
                                 interpolation='nearest',
                                 vmin=-vmax_csd, vmax=vmax_csd)
            ax_csd.axvline(0, color='black', ls='--', lw=0.5)
            ax_csd.set_xlabel('Time from stim (ms)', fontsize=8)
            ax_csd.set_ylabel('Channel (depth)', fontsize=8)
            csd_labels = ch_labs[1:-1] if n_csd_ch == n_ch - 2 else ch_labs[:n_csd_ch]
            ax_csd.set_yticks(range(n_csd_ch))
            ax_csd.set_yticklabels(csd_labels, fontsize=6)
            ax_csd.set_title('Evoked CSD (sink=blue)', fontsize=9)
            fig.colorbar(im2, ax=ax_csd, label='CSD', shrink=0.7, pad=0.02)
        except Exception as e:
            ax_csd.text(0.5, 0.5, f'CSD error:\n{e}', ha='center', va='center',
                        fontsize=7, transform=ax_csd.transAxes)
            ax_csd.set_title('Evoked CSD', fontsize=9)

    # --- スパイクラスタ + LFPトレース ---
    ax_raster = fig.add_subplot(gs[0, -1])
    # LFPトレースを深さ順に描画（最初の2秒）
    t_show = min(2.0, lfp_times[-1] - lfp_times[0])
    t_mask = (lfp_times >= lfp_times[0]) & (lfp_times <= lfp_times[0] + t_show)
    times_show = lfp_times[t_mask]

    lfp_show = (lfp_cleaned[t_mask, :] if lfp_cleaned.ndim > 1
                else lfp_cleaned[t_mask, np.newaxis])
    max_amp = np.percentile(np.abs(lfp_show), 95) if np.any(lfp_show) else 1
    offset_scale = max_amp * 3

    for ch_i in range(n_ch):
        offset = ch_i * offset_scale
        ax_raster.plot(times_show, lfp_show[:, ch_i] + offset,
                       'k-', lw=0.3, alpha=0.6)

    # スパイクラスタ表示
    if spike_data:
        from phase_locking import aggregate_spikes_by_channel
        ch_spikes = aggregate_spikes_by_channel(spike_data, n_ch)
        for ch_i, sp_times in ch_spikes.items():
            t_mask_sp = (sp_times >= times_show[0]) & (sp_times <= times_show[-1])
            sp_show = sp_times[t_mask_sp]
            if len(sp_show) > 0:
                y_pos = np.full_like(sp_show, ch_i * offset_scale)
                ax_raster.scatter(sp_show, y_pos, c='red', s=3, marker='|',
                                  linewidths=0.5, zorder=5)

    ax_raster.set_yticks([i * offset_scale for i in range(n_ch)])
    ax_raster.set_yticklabels(ch_labs, fontsize=5)
    ax_raster.invert_yaxis()
    ax_raster.set_xlabel('Time (s)', fontsize=8)
    ax_raster.set_title('LFP + Spike Raster', fontsize=9)
    ax_raster.tick_params(labelsize=6)


# ============================================================
# Page 2: PPC ヒートマップ + Preferred Phase Profile
# ============================================================

def _plot_page2(fig, agg, ch_spike_counts, bands, n_lfp_ch,
                original_ch_numbers, fdr_significant, phase_results, spike_data):
    """PPC ヒートマップ (1枚, 縦=LFPch, 横=帯域) + preferred phase 深さプロファイル"""
    band_names = list(bands.keys())
    n_bands = len(band_names)
    ch_labs = _ch_labels(n_lfp_ch, original_ch_numbers)

    gs = fig.add_gridspec(1, 1 + n_bands, wspace=0.3,
                          width_ratios=[1.5] + [1] * n_bands)
    fig.suptitle('Page 2: Phase Locking Strength (all units pooled)',
                 fontsize=13, y=0.98)

    # --- 全ユニット平均 PPC (LFP ch × band) ---
    ppc_mat = np.zeros((n_lfp_ch, n_bands))
    count_mat = np.zeros((n_lfp_ch, n_bands))
    fdr_any = np.zeros((n_lfp_ch, n_bands), dtype=bool)

    for unit_key, results in phase_results.items():
        for bi, band in enumerate(band_names):
            if band not in results:
                continue
            for ch in range(n_lfp_ch):
                r = results[band].get(ch)
                if r is not None:
                    ppc_mat[ch, bi] += r.ppc
                    count_mat[ch, bi] += 1
                if fdr_significant and fdr_significant.get(unit_key, {}).get(band, {}).get(ch, False):
                    fdr_any[ch, bi] = True

    with np.errstate(divide='ignore', invalid='ignore'):
        ppc_mean = np.where(count_mat > 0, ppc_mat / count_mat, 0)

    # ヒートマップ (縦=LFP ch, 横=帯域)
    ax_heat = fig.add_subplot(gs[0, 0])
    vmax = max(0.05, np.max(ppc_mean))
    im = ax_heat.imshow(ppc_mean, aspect='auto', cmap='YlGnBu',
                         vmin=0, vmax=vmax)
    ax_heat.set_yticks(range(n_lfp_ch))
    ax_heat.set_yticklabels(ch_labs, fontsize=6)
    ax_heat.set_xticks(range(n_bands))
    ax_heat.set_xticklabels(band_names, fontsize=8)
    ax_heat.set_ylabel('LFP Channel (depth)', fontsize=9)
    ax_heat.set_xlabel('Frequency Band', fontsize=9)
    n_total_spikes = sum(ch_spike_counts.values())
    ax_heat.set_title(f'PPC (all units avg, {n_total_spikes} spikes)', fontsize=9)
    fig.colorbar(im, ax=ax_heat, label='PPC', shrink=0.7)

    for ch in range(n_lfp_ch):
        for bi in range(n_bands):
            if fdr_any[ch, bi]:
                ax_heat.text(bi, ch, '*', ha='center', va='center',
                             fontsize=10, color='white', fontweight='bold')
            else:
                ax_heat.add_patch(plt.Rectangle(
                    (bi - 0.5, ch - 0.5), 1, 1,
                    fill=True, facecolor='gray', alpha=0.35, edgecolor='none'))

    # --- Preferred Phase 深さプロファイル（帯域ごと） ---
    for bi, band in enumerate(band_names):
        ax = fig.add_subplot(gs[0, 1 + bi])
        # 全ユニットの結果を集約
        phases_by_ch = {}
        for sp_ch in agg:
            if band not in agg[sp_ch]:
                continue
            for lfp_ch in range(n_lfp_ch):
                r = agg[sp_ch][band].get(lfp_ch)
                if r is not None and r.significant:
                    if lfp_ch not in phases_by_ch:
                        phases_by_ch[lfp_ch] = []
                    phases_by_ch[lfp_ch].append((r.preferred_phase, r.mrl))

        chs_plot = []
        phases_plot = []
        mrls_plot = []
        for lfp_ch in sorted(phases_by_ch.keys()):
            # 複数ユニットがある場合はMRL加重平均
            items = phases_by_ch[lfp_ch]
            total_mrl = sum(m for _, m in items)
            if total_mrl > 0:
                avg_phase = np.angle(sum(m * np.exp(1j * p) for p, m in items) / total_mrl)
            else:
                avg_phase = items[0][0]
            avg_mrl = np.mean([m for _, m in items])
            chs_plot.append(lfp_ch)
            phases_plot.append(avg_phase)
            mrls_plot.append(avg_mrl)

        if chs_plot:
            ax.scatter(np.degrees(phases_plot), chs_plot,
                       s=np.array(mrls_plot) * 300 + 15,
                       c=np.array(mrls_plot), cmap='YlOrRd',
                       vmin=0, vmax=max(0.3, max(mrls_plot)),
                       alpha=0.8, edgecolors='black', lw=0.5)
            ax.plot(np.degrees(phases_plot), chs_plot, '-', alpha=0.3,
                    lw=0.5, color='gray')

        ax.set_xlim(-180, 180)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_xlabel('Pref. Phase (deg)', fontsize=7)
        ax.set_yticks(range(n_lfp_ch))
        ax.set_yticklabels(ch_labs if bi == 0 else [], fontsize=5)
        ax.invert_yaxis()
        ax.set_title(f'{band}', fontsize=9)
        ax.tick_params(labelsize=6)
        if bi == 0:
            ax.set_ylabel('LFP Channel (depth)', fontsize=8)


# ============================================================
# Page 3: 極座標マトリクス
# ============================================================

def _plot_page3(fig, phase_results, bands, n_lfp_ch,
                original_ch_numbers, fdr_significant,
                ch_start=0, ch_end=None):
    """全スパイクchプール、LFP ch × 帯域 極座標マトリクス（ch範囲指定で分割対応）"""
    band_names = list(bands.keys())
    n_bands = len(band_names)
    ch_labs = _ch_labels(n_lfp_ch, original_ch_numbers)

    if ch_end is None:
        ch_end = n_lfp_ch
    n_rows = ch_end - ch_start

    gs = fig.add_gridspec(n_rows, n_bands, hspace=0.3, wspace=0.2)
    ch_range_str = f'{ch_labs[ch_start]}-{ch_labs[ch_end - 1]}'
    fig.suptitle(f'Page 3: Phase Distribution — {ch_range_str} '
                 f'(all spike ch pooled)',
                 fontsize=12, y=0.99)

    cmap_band = plt.cm.Set1(np.linspace(0, 1, max(n_bands, 1)))
    fdr_sig = fdr_significant or {}

    for row_i, lfp_ch in enumerate(range(ch_start, ch_end)):
        for bi, band in enumerate(band_names):
            ax = fig.add_subplot(gs[row_i, bi], projection='polar')

            # 全ユニットのspike_phasesをプール + FDR有意性チェック
            all_phases = []
            is_fdr_sig = False
            for unit_key, results in phase_results.items():
                if band in results:
                    r = results[band].get(lfp_ch)
                    if r is not None and len(r.spike_phases) > 0:
                        all_phases.append(r.spike_phases)
                if fdr_sig.get(unit_key, {}).get(band, {}).get(lfp_ch, False):
                    is_fdr_sig = True

            if all_phases:
                phases = np.concatenate(all_phases)
                bins = np.linspace(-np.pi, np.pi, 25)
                counts, _ = np.histogram(phases, bins=bins)
                bw = 2 * np.pi / 24
                counts_norm = counts / (len(phases) * bw)
                bc = (bins[:-1] + bins[1:]) / 2

                c = cmap_band[bi]
                alpha = 0.8 if is_fdr_sig else 0.15
                ax.bar(bc, counts_norm, width=bw * 0.9,
                       color=c, alpha=alpha, edgecolor='white', lw=0.2)

                mean_vec = np.mean(np.exp(1j * phases))
                mrl = np.abs(mean_vec)
                pref = np.angle(mean_vec)

                if is_fdr_sig:
                    mc = counts_norm.max() if counts_norm.max() > 0 else 1
                    ax.annotate('', xy=(pref, mrl * mc * 1.3),
                                xytext=(0, 0),
                                arrowprops=dict(arrowstyle='->', color='red',
                                                lw=1.2))

                sig_mark = '*' if is_fdr_sig else ''
                ax.text(0.95, 0.05, f'{mrl:.2f}{sig_mark}\nn={len(phases)}',
                        transform=ax.transAxes, fontsize=3.5,
                        ha='right', va='bottom')

            ax.tick_params(labelsize=3)
            ax.set_rticks([])

            if row_i == 0:
                ax.set_title(band, fontsize=7, pad=5)
            if bi == 0:
                ax.set_ylabel(ch_labs[lfp_ch], fontsize=5, labelpad=15)


# ============================================================
# Page 4: 深さ×深さマトリクス
# ============================================================

def _plot_page4(fig, agg, bands, n_lfp_ch, original_ch_numbers):
    """帯域ごとの spike_ch × lfp_ch PPC マトリクス"""
    band_names = list(bands.keys())
    n_bands = len(band_names)
    ch_labs = _ch_labels(n_lfp_ch, original_ch_numbers)

    n_cols = min(n_bands, 5)
    gs = fig.add_gridspec(1, n_cols + 1, wspace=0.3,
                          width_ratios=[1] * n_cols + [0.05])
    fig.suptitle('Page 4: Depth x Depth PPC Matrix (channel-aggregated)',
                 fontsize=13, y=0.98)

    all_vmax = 0
    matrices = []
    for bi, band in enumerate(band_names[:n_cols]):
        mat = np.full((n_lfp_ch, n_lfp_ch), np.nan)
        for sp_ch in agg:
            if band in agg[sp_ch]:
                for lfp_ch in range(n_lfp_ch):
                    r = agg[sp_ch].get(band, {}).get(lfp_ch)
                    if r is not None and sp_ch < n_lfp_ch:
                        mat[sp_ch, lfp_ch] = r.ppc
        matrices.append(mat)
        v = np.nanmax(mat) if not np.all(np.isnan(mat)) else 0
        if v > all_vmax:
            all_vmax = v

    all_vmax = max(all_vmax, 0.05)

    for bi, band in enumerate(band_names[:n_cols]):
        ax = fig.add_subplot(gs[0, bi])
        mat = matrices[bi]
        masked = np.ma.array(mat, mask=np.isnan(mat))
        im = ax.imshow(masked, aspect='equal', cmap='YlGnBu',
                        vmin=0, vmax=all_vmax, origin='upper')
        ax.set_xticks(range(n_lfp_ch))
        ax.set_xticklabels(ch_labs, rotation=90, fontsize=4)
        ax.set_yticks(range(n_lfp_ch))
        ax.set_yticklabels(ch_labs, fontsize=4)
        ax.set_title(band, fontsize=9)
        if bi == 0:
            ax.set_ylabel('Spike Channel (depth)', fontsize=8)
        ax.set_xlabel('LFP Channel (depth)', fontsize=7)

        # 対角線を強調
        for i in range(n_lfp_ch):
            ax.plot(i, i, 's', color='red', ms=2, alpha=0.3)

    # 共通カラーバー
    cax = fig.add_subplot(gs[0, -1])
    fig.colorbar(im, cax=cax, label='PPC')


# ============================================================
# Page 5: ローカル vs ディスタル + 条件比較
# ============================================================

def _plot_page5(fig, agg, condition_results, bands, n_lfp_ch,
                original_ch_numbers, spike_data):
    """ローカルvs.ディスタル比較 + 条件別MRL"""
    band_names = list(bands.keys())

    has_cond = condition_results and len(condition_results) > 0

    gs = fig.add_gridspec(1, 2 if has_cond else 1, wspace=0.4)
    fig.suptitle('Page 5: Local vs Distal + Condition Comparison',
                 fontsize=13, y=0.98)

    # --- 左: ローカル vs ディスタル ---
    ax_ld = fig.add_subplot(gs[0, 0])

    local_ppc = {b: [] for b in band_names}
    distal_ppc = {b: [] for b in band_names}

    for sp_ch, band_results in agg.items():
        for band in band_names:
            if band not in band_results:
                continue
            for lfp_ch in range(n_lfp_ch):
                r = band_results[band].get(lfp_ch)
                if r is None:
                    continue
                dist = abs(sp_ch - lfp_ch)
                if dist <= 1:
                    local_ppc[band].append(r.ppc)
                else:
                    distal_ppc[band].append(r.ppc)

    x = np.arange(len(band_names))
    width = 0.35
    local_means = [np.mean(local_ppc[b]) if local_ppc[b] else 0 for b in band_names]
    local_sems = [np.std(local_ppc[b]) / np.sqrt(len(local_ppc[b]))
                  if len(local_ppc[b]) > 1 else 0 for b in band_names]
    distal_means = [np.mean(distal_ppc[b]) if distal_ppc[b] else 0 for b in band_names]
    distal_sems = [np.std(distal_ppc[b]) / np.sqrt(len(distal_ppc[b]))
                   if len(distal_ppc[b]) > 1 else 0 for b in band_names]

    ax_ld.bar(x - width / 2, local_means, width, yerr=local_sems,
              color='#2196F3', alpha=0.8, label='Local (|Δch|≤1)', capsize=3)
    ax_ld.bar(x + width / 2, distal_means, width, yerr=distal_sems,
              color='#FF9800', alpha=0.8, label='Distal (|Δch|>1)', capsize=3)
    ax_ld.set_xticks(x)
    ax_ld.set_xticklabels(band_names)
    ax_ld.set_ylabel('PPC')
    ax_ld.set_title('Local vs Distal Phase Locking', fontsize=10)
    ax_ld.legend(fontsize=8)

    # --- 右: 条件比較 ---
    if has_cond:
        ax_cond = fig.add_subplot(gs[0, 1])
        conditions = ['baseline', 'stim', 'post']
        cond_colors = {'baseline': '#888888', 'stim': '#e74c3c', 'post': '#3498db'}

        # 全ユニットのMRLを帯域×条件で集約
        cond_mrl = {b: {c: [] for c in conditions} for b in band_names}
        for unit_key, band_conds in condition_results.items():
            for band in band_names:
                if band not in band_conds:
                    continue
                cond_dict = band_conds[band]
                if not isinstance(cond_dict, dict):
                    continue
                for cond in conditions:
                    r = cond_dict.get(cond)
                    if r is not None:
                        cond_mrl[band][cond].append(r.mrl)

        n_conds = len(conditions)
        group_width = 0.8
        bar_w = group_width / n_conds

        for ci, cond in enumerate(conditions):
            means = [np.mean(cond_mrl[b][cond]) if cond_mrl[b][cond] else 0
                     for b in band_names]
            pos = x + (ci - n_conds / 2 + 0.5) * bar_w
            ax_cond.bar(pos, means, bar_w, color=cond_colors[cond],
                        alpha=0.8, label=cond)

        ax_cond.set_xticks(x)
        ax_cond.set_xticklabels(band_names)
        ax_cond.set_ylabel('MRL')
        ax_cond.set_title('Condition Comparison', fontsize=10)
        ax_cond.legend(fontsize=8)


# ============================================================
# Page 6: PAC + STA
# ============================================================

def _plot_page6(fig, lfp_cleaned, lfp_times, fs, spike_data,
                bands, n_lfp_ch, original_ch_numbers):
    """PAC + STA深さプロファイル"""
    from phase_locking import (compute_phase_amplitude_coupling,
                                compute_spike_triggered_average,
                                aggregate_spikes_by_channel)

    ch_labs = _ch_labels(n_lfp_ch, original_ch_numbers)
    gs = fig.add_gridspec(1, 2, wspace=0.35)
    fig.suptitle('Page 6: PAC + STA', fontsize=13, y=0.98)

    # --- 左: PAC (theta phase × gamma amplitude) ---
    ax_pac = fig.add_subplot(gs[0, 0])
    phase_band = bands.get('theta', list(bands.values())[0])
    amp_band = bands.get('gamma', list(bands.values())[-1])
    phase_label = list(bands.keys())[0] if 'theta' not in bands else 'theta'
    amp_label = list(bands.keys())[-1] if 'gamma' not in bands else 'gamma'

    mi_values = []
    for ch_i in range(n_lfp_ch):
        lfp_1ch = lfp_cleaned[:, ch_i] if lfp_cleaned.ndim > 1 else lfp_cleaned
        mi, _, _ = compute_phase_amplitude_coupling(
            lfp_1ch, fs, phase_band, amp_band)
        mi_values.append(mi)

    bar_colors = plt.cm.YlOrRd(np.array(mi_values) / (max(mi_values) + 1e-10))
    ax_pac.barh(range(n_lfp_ch), mi_values, color=bar_colors, height=0.7)
    ax_pac.set_yticks(range(n_lfp_ch))
    ax_pac.set_yticklabels(ch_labs, fontsize=6)
    ax_pac.invert_yaxis()
    ax_pac.set_xlabel('Modulation Index', fontsize=8)
    ax_pac.set_title(f'PAC: {phase_label} phase x {amp_label} amplitude',
                     fontsize=9)
    ax_pac.tick_params(labelsize=6)

    # --- 右: STA 深さプロファイル ---
    ax_sta = fig.add_subplot(gs[0, 1])
    ch_spikes = aggregate_spikes_by_channel(spike_data, n_lfp_ch)

    # 最もスパイク数が多いチャンネルを代表として使う
    if ch_spikes:
        rep_ch = max(ch_spikes, key=lambda c: len(ch_spikes[c]))
        sta_mean, sta_sem, time_axis = compute_spike_triggered_average(
            ch_spikes[rep_ch], lfp_cleaned, lfp_times, fs,
            window_ms=(-50, 50))

        if sta_mean.ndim == 1:
            sta_mean = sta_mean[:, np.newaxis]
            sta_sem = sta_sem[:, np.newaxis]

        max_amp = np.max(np.abs(sta_mean)) if np.max(np.abs(sta_mean)) > 0 else 1
        offset_scale = max_amp * 2.5
        cmap_ch = plt.cm.viridis(np.linspace(0.1, 0.9, n_lfp_ch))

        for ch_i in range(min(n_lfp_ch, sta_mean.shape[1])):
            offset = ch_i * offset_scale
            ax_sta.plot(time_axis, sta_mean[:, ch_i] + offset,
                        color=cmap_ch[ch_i], lw=0.8)
            ax_sta.fill_between(time_axis,
                                sta_mean[:, ch_i] - sta_sem[:, ch_i] + offset,
                                sta_mean[:, ch_i] + sta_sem[:, ch_i] + offset,
                                color=cmap_ch[ch_i], alpha=0.2)

        ax_sta.axvline(0, color='red', ls='--', lw=0.5, alpha=0.5)
        ax_sta.set_yticks([i * offset_scale for i in range(n_lfp_ch)])
        ax_sta.set_yticklabels(ch_labs, fontsize=5)
        ax_sta.invert_yaxis()
        rep_label = _ch_label(rep_ch, original_ch_numbers)
        ax_sta.set_title(f'STA (trigger: Spike {rep_label}, '
                         f'n={len(ch_spikes[rep_ch])})', fontsize=9)
    else:
        ax_sta.text(0.5, 0.5, 'No spike data', ha='center', va='center')
        ax_sta.set_title('STA', fontsize=9)

    ax_sta.set_xlabel('Time (ms)', fontsize=8)
    ax_sta.tick_params(labelsize=6)


# ============================================================
# メインエントリポイント
# ============================================================

def plot_phase_grand_summary(
    phase_results: Dict[str, Dict],
    condition_results: Dict[str, Dict],
    bands: Dict[str, tuple],
    original_ch_numbers: list,
    spike_data: Dict,
    output_dir: str,
    basename: str,
    lfp_cleaned=None,
    lfp_times=None,
    fs=None,
    stim_times=None,
    channel_spacing_um: float = 50.0,
    fdr_significant=None,
    show: bool = True,
    save: bool = True
):
    """
    位相ロック解析グランドサマリー（6ページ構成）

    Parameters
    ----------
    phase_results : dict  {unit_key: {band: {ch: PhaseLockingResult}}}
    condition_results : dict
    bands : dict  {band_name: (low, high)}
    original_ch_numbers : list
    spike_data : dict
    output_dir, basename : str
    lfp_cleaned : ndarray (n_samples, n_channels)
    lfp_times : ndarray
    fs : int
    stim_times : ndarray or None
    channel_spacing_um : float
    fdr_significant : dict or None  (apply_fdr_to_phase_results の出力)
    show, save : bool
    """
    band_names = list(bands.keys())
    n_lfp_ch = lfp_cleaned.shape[1] if lfp_cleaned is not None and lfp_cleaned.ndim > 1 else 1

    if len(phase_results) == 0:
        print("  グランドサマリー: ユニットなし")
        return

    # チャンネル集約
    agg, ch_spike_counts = _aggregate_by_channel(
        phase_results, spike_data, n_lfp_ch, band_names)
    spike_chs = sorted(agg.keys())

    # PDF
    pdf_path = os.path.join(output_dir, f'{basename}_phase_grand_summary.pdf') if save else None
    pdf = PdfPages(pdf_path) if pdf_path else None

    try:
        # --- Page 1: Recording Context ---
        if lfp_cleaned is not None:
            fig1 = plt.figure(figsize=(16, 8))
            _plot_page1(fig1, lfp_cleaned, lfp_times, fs, stim_times,
                        channel_spacing_um, original_ch_numbers, spike_data, bands)
            if pdf:
                pdf.savefig(fig1, dpi=150, bbox_inches='tight')

        # --- Page 2: PPC + Preferred Phase ---
        fig2 = plt.figure(figsize=(16, 10))
        _plot_page2(fig2, agg, ch_spike_counts, bands, n_lfp_ch,
                    original_ch_numbers, fdr_significant, phase_results, spike_data)
        if pdf:
            pdf.savefig(fig2, dpi=150, bbox_inches='tight')

        # --- Page 3: Polar Matrix (全スパイクchプール、最大10行/ページ) ---
        MAX_ROWS_PER_PAGE = 10
        for ch_start in range(0, n_lfp_ch, MAX_ROWS_PER_PAGE):
            ch_end = min(ch_start + MAX_ROWS_PER_PAGE, n_lfp_ch)
            n_rows = ch_end - ch_start
            fig3 = plt.figure(figsize=(14, max(n_rows * 1.4, 6)))
            _plot_page3(fig3, phase_results, bands, n_lfp_ch,
                        original_ch_numbers, fdr_significant,
                        ch_start=ch_start, ch_end=ch_end)
            if pdf:
                pdf.savefig(fig3, dpi=120, bbox_inches='tight')

        # --- Page 4: Depth x Depth Matrix ---
        fig4 = plt.figure(figsize=(16, 6))
        _plot_page4(fig4, agg, bands, n_lfp_ch, original_ch_numbers)
        if pdf:
            pdf.savefig(fig4, dpi=150, bbox_inches='tight')

        # --- Page 5: Local vs Distal + Condition ---
        fig5 = plt.figure(figsize=(14, 6))
        _plot_page5(fig5, agg, condition_results, bands, n_lfp_ch,
                    original_ch_numbers, spike_data)
        if pdf:
            pdf.savefig(fig5, dpi=150, bbox_inches='tight')

        if pdf:
            pdf.close()
            print(f"  PDF保存: {pdf_path}")

        # PNG (Page 2 のみ)
        if save:
            png_path = os.path.join(output_dir, f'{basename}_phase_grand_summary.png')
            fig2.savefig(png_path, dpi=150, bbox_inches='tight')
            print(f"  PNG保存: {png_path}")

        if show:
            plt.show()
        else:
            plt.close('all')

    except Exception as e:
        if pdf:
            pdf.close()
        print(f"  プロットエラー: {e}")
        import traceback
        traceback.print_exc()
