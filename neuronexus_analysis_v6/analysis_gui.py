"""
analysis_gui.py - 統合解析エクスプローラ GUI

解析結果（スパイク・LFP・位相ロック）をインタラクティブに探索する
Tkinter + matplotlib ベースのビューワ。

構成:
  Tab 1: 🧠 スパイク概要  — 品質テーブル + 波形一覧 + 9パネル詳細
  Tab 2: 📊 LFP解析       — 全ch波形 + PSD + CSD + ヒートマップ
  Tab 3: 🔗 統合解析       — 位相ロック深度 + STA + 条件別MRL
  Tab 4: 💾 エクスポート   — PNG/CSV一括保存

起動方法:
    from analysis_gui import launch_explorer
    launch_explorer(session, lfp_result, sorting_results,
                    protocol=protocol, sla=analyzer, ca=comprehensive)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from scipy import signal
import os
import warnings
from typing import Dict, List, Optional, Tuple, Any

warnings.filterwarnings('ignore')

# フォント設定（日本語・英語両対応）
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['font.size'] = 9

# --- 定数 ---
ELECTRODE_SPACING_UM = 50
N_ELECTRODES = 16
DEPTHS_UM = np.arange(N_ELECTRODES) * ELECTRODE_SPACING_UM

LAYER_BOUNDARIES = {
    'L1':   (0, 100),
    'L2/3': (100, 350),
    'L4':   (350, 500),
    'L5':   (500, 650),
    'L6':   (650, 800),
}
LAYER_COLORS = {
    'L1':   '#E8E8E8', 'L2/3': '#B3D9FF', 'L4': '#FFD9B3',
    'L5':   '#B3FFB3', 'L6':   '#FFB3B3',
}
BAND_COLORS_LIST = ['#3b82f6', '#22c55e', '#f59e0b', '#a855f7',
                     '#ef4444', '#06b6d4', '#ec4899', '#84cc16']
CONDITION_COLORS = {'baseline': '#888888', 'stim': '#e74c3c', 'post': '#3498db'}


def _get_layer(depth):
    for name, (lo, hi) in LAYER_BOUNDARIES.items():
        if lo <= depth < hi:
            return name
    return 'L6'


def _draw_layers(ax, orientation='horizontal', depths=None):
    """層境界の背景色を描画"""
    max_d = (depths[-1] + 25) if depths is not None else 775
    for name, (lo, hi) in LAYER_BOUNDARIES.items():
        hi_clip = min(hi, max_d + 50)
        if lo > max_d + 50:
            continue
        c = LAYER_COLORS[name]
        if orientation == 'horizontal':
            ax.axhspan(lo - 25, hi_clip - 25, color=c, alpha=0.2, zorder=0)
        else:
            ax.axvspan(lo - 25, hi_clip - 25, color=c, alpha=0.2, zorder=0)


# ============================================================
# マルチプロット用 Canvas ウィジェット
# ============================================================

class PlotPanel(ttk.Frame):
    """matplotlib Figure を埋め込むパネル"""

    def __init__(self, parent, figsize=(12, 7), dpi=100):
        super().__init__(parent)
        self.fig = Figure(figsize=figsize, dpi=dpi)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.pack(side='bottom', fill='x')
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=True)

    def clear(self):
        self.fig.clear()

    def draw(self):
        self.fig.tight_layout()
        self.canvas.draw()


# ============================================================
# メインGUI
# ============================================================

class AnalysisExplorerGUI:
    """
    解析結果を対話的に探索する GUI

    Parameters
    ----------
    session : RecordingSession
    lfp_result : dict  (step2 output)
    sorting_results : dict {ch: ChannelSortResult}
    protocol : StimulusProtocol or None
    sla : SpikeLFPAnalyzer or None
    ca : ComprehensiveAnalyzer or None
    """

    def __init__(self, session, lfp_result, sorting_results,
                 protocol=None, sla=None, ca=None):
        self.session = session
        self.lfp_result = lfp_result
        self.sorting = sorting_results or {}
        self.protocol = protocol
        self.sla = sla
        self.ca = ca

        # LFP ショートカット
        self.lfp = lfp_result.get('lfp_cleaned', lfp_result.get('lfp_filtered', np.array([])))
        self.lfp_times = lfp_result.get('lfp_times', np.array([]))
        self.fs = lfp_result.get('fs', 1000)
        self.good_channels = lfp_result.get('good_channels', list(range(self.lfp.shape[1])))
        self.original_ch = lfp_result.get('original_ch_numbers', self.good_channels)
        self.noise_mask = lfp_result.get('noise_mask')
        self.n_lfp_ch = self.lfp.shape[1] if self.lfp.ndim == 2 else 0
        self.depths = DEPTHS_UM[self.good_channels] if len(self.good_channels) == self.n_lfp_ch \
            else DEPTHS_UM[:self.n_lfp_ch]

        # 品質テーブルのキャッシュ
        self._table_data = None

        # --- Tk ウィンドウ ---
        self.root = tk.Tk()
        self.root.title("Neuronexus Analysis Explorer")
        self.root.geometry("1500x950")
        self.root.minsize(1200, 750)

        self._build_gui()

    # ============================================================
    # GUI 構築
    # ============================================================

    def _build_gui(self):
        # タブ
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=4, pady=4)

        # --- Tab 1: スパイク ---
        tab_spike = ttk.Frame(self.notebook)
        self.notebook.add(tab_spike, text=' 🧠 スパイク概要 ')
        self._build_spike_tab(tab_spike)

        # --- Tab 2: LFP ---
        tab_lfp = ttk.Frame(self.notebook)
        self.notebook.add(tab_lfp, text=' 📊 LFP解析 ')
        self._build_lfp_tab(tab_lfp)

        # --- Tab 3: 統合 ---
        tab_integrated = ttk.Frame(self.notebook)
        self.notebook.add(tab_integrated, text=' 🔗 統合解析 ')
        self._build_integrated_tab(tab_integrated)

        # --- Tab 4: エクスポート ---
        tab_export = ttk.Frame(self.notebook)
        self.notebook.add(tab_export, text=' 💾 エクスポート ')
        self._build_export_tab(tab_export)

        # ステータスバー
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var,
                               relief='sunken', anchor='w')
        status_bar.pack(fill='x', side='bottom', padx=4, pady=2)

    # ============================================================
    # Tab 1: スパイク概要
    # ============================================================

    def _build_spike_tab(self, parent):
        pw = ttk.PanedWindow(parent, orient='horizontal')
        pw.pack(fill='both', expand=True)

        # --- 左: テーブル + コントロール ---
        left = ttk.Frame(pw, width=420)
        pw.add(left, weight=1)

        # コントロールバー
        ctrl = ttk.Frame(left)
        ctrl.pack(fill='x', padx=5, pady=5)

        ttk.Label(ctrl, text="Filter:").pack(side='left')
        self.spike_filter_var = tk.StringVar(value="All")
        filter_combo = ttk.Combobox(ctrl, textvariable=self.spike_filter_var,
                                     values=['All', 'SU only', 'MUA only', 'Good+Excellent'],
                                     state='readonly', width=14)
        filter_combo.pack(side='left', padx=5)
        filter_combo.bind('<<ComboboxSelected>>', lambda e: self._refresh_spike_table())

        ttk.Button(ctrl, text="Overview Plot",
                   command=self._plot_spike_overview).pack(side='right', padx=5)

        # Treeview テーブル
        cols = ('Ch', 'U', 'Depth', 'Layer', 'nSpk', 'SNR', 'ISI%', 'FR', 'Quality')
        tree_frame = ttk.Frame(left)
        tree_frame.pack(fill='both', expand=True, padx=5, pady=5)

        scroll_y = ttk.Scrollbar(tree_frame, orient='vertical')
        scroll_y.pack(side='right', fill='y')

        self.spike_tree = ttk.Treeview(tree_frame, columns=cols, show='headings',
                                        yscrollcommand=scroll_y.set, height=25)
        scroll_y.config(command=self.spike_tree.yview)

        widths = {'Ch': 35, 'U': 30, 'Depth': 50, 'Layer': 45,
                  'nSpk': 55, 'SNR': 50, 'ISI%': 50, 'FR': 50, 'Quality': 80}
        for c in cols:
            self.spike_tree.heading(c, text=c,
                                    command=lambda _c=c: self._sort_spike_table(_c))
            self.spike_tree.column(c, width=widths.get(c, 60), anchor='center')

        self.spike_tree.pack(fill='both', expand=True)
        self.spike_tree.bind('<<TreeviewSelect>>', self._on_spike_select)

        # 件数ラベル
        self.spike_count_var = tk.StringVar(value="")
        ttk.Label(left, textvariable=self.spike_count_var,
                  font=('', 9)).pack(fill='x', padx=5, pady=2)

        # --- 右: 詳細プロット ---
        right = ttk.Frame(pw)
        pw.add(right, weight=3)

        self.spike_plot = PlotPanel(right, figsize=(11, 8), dpi=90)
        self.spike_plot.pack(fill='both', expand=True)

        # 初期表示
        self._refresh_spike_table()

    def _get_table_data(self):
        """品質テーブルデータを取得（キャッシュ）"""
        if self._table_data is not None:
            return self._table_data

        rows = []
        for ch, result in sorted(self.sorting.items()):
            depth = DEPTHS_UM[ch] if ch < N_ELECTRODES else ch * ELECTRODE_SPACING_UM
            for unit in result.units:
                dur = self.session.duration if self.session.duration > 0 else 1.0
                fr = unit.n_spikes / dur
                if unit.is_noise:
                    q = 'Noise'
                elif unit.is_mua:
                    q = 'MUA'
                elif unit.isi_violation_rate < 1.0 and unit.snr > 5:
                    q = 'Excellent'
                elif unit.isi_violation_rate < 5.0 and unit.snr > 2.5:
                    q = 'Good SU'
                else:
                    q = 'Fair SU'
                rows.append({
                    'ch': ch, 'uid': unit.unit_id, 'depth': depth,
                    'layer': _get_layer(depth), 'nspk': unit.n_spikes,
                    'snr': round(unit.snr, 2), 'isi': round(unit.isi_violation_rate, 2),
                    'fr': round(fr, 2), 'quality': q,
                })
        self._table_data = rows
        return rows

    def _refresh_spike_table(self):
        """テーブルをフィルタに応じて更新"""
        self.spike_tree.delete(*self.spike_tree.get_children())
        filt = self.spike_filter_var.get()
        data = self._get_table_data()

        for r in data:
            show = True
            if filt == 'SU only' and r['quality'] in ('Noise', 'MUA'):
                show = False
            elif filt == 'MUA only' and r['quality'] != 'MUA':
                show = False
            elif filt == 'Good+Excellent' and r['quality'] not in ('Good SU', 'Excellent'):
                show = False

            if show:
                tag = 'noise' if r['quality'] == 'Noise' else \
                      'mua' if r['quality'] == 'MUA' else \
                      'excellent' if r['quality'] == 'Excellent' else 'su'
                self.spike_tree.insert('', 'end', values=(
                    r['ch'], r['uid'], r['depth'], r['layer'],
                    r['nspk'], r['snr'], r['isi'], r['fr'], r['quality']
                ), tags=(tag,))

        # 色分け
        self.spike_tree.tag_configure('noise', foreground='#999999')
        self.spike_tree.tag_configure('mua', foreground='#cc7700')
        self.spike_tree.tag_configure('excellent', foreground='#009900', font=('', 9, 'bold'))
        self.spike_tree.tag_configure('su', foreground='#333333')

        n_shown = len(self.spike_tree.get_children())
        n_total = len(data)
        n_su = sum(1 for r in data if r['quality'] not in ('Noise', 'MUA'))
        n_mua = sum(1 for r in data if r['quality'] == 'MUA')
        self.spike_count_var.set(
            f"Showing {n_shown}/{n_total}  |  SU:{n_su}  MUA:{n_mua}  Noise:{n_total-n_su-n_mua}")

    def _sort_spike_table(self, col):
        """テーブルの列でソート"""
        items = [(self.spike_tree.set(k, col), k) for k in self.spike_tree.get_children()]
        try:
            items.sort(key=lambda t: float(t[0]))
        except ValueError:
            items.sort(key=lambda t: t[0])
        for idx, (val, k) in enumerate(items):
            self.spike_tree.move(k, '', idx)

    def _on_spike_select(self, event):
        """テーブルの行を選択 → 右に9パネル詳細を表示"""
        sel = self.spike_tree.selection()
        if not sel:
            return
        vals = self.spike_tree.item(sel[0], 'values')
        ch = int(vals[0])
        uid = int(vals[1])
        self.status_var.set(f"Loading Ch{ch} Unit{uid} ...")
        self.root.update_idletasks()
        self._plot_unit_detail(ch, uid)
        self.status_var.set(f"Ch{ch} Unit{uid}")

    def _plot_unit_detail(self, ch, uid):
        """選択ユニットの9パネル詳細（spike_lfp_analysis.plot_unit_summary相当）"""
        self.spike_plot.clear()
        fig = self.spike_plot.fig

        result = self.sorting.get(ch)
        if result is None:
            return

        unit = None
        for u in result.units:
            if u.unit_id == uid:
                unit = u
                break
        if unit is None:
            return

        axes = fig.subplots(3, 3)

        # === Row 1: Sorting quality ===
        # 1-1: Waveforms
        ax = axes[0, 0]
        if unit.waveforms is not None and len(unit.waveforms) > 0:
            n_show = min(80, len(unit.waveforms))
            idx = np.random.choice(len(unit.waveforms), n_show, replace=False)
            t_ms = result.waveform_time_ms if result.waveform_time_ms is not None \
                else np.arange(unit.waveforms.shape[1]) / result.fs * 1000
            for i in idx:
                ax.plot(t_ms, unit.waveforms[i], color=unit.color, alpha=0.1, lw=0.5)
            ax.plot(t_ms, np.mean(unit.waveforms, axis=0), 'k-', lw=2)
            ax.axhline(0, color='gray', ls='--', alpha=0.5)
        ax.set_xlabel('ms', fontsize=8)
        ax.set_title(f'Waveforms (n={unit.n_spikes})', fontsize=9)
        ax.tick_params(labelsize=7)

        # 1-2: PCA
        ax = axes[0, 1]
        if unit.pca_features is not None:
            for other in result.units:
                if other.unit_id != uid and not other.is_noise and other.pca_features is not None:
                    ax.scatter(other.pca_features[:, 0], other.pca_features[:, 1],
                               s=2, alpha=0.15, color='gray')
            ax.scatter(unit.pca_features[:, 0], unit.pca_features[:, 1],
                       s=3, alpha=0.5, color=unit.color)
        ax.set_xlabel('PC1', fontsize=8)
        ax.set_ylabel('PC2', fontsize=8)
        ax.set_title(f'PCA (SNR={unit.snr:.1f}, ISI={unit.isi_violation_rate:.1f}%)', fontsize=9)
        ax.tick_params(labelsize=7)

        # 1-3: ISI histogram
        ax = axes[0, 2]
        if unit.spike_times is not None and len(unit.spike_times) > 1:
            isi_ms = np.diff(unit.spike_times) * 1000
            bins = np.arange(0, min(50, isi_ms.max() + 1), 0.5)
            if len(bins) > 1:
                ax.hist(isi_ms, bins=bins, color=unit.color, alpha=0.7, edgecolor='black', lw=0.3)
                ax.axvline(2.0, color='red', ls='--', lw=1, label='2ms')
        ax.set_xlabel('ISI (ms)', fontsize=8)
        ax.set_title('ISI Histogram', fontsize=9)
        ax.tick_params(labelsize=7)

        # === Row 2: Stimulus response ===
        if self.protocol and unit.spike_times is not None:
            # 2-1: PSTH
            ax = axes[1, 0]
            try:
                self.protocol.plot_psth(unit.spike_times, ax=ax)
                ax.set_title('PSTH', fontsize=9)
            except Exception:
                ax.text(0.5, 0.5, 'PSTH N/A', ha='center', va='center',
                        transform=ax.transAxes, fontsize=9, color='gray')
            ax.tick_params(labelsize=7)

            # 2-2: Raster
            ax = axes[1, 1]
            try:
                self.protocol.plot_raster(unit.spike_times, ax=ax)
                ax.set_title('Raster', fontsize=9)
            except Exception:
                ax.text(0.5, 0.5, 'Raster N/A', ha='center', va='center',
                        transform=ax.transAxes, fontsize=9, color='gray')
            ax.tick_params(labelsize=7)

            # 2-3: Adaptation
            ax = axes[1, 2]
            try:
                adapt = self.protocol.compute_adaptation(unit.spike_times)
                if adapt is not None and adapt.response_rates is not None:
                    positions = np.arange(1, len(adapt.response_rates) + 1)
                    ax.bar(positions, adapt.response_rates, color='steelblue',
                           alpha=0.7, edgecolor='black')
                    ax.set_xlabel('Stim #', fontsize=8)
                    ax.set_ylabel('Rate (Hz)', fontsize=8)
                    ax.set_title('Adaptation', fontsize=9)
            except Exception:
                ax.text(0.5, 0.5, 'Adapt N/A', ha='center', va='center',
                        transform=ax.transAxes, fontsize=9, color='gray')
            ax.tick_params(labelsize=7)
        else:
            for i in range(3):
                axes[1, i].text(0.5, 0.5, 'No protocol', ha='center', va='center',
                                transform=axes[1, i].transAxes, color='gray')

        # === Row 3: Phase locking ===
        unit_key = f"ch{ch}_unit{uid}"
        pl_result = self.sla.unit_results.get(unit_key) if self.sla else None

        if pl_result is not None:
            # 3-1: Phase polar (best band)
            ax_polar = fig.add_subplot(3, 3, 7, projection='polar')
            axes[2, 0].set_visible(False)
            best_band = None
            best_mrl = 0
            best_phases = None
            for band_name, ch_results in pl_result.band_results.items():
                for lfp_ch, res in ch_results.items():
                    if res is not None and res.mrl > best_mrl:
                        best_mrl = res.mrl
                        best_band = band_name
                        best_phases = res.spike_phases if hasattr(res, 'spike_phases') else None

            if best_phases is not None and len(best_phases) > 0:
                bins_polar = np.linspace(-np.pi, np.pi, 37)
                counts, _ = np.histogram(best_phases, bins=bins_polar)
                centers = (bins_polar[:-1] + bins_polar[1:]) / 2
                ax_polar.bar(centers, counts, width=np.diff(bins_polar)[0],
                             color='steelblue', alpha=0.7, edgecolor='white', lw=0.3)
                ax_polar.set_title(f'{best_band}\nMRL={best_mrl:.3f}', fontsize=8, pad=10)
            else:
                ax_polar.set_title(f'Best: {best_band or "N/A"}\nMRL={best_mrl:.3f}',
                                   fontsize=8, pad=10)
            ax_polar.tick_params(labelsize=6)

            # 3-2: MRL heatmap (bands × lfp channels)
            ax = axes[2, 1]
            band_names = list(pl_result.band_results.keys())
            if band_names:
                n_bands = len(band_names)
                n_lfp = self.n_lfp_ch
                heatmap = np.zeros((n_bands, n_lfp))
                for bi, band in enumerate(band_names):
                    for lfp_ch_key, res in pl_result.band_results[band].items():
                        if res is not None and isinstance(lfp_ch_key, int) and lfp_ch_key < n_lfp:
                            heatmap[bi, lfp_ch_key] = res.mrl
                im = ax.imshow(heatmap, aspect='auto', cmap='YlOrRd', vmin=0,
                               origin='lower')
                ax.set_yticks(range(n_bands))
                ax.set_yticklabels(band_names, fontsize=7)
                ax.set_xlabel('LFP Ch', fontsize=8)
                ax.set_title('MRL Heatmap', fontsize=9)
                fig.colorbar(im, ax=ax, shrink=0.7)
            ax.tick_params(labelsize=6)

            # 3-3: STA
            ax = axes[2, 2]
            if pl_result.sta is not None:
                sta = pl_result.sta[:, 0] if pl_result.sta.ndim > 1 else pl_result.sta
                ax.plot(pl_result.sta_time, sta, 'k-', lw=1.5)
                ax.axvline(0, color='red', ls='--', lw=1)
                ax.set_xlabel('ms from spike', fontsize=8)
                ax.set_title('STA', fontsize=9)
            else:
                ax.text(0.5, 0.5, 'No STA', ha='center', va='center',
                        transform=ax.transAxes, color='gray')
            ax.tick_params(labelsize=7)
        else:
            for i in range(3):
                if i == 0:
                    axes[2, 0].text(0.5, 0.5, 'No phase data', ha='center', va='center',
                                    transform=axes[2, 0].transAxes, color='gray')
                else:
                    axes[2, i].text(0.5, 0.5, 'No phase data', ha='center', va='center',
                                    transform=axes[2, i].transAxes, color='gray')

        depth = DEPTHS_UM[ch] if ch < N_ELECTRODES else 0
        fig.suptitle(f'Ch{ch} Unit{uid}  |  Depth {depth}µm ({_get_layer(depth)})  |  '
                     f'SNR={unit.snr:.1f}  ISI={unit.isi_violation_rate:.1f}%  '
                     f'n={unit.n_spikes}',
                     fontsize=11, fontweight='bold')
        self.spike_plot.draw()

    def _plot_spike_overview(self):
        """全チャンネル波形概要"""
        self.spike_plot.clear()
        fig = self.spike_plot.fig
        self.status_var.set("Drawing spike overview ...")
        self.root.update_idletasks()

        sorted_channels = sorted(self.sorting.keys())
        n_ch = len(sorted_channels)
        if n_ch == 0:
            return

        # 4×4 波形 + 右側にメトリクス
        gs = GridSpec(4, 5, figure=fig, hspace=0.4, wspace=0.35)

        n_cols, n_rows = 4, 4
        for idx, ch in enumerate(sorted_channels[:n_rows * n_cols]):
            row, col = idx // n_cols, idx % n_cols
            ax = fig.add_subplot(gs[row, col])
            result = self.sorting[ch]
            depth = DEPTHS_UM[ch] if ch < N_ELECTRODES else 0

            t_ms = None
            for unit in result.units:
                if unit.is_noise or unit.waveforms is None or len(unit.waveforms) == 0:
                    continue
                if t_ms is None:
                    t_ms = result.waveform_time_ms if result.waveform_time_ms is not None \
                        else np.arange(unit.waveforms.shape[1]) / result.fs * 1000
                mean_wf = np.mean(unit.waveforms, axis=0)
                label = f'U{unit.unit_id}'
                ls = '--' if unit.is_mua else '-'
                ax.plot(t_ms, mean_wf, ls, lw=1.5, label=label, color=unit.color)

            ax.set_title(f'Ch{ch} ({depth}µm)', fontsize=8, pad=2)
            ax.tick_params(labelsize=6)
            if row == n_rows - 1:
                ax.set_xlabel('ms', fontsize=7)
            ax.legend(fontsize=6, loc='upper right', framealpha=0.6)

        # 右列: SNR vs ISI
        ax_qi = fig.add_subplot(gs[0:2, 4])
        table = self._get_table_data()
        valid = [r for r in table if r['quality'] != 'Noise']
        if valid:
            su = [r for r in valid if r['quality'] != 'MUA']
            mua = [r for r in valid if r['quality'] == 'MUA']
            if su:
                ax_qi.scatter([r['snr'] for r in su], [r['isi'] for r in su],
                              c=[r['depth'] for r in su], cmap='viridis',
                              s=25, edgecolors='black', lw=0.5, alpha=0.8, label='SU')
            if mua:
                ax_qi.scatter([r['snr'] for r in mua], [r['isi'] for r in mua],
                              c='orange', s=25, marker='s', edgecolors='black',
                              lw=0.5, alpha=0.7, label='MUA')
            ax_qi.axhline(2, color='red', ls='--', alpha=0.4)
            ax_qi.axvline(3, color='blue', ls='--', alpha=0.4)
        ax_qi.set_xlabel('SNR', fontsize=8)
        ax_qi.set_ylabel('ISI viol.(%)', fontsize=8)
        ax_qi.set_title('Unit Quality', fontsize=9)
        ax_qi.legend(fontsize=7)
        ax_qi.tick_params(labelsize=7)

        # 右下: 発火率 × 深度
        ax_fr = fig.add_subplot(gs[2:4, 4])
        if valid:
            for r in valid:
                m = 's' if r['quality'] == 'MUA' else 'o'
                c = 'gray' if r['quality'] == 'MUA' else 'steelblue'
                ax_fr.scatter(r['fr'], r['depth'], s=max(r['nspk'] / 15, 10),
                              marker=m, color=c, alpha=0.7, edgecolors='black', lw=0.5)
            _draw_layers(ax_fr, 'horizontal', self.depths)
            ax_fr.invert_yaxis()
        ax_fr.set_xlabel('FR (Hz)', fontsize=8)
        ax_fr.set_ylabel('Depth (µm)', fontsize=8)
        ax_fr.set_title('FR × Depth', fontsize=9)
        ax_fr.tick_params(labelsize=7)

        n_su = sum(1 for r in table if r['quality'] not in ('Noise', 'MUA'))
        n_mua = sum(1 for r in table if r['quality'] == 'MUA')
        fig.suptitle(f'Spike Overview — {n_ch} ch, {n_su} SU + {n_mua} MUA',
                     fontsize=12, fontweight='bold')
        self.spike_plot.draw()
        self.status_var.set("Spike overview done")

    # ============================================================
    # Tab 2: LFP 解析
    # ============================================================

    def _build_lfp_tab(self, parent):
        pw = ttk.PanedWindow(parent, orient='horizontal')
        pw.pack(fill='both', expand=True)

        # --- 左: コントロール ---
        left = ttk.Frame(pw, width=280)
        pw.add(left, weight=0)

        # 表示モード
        mode_frame = ttk.LabelFrame(left, text="表示モード")
        mode_frame.pack(fill='x', padx=5, pady=5)

        self.lfp_mode_var = tk.StringVar(value="all_channels")
        modes = [
            ("全チャンネル波形", "all_channels"),
            ("パワースペクトル", "power_spectrum"),
            ("チャンネルヒートマップ", "heatmap"),
            ("CSD", "csd"),
            ("帯域パワー × 深度", "band_depth"),
            ("コヒーレンス行列", "coherence"),
        ]
        for text, val in modes:
            ttk.Radiobutton(mode_frame, text=text, value=val,
                            variable=self.lfp_mode_var,
                            command=self._draw_lfp).pack(anchor='w', padx=10, pady=2)

        ttk.Button(mode_frame, text="▶ 再描画",
                   command=self._draw_lfp).pack(fill='x', padx=10, pady=5)

        # 時間範囲
        time_frame = ttk.LabelFrame(left, text="時間範囲 (秒)")
        time_frame.pack(fill='x', padx=5, pady=5)

        t_max = float(self.lfp_times[-1]) if len(self.lfp_times) > 0 else 100
        t_default_end = min(5.0, t_max)

        row_f = ttk.Frame(time_frame)
        row_f.pack(fill='x', padx=5, pady=2)
        ttk.Label(row_f, text="Start:").pack(side='left')
        self.lfp_t_start_var = tk.StringVar(value="0")
        ttk.Entry(row_f, textvariable=self.lfp_t_start_var, width=8).pack(side='left', padx=5)

        row_f2 = ttk.Frame(time_frame)
        row_f2.pack(fill='x', padx=5, pady=2)
        ttk.Label(row_f2, text="End:").pack(side='left')
        self.lfp_t_end_var = tk.StringVar(value=f"{t_default_end:.1f}")
        ttk.Entry(row_f2, textvariable=self.lfp_t_end_var, width=8).pack(side='left', padx=5)

        ttk.Label(time_frame, text=f"(max: {t_max:.1f}s)",
                  font=('', 8), foreground='gray').pack(padx=5, pady=2)

        # チャンネル選択
        ch_frame = ttk.LabelFrame(left, text="チャンネル (PSD等)")
        ch_frame.pack(fill='x', padx=5, pady=5)

        self.lfp_ch_var = tk.StringVar(value="avg")
        ch_vals = ["avg"] + [f"Ch{c}" for c in self.original_ch]
        ttk.Combobox(ch_frame, textvariable=self.lfp_ch_var,
                     values=ch_vals, state='readonly', width=10).pack(padx=10, pady=5)

        # 条件マスク表示
        mask_frame = ttk.LabelFrame(left, text="条件マスク表示")
        mask_frame.pack(fill='x', padx=5, pady=5)
        self.show_masks_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(mask_frame, text="Baseline/Stim/Post 表示",
                        variable=self.show_masks_var).pack(padx=10, pady=3)
        self.show_stim_lines_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(mask_frame, text="刺激タイミング線",
                        variable=self.show_stim_lines_var).pack(padx=10, pady=3)

        # 帯域設定
        band_frame = ttk.LabelFrame(left, text="周波数帯域プリセット")
        band_frame.pack(fill='x', padx=5, pady=5)
        self.band_preset_var = tk.StringVar(value="Standard")
        ttk.Combobox(band_frame, textvariable=self.band_preset_var,
                     values=['Standard', 'High Gamma', 'Rodent', 'Simple'],
                     state='readonly', width=14).pack(padx=10, pady=5)

        # 振幅スケール
        amp_frame = ttk.LabelFrame(left, text="振幅スケール (µV)")
        amp_frame.pack(fill='x', padx=5, pady=5)
        self.lfp_amp_var = tk.StringVar(value="auto")
        ttk.Radiobutton(amp_frame, text="自動", value="auto",
                        variable=self.lfp_amp_var).pack(anchor='w', padx=10, pady=1)
        for val in ["100", "200", "500", "1000"]:
            ttk.Radiobutton(amp_frame, text=val, value=val,
                            variable=self.lfp_amp_var).pack(anchor='w', padx=10, pady=1)

        # --- 右: プロット ---
        right = ttk.Frame(pw)
        pw.add(right, weight=3)

        self.lfp_plot = PlotPanel(right, figsize=(12, 8), dpi=90)
        self.lfp_plot.pack(fill='both', expand=True)

    def _get_lfp_t_range(self):
        try:
            t0 = float(self.lfp_t_start_var.get())
            t1 = float(self.lfp_t_end_var.get())
        except ValueError:
            t0, t1 = 0, 5
        return max(0, t0), min(float(self.lfp_times[-1]), t1)

    def _get_bands(self):
        presets = {
            'Standard': {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 14),
                         'beta': (14, 30), 'gamma': (30, 80)},
            'High Gamma': {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13),
                           'beta': (13, 30), 'low_gamma': (30, 60), 'high_gamma': (60, 120)},
            'Rodent': {'delta': (1, 4), 'theta': (6, 10), 'alpha': (10, 14),
                       'beta': (14, 30), 'gamma': (30, 100)},
            'Simple': {'low': (1, 30), 'high': (30, 100)},
        }
        return presets.get(self.band_preset_var.get(), presets['Standard'])

    def _get_condition_masks(self):
        """条件マスクを取得"""
        if self.protocol is None:
            return None
        try:
            return self.protocol.create_condition_masks(self.lfp_times, self.fs)
        except Exception:
            return None

    def _draw_lfp(self):
        mode = self.lfp_mode_var.get()
        self.lfp_plot.clear()
        self.status_var.set(f"Drawing LFP: {mode} ...")
        self.root.update_idletasks()

        fig = self.lfp_plot.fig

        try:
            if mode == 'all_channels':
                self._draw_lfp_all_channels(fig)
            elif mode == 'power_spectrum':
                self._draw_lfp_power(fig)
            elif mode == 'heatmap':
                self._draw_lfp_heatmap(fig)
            elif mode == 'csd':
                self._draw_lfp_csd(fig)
            elif mode == 'band_depth':
                self._draw_lfp_band_depth(fig)
            elif mode == 'coherence':
                self._draw_lfp_coherence(fig)
        except Exception as e:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f'Error: {e}', ha='center', va='center',
                    transform=ax.transAxes, color='red', fontsize=12)

        self.lfp_plot.draw()
        self.status_var.set(f"LFP: {mode} done")

    def _draw_lfp_all_channels(self, fig):
        """全チャンネルLFP波形 + 条件マスク + 刺激ライン"""
        t0, t1 = self._get_lfp_t_range()
        tmask = (self.lfp_times >= t0) & (self.lfp_times <= t1)
        times = self.lfp_times[tmask]

        ax = fig.add_subplot(111)

        # 振幅スケール
        amp_str = self.lfp_amp_var.get()
        if amp_str == "auto":
            spacing = np.percentile(np.abs(self.lfp[tmask]), 95) * 2.5
            if spacing < 1e-6:
                spacing = np.std(self.lfp[tmask]) * 6
            if spacing < 1e-6:
                spacing = 1.0
        else:
            spacing = float(amp_str)

        # 条件マスク背景
        if self.show_masks_var.get():
            masks = self._get_condition_masks()
            if masks is not None:
                ymin = -self.n_lfp_ch * spacing - spacing
                ymax = spacing
                for cond, color in CONDITION_COLORS.items():
                    if cond in masks:
                        ax.fill_between(times, ymin, ymax, where=masks[cond][tmask],
                                        color=color, alpha=0.12, label=cond.capitalize())
                # ノイズマスク
                if self.noise_mask is not None:
                    ax.fill_between(times, ymin, ymax, where=self.noise_mask[tmask],
                                    color='gray', alpha=0.3, label='Noise')

        # LFP波形
        for i in range(self.n_lfp_ch):
            offset = -i * spacing
            ax.plot(times, self.lfp[tmask, i] + offset, 'k-', lw=0.5)
            depth = self.depths[i] if i < len(self.depths) else i * 50
            layer = _get_layer(depth)
            ax.text(t0 - (t1 - t0) * 0.01, offset,
                    f'{depth}µm\n{layer}', fontsize=7, ha='right', va='center',
                    color='steelblue')

        # 刺激タイミング
        if self.show_stim_lines_var.get() and self.protocol is not None:
            for st in self.protocol.stim_times:
                if t0 <= st <= t1:
                    ax.axvline(st, color='red', alpha=0.2, lw=0.5)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Depth')
        ax.set_yticks([])
        ax.set_xlim(t0, t1)
        ax.set_title(f'All Channels LFP ({t0:.1f}–{t1:.1f}s)')
        if self.show_masks_var.get():
            ax.legend(fontsize=8, loc='upper right', ncol=4, framealpha=0.7)

    def _draw_lfp_power(self, fig):
        """パワースペクトル（plotting.py の plot_power_analysis 相当）"""
        bands = self._get_bands()
        band_names = list(bands.keys())

        # チャンネル選択
        ch_str = self.lfp_ch_var.get()
        if ch_str == 'avg':
            data = np.mean(self.lfp, axis=1)
        else:
            ch_idx = self.original_ch.index(int(ch_str.replace('Ch', '')))
            data = self.lfp[:, ch_idx]

        # 条件マスク
        masks = self._get_condition_masks()

        axes = fig.subplots(2, 2)

        # 条件別 PSD 計算
        def compute_psd(mask):
            clean = mask
            if self.noise_mask is not None:
                clean = clean & ~self.noise_mask
            d = data[clean] if data.ndim == 1 else data[clean]
            if len(d) < 256:
                return np.array([]), np.array([])
            return signal.welch(d, fs=self.fs, nperseg=min(1024, len(d) // 2))

        if masks is not None:
            freqs_b, psd_base = compute_psd(masks.get('baseline', np.ones(len(data), dtype=bool)))
            freqs_s, psd_stim = compute_psd(masks.get('stim', np.ones(len(data), dtype=bool)))
            freqs_p, psd_post = compute_psd(masks.get('post', np.ones(len(data), dtype=bool)))
        else:
            freqs_b, psd_base = signal.welch(data, fs=self.fs, nperseg=1024)
            psd_stim = psd_base
            psd_post = psd_base
            freqs_s = freqs_p = freqs_b

        # 1-1: PSD + 帯域背景
        ax = axes[0, 0]
        for i, (bname, (lo, hi)) in enumerate(bands.items()):
            c = BAND_COLORS_LIST[i % len(BAND_COLORS_LIST)]
            ax.axvspan(lo, hi, alpha=0.15, color=c)
        if len(freqs_b) > 0:
            ax.semilogy(freqs_b, psd_base, 'b-', lw=1.5, label='Baseline')
            ax.semilogy(freqs_s, psd_stim, 'r-', lw=1.5, label='Stim')
            ax.semilogy(freqs_p, psd_post, 'g-', lw=1.5, label='Post')
        ax.set_xlim(0, 100)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power')
        ax.set_title('Power Spectrum')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 1-2: 変化率
        ax = axes[0, 1]
        if len(freqs_b) > 0 and len(psd_base) > 0:
            change_stim = (psd_stim - psd_base) / np.maximum(psd_base, 1e-10) * 100
            change_post = (psd_post - psd_base) / np.maximum(psd_base, 1e-10) * 100
            fmask = freqs_b <= 100
            for i, (bname, (lo, hi)) in enumerate(bands.items()):
                c = BAND_COLORS_LIST[i % len(BAND_COLORS_LIST)]
                ax.axvspan(lo, hi, alpha=0.15, color=c)
            ax.plot(freqs_b[fmask], change_stim[fmask], 'r-', lw=1.5, label='Stim')
            ax.plot(freqs_b[fmask], change_post[fmask], 'g-', lw=1.5, label='Post')
            ax.axhline(0, color='k', ls='--')
        ax.set_xlim(0, 100)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Change (%)')
        ax.set_title('Power Change')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 2-1: バンドパワー棒グラフ
        ax = axes[1, 0]
        if len(freqs_b) > 0:
            bp_base = [np.mean(psd_base[(freqs_b >= lo) & (freqs_b < hi)])
                       for _, (lo, hi) in bands.items()]
            bp_stim = [np.mean(psd_stim[(freqs_s >= lo) & (freqs_s < hi)])
                       for _, (lo, hi) in bands.items()]
            bp_post = [np.mean(psd_post[(freqs_p >= lo) & (freqs_p < hi)])
                       for _, (lo, hi) in bands.items()]
            x = np.arange(len(band_names))
            w = 0.25
            ax.bar(x - w, bp_base, w, label='Baseline', color='blue', alpha=0.7)
            ax.bar(x, bp_stim, w, label='Stim', color='red', alpha=0.7)
            ax.bar(x + w, bp_post, w, label='Post', color='green', alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(band_names, rotation=30, ha='right')
        ax.set_ylabel('Power')
        ax.set_title('Band Power')
        ax.legend(fontsize=8)

        # 2-2: 変化率棒グラフ
        ax = axes[1, 1]
        if len(freqs_b) > 0:
            change_s = [(s - b) / max(b, 1e-10) * 100
                        for s, b in zip(bp_stim, bp_base)]
            change_p = [(p - b) / max(b, 1e-10) * 100
                        for p, b in zip(bp_post, bp_base)]
            ax.bar(x - w / 2, change_s, w, label='Stim', color='red', alpha=0.7)
            ax.bar(x + w / 2, change_p, w, label='Post', color='green', alpha=0.7)
            ax.axhline(0, color='k', lw=1)
            ax.set_xticks(x)
            ax.set_xticklabels(band_names, rotation=30, ha='right')
        ax.set_ylabel('Change (%)')
        ax.set_title('Band Power Change')
        ax.legend(fontsize=8)

        ch_label = ch_str if ch_str != 'avg' else 'All Ch Average'
        fig.suptitle(f'Power Spectrum — {ch_label}', fontsize=12, fontweight='bold')

    def _draw_lfp_heatmap(self, fig):
        """チャンネル × 帯域 パワー変化ヒートマップ"""
        bands = self._get_bands()
        band_names = list(bands.keys())
        masks = self._get_condition_masks()

        axes = fig.subplots(1, 2)

        for ax_idx, (label, cond) in enumerate([('Stim vs Baseline', 'stim'),
                                                  ('Post vs Baseline', 'post')]):
            ax = axes[ax_idx]
            change_matrix = np.zeros((self.n_lfp_ch, len(band_names)))

            for ch_i in range(self.n_lfp_ch):
                d = self.lfp[:, ch_i]
                for bi, (bname, (lo, hi)) in enumerate(bands.items()):
                    if masks is not None:
                        base_m = masks.get('baseline', np.ones(len(d), dtype=bool))
                        cond_m = masks.get(cond, np.ones(len(d), dtype=bool))
                        if self.noise_mask is not None:
                            base_m = base_m & ~self.noise_mask
                            cond_m = cond_m & ~self.noise_mask
                    else:
                        base_m = cond_m = np.ones(len(d), dtype=bool)

                    def bp(mask):
                        seg = d[mask]
                        if len(seg) < 256:
                            return 0
                        f, p = signal.welch(seg, fs=self.fs, nperseg=min(1024, len(seg) // 2))
                        fm = (f >= lo) & (f < hi)
                        return np.mean(p[fm]) if np.any(fm) else 0

                    b_power = bp(base_m)
                    c_power = bp(cond_m)
                    change_matrix[ch_i, bi] = (c_power - b_power) / max(b_power, 1e-10) * 100

            ch_labels = [f'{self.depths[i]}µm' for i in range(self.n_lfp_ch)]
            im = ax.imshow(change_matrix, aspect='auto', cmap='RdBu_r',
                           vmin=-np.percentile(np.abs(change_matrix), 95),
                           vmax=np.percentile(np.abs(change_matrix), 95),
                           origin='upper')
            ax.set_xticks(range(len(band_names)))
            ax.set_xticklabels(band_names, rotation=30, ha='right', fontsize=8)
            ax.set_yticks(range(self.n_lfp_ch))
            ax.set_yticklabels(ch_labels, fontsize=7)
            ax.set_title(label, fontsize=10)
            fig.colorbar(im, ax=ax, label='Change (%)', shrink=0.8)

        fig.suptitle('Channel × Band Power Change Heatmap', fontsize=12, fontweight='bold')

    def _draw_lfp_csd(self, fig):
        """CSD表示（ComprehensiveAnalyzer経由 or 直接計算）"""
        if self.ca is not None:
            try:
                avg_csd, csd_times, csd_depths = self.ca.compute_trial_averaged_csd()
            except Exception as e:
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, f'CSD error: {e}', ha='center', va='center',
                        transform=ax.transAxes, color='red')
                return
        else:
            # 直接計算
            if self.n_lfp_ch < 3:
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, 'Need ≥3 channels for CSD', ha='center', va='center',
                        transform=ax.transAxes, color='gray')
                return
            dz = ELECTRODE_SPACING_UM * 1e-6
            csd_full = np.zeros((len(self.lfp), self.n_lfp_ch - 2))
            for i in range(1, self.n_lfp_ch - 1):
                csd_full[:, i - 1] = -(self.lfp[:, i - 1] - 2 * self.lfp[:, i] + self.lfp[:, i + 1]) / dz ** 2
            csd_depths = self.depths[1:-1]

            if self.protocol is None or len(self.protocol.stim_times) == 0:
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, 'Need protocol for trial-avg CSD', ha='center', va='center',
                        transform=ax.transAxes, color='gray')
                return

            pre_s, post_s = int(0.05 * self.fs), int(0.2 * self.fs)
            total = pre_s + post_s
            csd_sum = np.zeros((total, self.n_lfp_ch - 2))
            n_v = 0
            for st in self.protocol.stim_times:
                idx = int(st * self.fs)
                if idx - pre_s >= 0 and idx + post_s < len(csd_full):
                    csd_sum += csd_full[idx - pre_s:idx + post_s]
                    n_v += 1
            avg_csd = csd_sum / max(n_v, 1)
            csd_times = np.linspace(-50, 200, total)

        axes = fig.subplots(1, 3)

        # CSD ヒートマップ
        ax = axes[0]
        vmax = np.percentile(np.abs(avg_csd), 95)
        im = ax.pcolormesh(csd_times, csd_depths, avg_csd.T,
                           cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
        ax.axvline(0, color='black', lw=1.5, ls='--')
        ax.invert_yaxis()
        ax.set_xlabel('ms from stim')
        ax.set_ylabel('Depth (µm)')
        ax.set_title('Trial-Averaged CSD')
        fig.colorbar(im, ax=ax, shrink=0.8)

        # Trial平均LFP
        ax = axes[1]
        pre_s = int(0.05 * self.fs)
        post_s = int(0.2 * self.fs)
        total = pre_s + post_s
        lfp_sum = np.zeros((total, self.n_lfp_ch))
        n_v = 0
        for st in self.protocol.stim_times:
            idx = int(st * self.fs)
            if idx - pre_s >= 0 and idx + post_s < len(self.lfp):
                lfp_sum += self.lfp[idx - pre_s:idx + post_s]
                n_v += 1
        avg_lfp = lfp_sum / max(n_v, 1)
        t_ms = np.linspace(-50, 200, total)
        spacing_lfp = np.percentile(np.abs(avg_lfp), 98) * 1.5
        if spacing_lfp == 0:
            spacing_lfp = 1
        for i in range(self.n_lfp_ch):
            ax.plot(t_ms, avg_lfp[:, i] - i * spacing_lfp, 'k-', lw=0.8)
            ax.text(-55, -i * spacing_lfp, f'{self.depths[i]}', fontsize=6,
                    ha='right', va='center', color='steelblue')
        ax.axvline(0, color='red', ls='--', lw=1)
        ax.set_xlabel('ms from stim')
        ax.set_yticks([])
        ax.set_title(f'Trial-Avg LFP (n={n_v})')

        # CSD断面（ピーク時刻）
        ax = axes[2]
        post_idx = csd_times >= 0
        if np.any(post_idx):
            peak_idx = np.argmax(np.max(np.abs(avg_csd[post_idx, :]), axis=1))
            peak_time = csd_times[post_idx][peak_idx]
            csd_profile = avg_csd[post_idx, :][peak_idx, :]
            ax.barh(csd_depths, csd_profile, height=ELECTRODE_SPACING_UM * 0.8,
                    color=['#e74c3c' if v > 0 else '#3498db' for v in csd_profile], alpha=0.7)
            ax.axvline(0, color='gray', ls='--')
            ax.invert_yaxis()
            _draw_layers(ax, 'horizontal', csd_depths)
        ax.set_xlabel('CSD amplitude')
        ax.set_ylabel('Depth (µm)')
        ax.set_title(f'CSD @ {peak_time:.1f}ms')

        fig.suptitle('Current Source Density', fontsize=12, fontweight='bold')

    def _draw_lfp_band_depth(self, fig):
        """帯域パワー × 深度プロファイル"""
        bands = self._get_bands()

        # チャンネル別 PSD
        clean = ~self.noise_mask if self.noise_mask is not None else \
            np.ones(len(self.lfp), dtype=bool)
        data = self.lfp[clean]
        freqs, psd = signal.welch(data, fs=self.fs, nperseg=min(1024, len(data) // 2), axis=0)

        axes = fig.subplots(1, 2)

        # PSD ヒートマップ
        ax = axes[0]
        fmask = freqs <= 100
        psd_db = 10 * np.log10(psd[fmask, :] + 1e-10)
        im = ax.pcolormesh(freqs[fmask], self.depths, psd_db.T,
                           cmap='YlOrRd', shading='auto')
        ax.invert_yaxis()
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Depth (µm)')
        ax.set_title('PSD by Depth')
        fig.colorbar(im, ax=ax, label='dB')

        # 帯域パワー深度
        ax = axes[1]
        colors = ['blue', 'green', 'orange', 'red', 'purple', 'cyan', 'magenta', 'brown']
        for i, (bname, (lo, hi)) in enumerate(bands.items()):
            bmask = (freqs >= lo) & (freqs < hi)
            bp = np.mean(psd[bmask, :], axis=0)
            bp_norm = bp / bp.max() if bp.max() > 0 else bp
            ax.plot(bp_norm, self.depths, 'o-', color=colors[i % len(colors)],
                    label=bname, lw=1.5, markersize=4)
        ax.invert_yaxis()
        _draw_layers(ax, 'horizontal', self.depths)
        ax.set_xlabel('Normalized Power')
        ax.set_ylabel('Depth (µm)')
        ax.set_title('Band Power Depth Profile')
        ax.legend(fontsize=8)

        fig.suptitle('LFP Power × Depth', fontsize=12, fontweight='bold')

    def _draw_lfp_coherence(self, fig):
        """コヒーレンス行列"""
        axes = fig.subplots(1, 2)
        band_pairs = [('Theta (4-8Hz)', (4, 8)), ('Gamma (30-80Hz)', (30, 80))]

        for idx, (title, freq_range) in enumerate(band_pairs):
            ax = axes[idx]
            if self.ca is not None:
                coh = self.ca._compute_coherence_matrix(freq_range)
            else:
                clean = ~self.noise_mask if self.noise_mask is not None else \
                    np.ones(len(self.lfp), dtype=bool)
                data = self.lfp[clean]
                n = data.shape[1]
                coh = np.eye(n)
                for i in range(n):
                    for j in range(i + 1, n):
                        f, c = signal.coherence(data[:, i], data[:, j],
                                                fs=self.fs, nperseg=1024)
                        fm = (f >= freq_range[0]) & (f <= freq_range[1])
                        coh[i, j] = coh[j, i] = np.mean(c[fm])

            im = ax.imshow(coh, cmap='viridis', vmin=0, vmax=1, origin='upper')
            labels = [f'{d}' for d in self.depths]
            ax.set_xticks(range(self.n_lfp_ch))
            ax.set_xticklabels(labels, fontsize=6, rotation=45)
            ax.set_yticks(range(self.n_lfp_ch))
            ax.set_yticklabels(labels, fontsize=6)
            ax.set_title(title)
            fig.colorbar(im, ax=ax, label='Coherence', shrink=0.8)

        fig.suptitle('Channel Coherence Matrix', fontsize=12, fontweight='bold')

    # ============================================================
    # Tab 3: 統合解析
    # ============================================================

    def _build_integrated_tab(self, parent):
        pw = ttk.PanedWindow(parent, orient='horizontal')
        pw.pack(fill='both', expand=True)

        # --- 左: コントロール ---
        left = ttk.Frame(pw, width=280)
        pw.add(left, weight=0)

        mode_frame = ttk.LabelFrame(left, text="表示モード")
        mode_frame.pack(fill='x', padx=5, pady=5)

        self.integ_mode_var = tk.StringVar(value="grand_summary")
        modes = [
            ("グランドサマリー", "grand_summary"),
            ("位相ロック × 深度", "pl_depth"),
            ("STA × 深度", "sta_depth"),
            ("条件別 MRL", "condition_mrl"),
            ("FR × 深度 (条件別)", "fr_depth"),
            ("Population 位相ロック", "population_pl"),
        ]
        for text, val in modes:
            ttk.Radiobutton(mode_frame, text=text, value=val,
                            variable=self.integ_mode_var,
                            command=self._draw_integrated).pack(anchor='w', padx=10, pady=2)

        ttk.Button(mode_frame, text="▶ 再描画",
                   command=self._draw_integrated).pack(fill='x', padx=10, pady=5)

        # 帯域選択（位相ロック用）
        band_frame = ttk.LabelFrame(left, text="帯域 (位相ロック)")
        band_frame.pack(fill='x', padx=5, pady=5)

        pl_bands = []
        if self.sla and self.sla.freq_bands:
            pl_bands = list(self.sla.freq_bands.keys())
        elif self.sla and self.sla.unit_results:
            first = list(self.sla.unit_results.values())[0]
            pl_bands = list(first.band_results.keys())
        if not pl_bands:
            pl_bands = ['theta', 'gamma']

        self.integ_band_var = tk.StringVar(value=pl_bands[0] if pl_bands else 'theta')
        ttk.Combobox(band_frame, textvariable=self.integ_band_var,
                     values=pl_bands, state='readonly', width=14).pack(padx=10, pady=5)

        # 情報
        info_frame = ttk.LabelFrame(left, text="解析情報")
        info_frame.pack(fill='x', padx=5, pady=5)

        n_units = len(self.sla.unit_results) if self.sla else 0
        n_sig = 0
        if self.sla:
            for pr in self.sla.unit_results.values():
                for band, crs in pr.band_results.items():
                    for _, res in crs.items():
                        if res is not None and res.significant:
                            n_sig += 1
        info_text = f"Units: {n_units}\nSignificant pairs: {n_sig}\n"
        info_text += f"Channels: {len(self.sorting)}\nLFP Ch: {self.n_lfp_ch}"
        ttk.Label(info_frame, text=info_text, font=('', 9)).pack(padx=5, pady=5)

        # --- 右: プロット ---
        right = ttk.Frame(pw)
        pw.add(right, weight=3)

        self.integ_plot = PlotPanel(right, figsize=(12, 8), dpi=90)
        self.integ_plot.pack(fill='both', expand=True)

    def _draw_integrated(self):
        mode = self.integ_mode_var.get()
        self.integ_plot.clear()
        self.status_var.set(f"Drawing: {mode} ...")
        self.root.update_idletasks()

        fig = self.integ_plot.fig

        try:
            if mode == 'grand_summary':
                self._draw_grand_summary(fig)
            elif mode == 'pl_depth':
                self._draw_pl_depth(fig)
            elif mode == 'sta_depth':
                self._draw_sta_depth(fig)
            elif mode == 'condition_mrl':
                self._draw_condition_mrl(fig)
            elif mode == 'fr_depth':
                self._draw_fr_depth(fig)
            elif mode == 'population_pl':
                self._draw_population_pl(fig)
        except Exception as e:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f'Error: {e}', ha='center', va='center',
                    transform=ax.transAxes, color='red', fontsize=12)
            import traceback; traceback.print_exc()

        self.integ_plot.draw()
        self.status_var.set(f"Integrated: {mode} done")

    def _draw_grand_summary(self, fig):
        """12パネルグランドサマリー（comprehensiveの改良版）"""
        if self.ca is not None:
            # ComprehensiveAnalyzer のメソッドを直接使う（Figureを渡す）
            self.ca.plot_grand_summary(fig=fig)
            return

        # ca がなければ簡易版
        gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.4)

        # 1-1: 波形概要
        ax = fig.add_subplot(gs[0, 0])
        table = self._get_table_data()
        valid = [r for r in table if r['quality'] != 'Noise']
        if valid:
            su = [r for r in valid if r['quality'] != 'MUA']
            mua = [r for r in valid if r['quality'] == 'MUA']
            if su:
                ax.scatter([r['snr'] for r in su], [r['isi'] for r in su],
                           c=[r['depth'] for r in su], cmap='viridis', s=20,
                           edgecolors='black', lw=0.5)
            if mua:
                ax.scatter([r['snr'] for r in mua], [r['isi'] for r in mua],
                           c='orange', s=20, marker='s', edgecolors='black', lw=0.5)
        ax.set_xlabel('SNR', fontsize=8)
        ax.set_ylabel('ISI%', fontsize=8)
        ax.set_title('Unit Quality', fontsize=9)
        ax.tick_params(labelsize=7)

        # 1-2: FR × depth
        ax = fig.add_subplot(gs[0, 1])
        if valid:
            for r in valid:
                m = 's' if r['quality'] == 'MUA' else 'o'
                c = 'gray' if r['quality'] == 'MUA' else 'steelblue'
                ax.scatter(r['fr'], r['depth'], s=25, marker=m, color=c,
                           edgecolors='black', lw=0.3)
            ax.invert_yaxis()
            _draw_layers(ax, 'horizontal', self.depths)
        ax.set_xlabel('FR (Hz)', fontsize=8)
        ax.set_ylabel('Depth', fontsize=8)
        ax.set_title('FR × Depth', fontsize=9)
        ax.tick_params(labelsize=7)

        # 1-3: Band power depth
        ax = fig.add_subplot(gs[0, 2])
        try:
            clean = ~self.noise_mask if self.noise_mask is not None else \
                np.ones(len(self.lfp), dtype=bool)
            freqs, psd = signal.welch(self.lfp[clean], fs=self.fs, nperseg=1024, axis=0)
            for bname, (lo, hi), c in [('θ', (4, 8), 'blue'), ('γ', (30, 80), 'red')]:
                bm = (freqs >= lo) & (freqs < hi)
                bp = np.mean(psd[bm, :], axis=0)
                bp_n = bp / bp.max() if bp.max() > 0 else bp
                ax.plot(bp_n, self.depths, 'o-', color=c, label=bname, lw=1.5, ms=3)
            ax.invert_yaxis()
            ax.legend(fontsize=7)
        except Exception:
            pass
        ax.set_xlabel('Norm. power', fontsize=8)
        ax.set_title('Band Power', fontsize=9)
        ax.tick_params(labelsize=7)

        # 2-1, 2-2, 2-3: Phase locking
        if self.sla and self.sla.unit_results:
            # MRL depth
            ax = fig.add_subplot(gs[1, 0])
            band_names = list(self.sla.freq_bands.keys()) if self.sla.freq_bands else []
            cols = plt.cm.Set1(np.linspace(0, 0.8, max(len(band_names), 1)))
            for bi, band in enumerate(band_names):
                for uk, pr in self.sla.unit_results.items():
                    depth = DEPTHS_UM[pr.channel] if pr.channel < N_ELECTRODES else 0
                    best = 0
                    if band in pr.band_results:
                        for _, res in pr.band_results[band].items():
                            if res is not None and res.mrl > best:
                                best = res.mrl
                    if best > 0:
                        ax.scatter(best, depth, s=20, color=cols[bi], alpha=0.7,
                                   edgecolors='black', lw=0.3)
            ax.invert_yaxis()
            if band_names:
                ax.legend(band_names, fontsize=5, loc='lower right')
            ax.set_xlabel('MRL', fontsize=8)
            ax.set_ylabel('Depth', fontsize=8)
            ax.set_title('PL Depth', fontsize=9)
            ax.tick_params(labelsize=7)

            # Condition MRL
            ax = fig.add_subplot(gs[1, 1])
            cond_mrl = {'baseline': [], 'stim': [], 'post': []}
            for pr in self.sla.unit_results.values():
                for cond, res in pr.condition_results.items():
                    if res is not None and cond in cond_mrl:
                        cond_mrl[cond].append(res.mrl)
            conds = ['baseline', 'stim', 'post']
            cc = ['gray', 'red', 'blue']
            means = [np.mean(cond_mrl[c]) if cond_mrl[c] else 0 for c in conds]
            sems = [np.std(cond_mrl[c]) / np.sqrt(max(len(cond_mrl[c]), 1))
                    for c in conds]
            ax.bar(conds, means, yerr=sems, color=cc, alpha=0.7,
                   edgecolor='black', capsize=3)
            ax.set_ylabel('MRL', fontsize=8)
            ax.set_title('MRL × Condition', fontsize=9)
            ax.tick_params(labelsize=7)

            # STA (first unit)
            ax = fig.add_subplot(gs[1, 2])
            first_key = list(self.sla.unit_results.keys())[0]
            pr = self.sla.unit_results[first_key]
            if pr.sta is not None:
                sta = pr.sta[:, 0] if pr.sta.ndim > 1 else pr.sta
                ax.plot(pr.sta_time, sta, 'k-', lw=1.5)
                ax.axvline(0, color='red', ls='--', lw=1)
                ax.set_title(f'STA: {first_key}', fontsize=9)
            ax.set_xlabel('ms', fontsize=8)
            ax.tick_params(labelsize=7)
        else:
            for r in range(2):
                for c in range(3):
                    if r == 1:
                        ax = fig.add_subplot(gs[1, c])
                        ax.text(0.5, 0.5, 'No phase data', ha='center', va='center',
                                transform=ax.transAxes, color='gray')

        # 3rd row: CSD (if protocol available)
        if self.protocol is not None:
            ax = fig.add_subplot(gs[2, 0])
            try:
                if self.ca:
                    avg_csd, ct, cd = self.ca.compute_trial_averaged_csd()
                    vmax = np.percentile(np.abs(avg_csd), 95)
                    ax.pcolormesh(ct, cd, avg_csd.T, cmap='RdBu_r',
                                  vmin=-vmax, vmax=vmax, shading='auto')
                    ax.axvline(0, color='black', lw=1, ls='--')
                    ax.invert_yaxis()
                ax.set_title('CSD', fontsize=9)
            except Exception:
                ax.text(0.5, 0.5, 'CSD N/A', ha='center', va='center',
                        transform=ax.transAxes, color='gray')
            ax.tick_params(labelsize=7)

        fig.suptitle('Grand Summary', fontsize=12, fontweight='bold')

    def _draw_pl_depth(self, fig):
        """位相ロック × 深度"""
        if not self.sla or not self.sla.unit_results:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No phase locking data', ha='center', va='center',
                    transform=ax.transAxes, color='gray')
            return

        band_names = list(self.sla.freq_bands.keys()) if self.sla.freq_bands \
            else list(list(self.sla.unit_results.values())[0].band_results.keys())
        n_bands = len(band_names)

        axes = fig.subplots(1, n_bands, sharey=True)
        if n_bands == 1:
            axes = [axes]

        for bi, band in enumerate(band_names):
            ax = axes[bi]
            for uk, pr in self.sla.unit_results.items():
                depth = DEPTHS_UM[pr.channel] if pr.channel < N_ELECTRODES else 0
                best_mrl = 0
                sig = False
                if band in pr.band_results:
                    for _, res in pr.band_results[band].items():
                        if res is not None and res.mrl > best_mrl:
                            best_mrl = res.mrl
                            sig = res.significant
                marker = 'o' if not pr.is_mua else 's'
                color = 'steelblue' if sig else 'lightgray'
                edge = 'black' if sig else 'gray'
                ax.scatter(best_mrl, depth, s=40, marker=marker, color=color,
                           edgecolors=edge, lw=0.5, alpha=0.8)
            ax.invert_yaxis()
            _draw_layers(ax, 'horizontal', self.depths)
            ax.set_xlabel('MRL', fontsize=8)
            ax.set_title(band, fontsize=10)
            ax.tick_params(labelsize=7)

        axes[0].set_ylabel('Depth (µm)', fontsize=9)
        fig.suptitle('Phase Locking × Depth (filled=significant)', fontsize=12, fontweight='bold')

    def _draw_sta_depth(self, fig):
        """STA × 深度"""
        if not self.sla or not self.sla.unit_results:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No STA data', ha='center', va='center',
                    transform=ax.transAxes, color='gray')
            return

        ax = fig.add_subplot(111)

        items = [(DEPTHS_UM[pr.channel] if pr.channel < N_ELECTRODES else 0, uk, pr)
                 for uk, pr in self.sla.unit_results.items()]
        items.sort(key=lambda x: x[0])

        spacing = 0
        max_amp = 0
        for depth, uk, pr in items:
            if pr.sta is not None:
                sta = pr.sta[:, 0] if pr.sta.ndim > 1 else pr.sta
                max_amp = max(max_amp, np.max(np.abs(sta)))
        if max_amp == 0:
            max_amp = 1
        spacing = max_amp * 2

        for i, (depth, uk, pr) in enumerate(items):
            if pr.sta is not None:
                sta = pr.sta[:, 0] if pr.sta.ndim > 1 else pr.sta
                offset = -i * spacing
                ax.plot(pr.sta_time, sta + offset, 'k-', lw=1)
                ax.text(pr.sta_time[0] - 2, offset, f'{depth}µm\n{_get_layer(depth)}',
                        fontsize=7, ha='right', va='center', color='steelblue')

        ax.axvline(0, color='red', ls='--', lw=1)
        ax.set_xlabel('ms from spike')
        ax.set_yticks([])
        ax.set_title('STA Depth Profile')
        fig.suptitle('Spike-Triggered Average × Depth', fontsize=12, fontweight='bold')

    def _draw_condition_mrl(self, fig):
        """条件別 MRL 比較"""
        if not self.sla or not self.sla.unit_results:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No phase locking data', ha='center', va='center',
                    transform=ax.transAxes, color='gray')
            return

        band = self.integ_band_var.get()
        axes = fig.subplots(1, 3, sharey=True)

        conds = ['baseline', 'stim', 'post']
        for ci, cond in enumerate(conds):
            ax = axes[ci]
            for uk, pr in self.sla.unit_results.items():
                depth = DEPTHS_UM[pr.channel] if pr.channel < N_ELECTRODES else 0
                mrl = 0
                if cond in pr.condition_results and pr.condition_results[cond] is not None:
                    mrl = pr.condition_results[cond].mrl
                marker = 's' if pr.is_mua else 'o'
                ax.scatter(mrl, depth, s=40, marker=marker,
                           color=CONDITION_COLORS[cond], edgecolors='black', lw=0.5, alpha=0.8)
            ax.invert_yaxis()
            _draw_layers(ax, 'horizontal', self.depths)
            ax.set_xlabel('MRL', fontsize=9)
            ax.set_title(cond.capitalize(), fontsize=10)

        axes[0].set_ylabel('Depth (µm)')
        fig.suptitle(f'Phase Locking by Condition ({band})', fontsize=12, fontweight='bold')

    def _draw_fr_depth(self, fig):
        """条件別 発火率 × 深度"""
        if self.ca is not None:
            self.ca.plot_firing_rate_by_condition(fig=fig)
            return

        if self.protocol is None:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'Need protocol', ha='center', va='center',
                    transform=ax.transAxes, color='gray')
            return

        masks = self.protocol.create_condition_masks(self.lfp_times, self.fs)
        axes = fig.subplots(1, 3, sharey=True)
        conditions = [('baseline', 'gray'), ('stim', 'red'), ('post', 'blue')]

        for ax, (cond, color) in zip(axes, conditions):
            mask = masks[cond]
            if self.noise_mask is not None:
                mask = mask & ~self.noise_mask
            duration = np.sum(mask) / self.fs

            for ch, result in sorted(self.sorting.items()):
                depth = DEPTHS_UM[ch] if ch < N_ELECTRODES else 0
                for unit in result.units:
                    if unit.is_noise:
                        continue
                    si = np.searchsorted(self.lfp_times, unit.spike_times)
                    si = np.clip(si, 0, len(mask) - 1)
                    n_in = np.sum(mask[si])
                    fr = n_in / duration if duration > 0 else 0
                    m = 's' if unit.is_mua else 'o'
                    ax.scatter(fr, depth, s=40, marker=m, color=color,
                               alpha=0.7, edgecolors='black', lw=0.5)

            _draw_layers(ax, 'horizontal', self.depths)
            ax.invert_yaxis()
            ax.set_xlabel('FR (Hz)')
            ax.set_title(f'{cond.capitalize()}\n({duration:.1f}s)')

        axes[0].set_ylabel('Depth (µm)')
        fig.suptitle('Firing Rate × Depth × Condition', fontsize=12, fontweight='bold')

    def _draw_population_pl(self, fig):
        """Population 位相ロックサマリー（spike_lfp_analysis.plot_population_summary相当）"""
        if not self.sla or not self.sla.unit_results:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No phase locking data', ha='center', va='center',
                    transform=ax.transAxes, color='gray')
            return

        axes = fig.subplots(2, 3)

        all_mrl = {}
        all_phase = {}
        unit_snr, unit_mrl, unit_n = [], [], []

        for uk, pr in self.sla.unit_results.items():
            best = 0
            for band, crs in pr.band_results.items():
                if band not in all_mrl:
                    all_mrl[band] = []
                    all_phase[band] = []
                for _, res in crs.items():
                    if res is not None:
                        all_mrl[band].append(res.mrl)
                        if res.significant:
                            all_phase[band].append(res.preferred_phase)
                        if res.mrl > best:
                            best = res.mrl
            unit_snr.append(pr.snr)
            unit_mrl.append(best)
            unit_n.append(pr.n_spikes)

        band_names = list(all_mrl.keys())

        # 1-1: MRL boxplot
        ax = axes[0, 0]
        data = [all_mrl[b] for b in band_names]
        if data and any(len(d) > 0 for d in data):
            bp = ax.boxplot(data, labels=band_names, patch_artist=True)
            for p in bp['boxes']:
                p.set_facecolor('steelblue')
                p.set_alpha(0.6)
        ax.set_ylabel('MRL')
        ax.set_title('MRL by Band')
        ax.tick_params(axis='x', rotation=45, labelsize=7)

        # 1-2: Phase distribution
        ax = axes[0, 1]
        colors_pl = plt.cm.Set2(np.linspace(0, 1, len(band_names)))
        for i, band in enumerate(band_names):
            phases = all_phase.get(band, [])
            if phases:
                ax.scatter(np.degrees(phases), [i] * len(phases),
                           color=colors_pl[i], s=20, alpha=0.7)
        ax.set_xlabel('Phase (°)')
        ax.set_yticks(range(len(band_names)))
        ax.set_yticklabels(band_names, fontsize=7)
        ax.set_xlim(-180, 180)
        ax.axvline(0, color='gray', ls='--', alpha=0.5)
        ax.set_title('Preferred Phase (sig.)')

        # 1-3: Mean MRL bar
        ax = axes[0, 2]
        means = [np.mean(all_mrl[b]) if all_mrl[b] else 0 for b in band_names]
        sems = [np.std(all_mrl[b]) / np.sqrt(max(len(all_mrl[b]), 1)) for b in band_names]
        ax.bar(band_names, means, yerr=sems, color='steelblue', alpha=0.7,
               edgecolor='black', capsize=3)
        ax.set_ylabel('Mean MRL')
        ax.set_title('PL by Band')
        ax.tick_params(axis='x', rotation=45, labelsize=7)

        # 2-1: MRL vs SNR
        ax = axes[1, 0]
        ax.scatter(unit_snr, unit_mrl, s=25, alpha=0.7, color='steelblue')
        ax.set_xlabel('SNR')
        ax.set_ylabel('Best MRL')
        ax.set_title('MRL vs SNR')

        # 2-2: MRL vs n_spikes
        ax = axes[1, 1]
        ax.scatter(unit_n, unit_mrl, s=25, alpha=0.7, color='steelblue')
        ax.set_xlabel('n Spikes')
        ax.set_ylabel('Best MRL')
        ax.set_title('MRL vs Spike Count')

        # 2-3: Condition
        ax = axes[1, 2]
        cond_mrl = {'baseline': [], 'stim': [], 'post': []}
        for pr in self.sla.unit_results.values():
            for cond, res in pr.condition_results.items():
                if res is not None and cond in cond_mrl:
                    cond_mrl[cond].append(res.mrl)
        conds = ['baseline', 'stim', 'post']
        cc = ['gray', 'red', 'blue']
        means_c = [np.mean(cond_mrl[c]) if cond_mrl[c] else 0 for c in conds]
        sems_c = [np.std(cond_mrl[c]) / np.sqrt(max(len(cond_mrl[c]), 1)) for c in conds]
        ax.bar(conds, means_c, yerr=sems_c, color=cc, alpha=0.7,
               edgecolor='black', capsize=3)
        ax.set_ylabel('MRL')
        ax.set_title('MRL by Condition')

        n_u = len(self.sla.unit_results)
        fig.suptitle(f'Population Phase Locking ({n_u} units)', fontsize=12, fontweight='bold')

    # ============================================================
    # Tab 4: エクスポート
    # ============================================================

    def _build_export_tab(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill='both', expand=True, padx=20, pady=20)

        ttk.Label(frame, text="💾 エクスポート",
                  font=('', 14, 'bold')).pack(pady=10)

        # 保存先
        dir_frame = ttk.Frame(frame)
        dir_frame.pack(fill='x', pady=10)
        ttk.Label(dir_frame, text="保存先:").pack(side='left')
        self.export_dir_var = tk.StringVar(value="output/explorer/")
        ttk.Entry(dir_frame, textvariable=self.export_dir_var, width=40).pack(side='left', padx=5)
        ttk.Button(dir_frame, text="参照",
                   command=lambda: self.export_dir_var.set(
                       filedialog.askdirectory() or self.export_dir_var.get()
                   )).pack(side='left', padx=5)

        # チェックボックス
        self.export_opts = {}
        opts = [
            ("spike_overview", "スパイク概要プロット (PNG)", True),
            ("spike_table", "品質テーブル (CSV)", True),
            ("lfp_power", "パワースペクトル (PNG)", True),
            ("lfp_heatmap", "チャンネルヒートマップ (PNG)", True),
            ("lfp_csd", "CSD (PNG)", True),
            ("lfp_coherence", "コヒーレンス (PNG)", False),
            ("pl_depth", "位相ロック × 深度 (PNG)", True),
            ("grand_summary", "グランドサマリー (PNG)", True),
            ("unit_details", "全ユニット詳細 (PNG, 個別)", False),
        ]
        opt_frame = ttk.LabelFrame(frame, text="出力項目")
        opt_frame.pack(fill='x', pady=10)
        for key, label, default in opts:
            var = tk.BooleanVar(value=default)
            self.export_opts[key] = var
            ttk.Checkbutton(opt_frame, text=label, variable=var).pack(anchor='w', padx=20, pady=2)

        # 実行ボタン
        ttk.Button(frame, text="▶ 一括エクスポート",
                   command=self._do_export).pack(pady=20)

        self.export_log_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.export_log_var,
                  font=('', 9), foreground='gray').pack(pady=5)

    def _do_export(self):
        out_dir = self.export_dir_var.get()
        os.makedirs(out_dir, exist_ok=True)
        log = []
        self.status_var.set("Exporting ...")
        self.root.update_idletasks()

        try:
            if self.export_opts['spike_table'].get():
                import csv
                path = os.path.join(out_dir, 'spike_quality_table.csv')
                data = self._get_table_data()
                if data:
                    with open(path, 'w', newline='') as f:
                        w = csv.DictWriter(f, fieldnames=data[0].keys())
                        w.writeheader()
                        w.writerows(data)
                    log.append(f"✓ {path}")

            plot_exports = {
                'spike_overview': ('spike_overview.png', self._export_spike_overview),
                'lfp_power': ('lfp_power.png', self._export_lfp_power),
                'lfp_heatmap': ('lfp_heatmap.png', self._export_lfp_heatmap),
                'lfp_csd': ('lfp_csd.png', self._export_lfp_csd),
                'lfp_coherence': ('lfp_coherence.png', self._export_lfp_coherence),
                'pl_depth': ('phase_locking_depth.png', self._export_pl_depth),
                'grand_summary': ('grand_summary.png', self._export_grand_summary),
            }

            for key, (fname, func) in plot_exports.items():
                if self.export_opts[key].get():
                    path = os.path.join(out_dir, fname)
                    try:
                        func(path)
                        log.append(f"✓ {fname}")
                    except Exception as e:
                        log.append(f"✗ {fname}: {e}")

            if self.export_opts['unit_details'].get():
                ud_dir = os.path.join(out_dir, 'unit_details')
                os.makedirs(ud_dir, exist_ok=True)
                for ch, result in sorted(self.sorting.items()):
                    for unit in result.units:
                        if unit.is_noise:
                            continue
                        path = os.path.join(ud_dir, f'ch{ch}_unit{unit.unit_id}.png')
                        try:
                            tmp_fig = Figure(figsize=(14, 10), dpi=100)
                            # Use a simplified export
                            tmp_fig.text(0.5, 0.5, f'Ch{ch} Unit{unit.unit_id}',
                                         ha='center', va='center')
                            tmp_fig.savefig(path, dpi=150, bbox_inches='tight')
                            plt.close('all')
                        except Exception:
                            pass
                log.append(f"✓ unit_details/ ({len(os.listdir(ud_dir))} files)")

        except Exception as e:
            log.append(f"Error: {e}")

        self.export_log_var.set("\n".join(log))
        self.status_var.set(f"Export done → {out_dir}")
        messagebox.showinfo("完了", f"エクスポート完了\n{out_dir}\n\n" + "\n".join(log))

    def _save_fig(self, draw_func, path, figsize=(14, 10)):
        """ヘルパー: Figure作成 → 描画 → 保存"""
        fig = Figure(figsize=figsize, dpi=150)
        draw_func(fig)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close('all')

    def _export_spike_overview(self, path):
        self._save_fig(lambda fig: self._draw_spike_overview_on_fig(fig), path, (18, 14))

    def _draw_spike_overview_on_fig(self, fig):
        """spike overview を任意の Figure に描画"""
        gs = GridSpec(4, 5, figure=fig, hspace=0.4, wspace=0.35)
        sorted_channels = sorted(self.sorting.keys())
        for idx, ch in enumerate(sorted_channels[:16]):
            row, col = idx // 4, idx % 4
            ax = fig.add_subplot(gs[row, col])
            result = self.sorting[ch]
            depth = DEPTHS_UM[ch] if ch < N_ELECTRODES else 0
            t_ms = None
            for unit in result.units:
                if unit.is_noise or unit.waveforms is None or len(unit.waveforms) == 0:
                    continue
                if t_ms is None:
                    t_ms = result.waveform_time_ms if result.waveform_time_ms is not None \
                        else np.arange(unit.waveforms.shape[1]) / result.fs * 1000
                ax.plot(t_ms, np.mean(unit.waveforms, axis=0), lw=1.5, color=unit.color,
                        label=f'U{unit.unit_id}')
            ax.set_title(f'Ch{ch}({depth}µm)', fontsize=8)
            ax.tick_params(labelsize=6)
            ax.legend(fontsize=5, loc='upper right')

        table = self._get_table_data()
        valid = [r for r in table if r['quality'] != 'Noise']
        ax_qi = fig.add_subplot(gs[0:2, 4])
        su = [r for r in valid if r['quality'] != 'MUA']
        mua = [r for r in valid if r['quality'] == 'MUA']
        if su:
            ax_qi.scatter([r['snr'] for r in su], [r['isi'] for r in su],
                          c=[r['depth'] for r in su], cmap='viridis', s=20,
                          edgecolors='black', lw=0.5)
        if mua:
            ax_qi.scatter([r['snr'] for r in mua], [r['isi'] for r in mua],
                          c='orange', s=20, marker='s', edgecolors='black', lw=0.5)
        ax_qi.set_xlabel('SNR')
        ax_qi.set_ylabel('ISI%')
        ax_qi.set_title('Quality')

        ax_fr = fig.add_subplot(gs[2:4, 4])
        for r in valid:
            m = 's' if r['quality'] == 'MUA' else 'o'
            c = 'gray' if r['quality'] == 'MUA' else 'steelblue'
            ax_fr.scatter(r['fr'], r['depth'], s=20, marker=m, color=c,
                          edgecolors='black', lw=0.5)
        ax_fr.invert_yaxis()
        ax_fr.set_xlabel('FR (Hz)')
        ax_fr.set_ylabel('Depth')

    def _export_lfp_power(self, path):
        self._save_fig(self._draw_lfp_power, path, (12, 8))

    def _export_lfp_heatmap(self, path):
        self._save_fig(self._draw_lfp_heatmap, path, (12, 6))

    def _export_lfp_csd(self, path):
        self._save_fig(self._draw_lfp_csd, path, (14, 6))

    def _export_lfp_coherence(self, path):
        self._save_fig(self._draw_lfp_coherence, path, (12, 6))

    def _export_pl_depth(self, path):
        self._save_fig(self._draw_pl_depth, path, (16, 8))

    def _export_grand_summary(self, path):
        self._save_fig(self._draw_grand_summary, path, (16, 14))

    # ============================================================
    # 起動
    # ============================================================

    def run(self):
        """GUIメインループ"""
        self.root.mainloop()


# ============================================================
# ランチャー関数
# ============================================================

def launch_explorer(session, lfp_result, sorting_results,
                    protocol=None, sla=None, ca=None):
    """
    解析エクスプローラGUIを起動

    Parameters
    ----------
    session : RecordingSession
    lfp_result : dict
    sorting_results : dict {ch: ChannelSortResult}
    protocol : StimulusProtocol, optional
    sla : SpikeLFPAnalyzer, optional
    ca : ComprehensiveAnalyzer, optional
    """
    gui = AnalysisExplorerGUI(
        session, lfp_result, sorting_results,
        protocol=protocol, sla=sla, ca=ca
    )
    gui.run()
    return gui


if __name__ == '__main__':
    print("usage: from analysis_gui import launch_explorer")
    print("       launch_explorer(session, lfp_result, sorting_results, ...)")
