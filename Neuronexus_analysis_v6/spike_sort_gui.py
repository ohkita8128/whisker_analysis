"""
spike_sort_gui.py - スパイクソーティング GUI (v6)

改善点:
- 設定パネル（フィルタ、検出閾値、PCA、クラスタリング）をGUI内で変更可能
- 全チャンネル一括ソーティング + 個別チャンネル手動修正
- 品質テーブル表示
- グランドサマリー & 選択チャンネル詳細プロットの出力
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import json
import os
import matplotlib
if matplotlib.get_backend() == '' or matplotlib.get_backend().lower() == 'agg':
    matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from typing import Dict, List, Optional

from spike_sorting import (
    SortingConfig, ChannelSortResult, SpikeUnit,
    sort_all_channels, merge_units, delete_unit, undelete_unit,
    mark_as_mua, unmark_mua, recluster, save_sorting_results,
    export_spike_times_csv, compute_isi_histogram, compute_autocorrelogram
)

SPIKE_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "spike_config.json")


class SpikeSortGUI:
    """スパイクソーティングGUI（設定パネル付き）"""

    def __init__(self, wideband_data: np.ndarray = None, fs: float = None,
                 results: Dict[int, ChannelSortResult] = None,
                 output_dir: str = "", basename: str = "",
                 on_done=None):
        self.wideband_data = wideband_data
        self.fs = fs
        self.results = results or {}
        self.output_dir = output_dir
        self.basename = basename
        self.on_done = on_done
        self.current_channel = 0
        self.selected_units: List[int] = []

        self.root = tk.Tk()
        self.root.title("Spike Sorting GUI v6")
        self.root.geometry("1500x950")
        self.root.minsize(1200, 700)
        self._build_gui()

        if self.results:
            self._update_channel_list()
            self._update_display()

    def _build_gui(self):
        # === 左: 設定パネル ===
        pane = ttk.PanedWindow(self.root, orient='horizontal')
        pane.pack(fill='both', expand=True, padx=3, pady=3)

        left = ttk.Frame(pane, width=280)
        pane.add(left, weight=0)
        self._build_config_panel(left)

        right = ttk.Frame(pane)
        pane.add(right, weight=1)
        self._build_main_panel(right)

    def _build_config_panel(self, parent):
        """設定パネル"""
        lf = ttk.LabelFrame(parent, text="⚙ ソーティング設定")
        lf.pack(fill='both', expand=True, padx=3, pady=3)

        self.cfg_vars = {}

        # --- ソーティング手法選択 ---
        row_method = 0
        ttk.Label(lf, text="🧪 ソーティング手法", font=('', 9, 'bold')).grid(
            row=row_method, column=0, columnspan=2, sticky='w', pady=(8, 2), padx=5)
        row_method += 1

        # KiloSort4 利用可能チェック
        try:
            from kilosort_wrapper import is_kilosort_available
            ks_available = is_kilosort_available()
        except Exception:
            ks_available = False

        methods = ['GMM (内蔵)']
        if ks_available:
            methods.append('KiloSort4')

        self.method_var = tk.StringVar(master=self.root, value='GMM (内蔵)')
        ttk.Label(lf, text="手法", font=('', 8)).grid(
            row=row_method, column=0, sticky='w', padx=10, pady=1)
        method_combo = ttk.Combobox(lf, textvariable=self.method_var,
                                     values=methods, state='readonly', width=12)
        method_combo.grid(row=row_method, column=1, sticky='w', pady=1)
        row_method += 1

        if not ks_available:
            ttk.Label(lf, text="(KiloSort4 未インストール)", font=('', 7),
                      foreground='gray').grid(
                row=row_method, column=0, columnspan=2, sticky='w', padx=10)
            row_method += 1

        # --- パラメータリスト（row_method 以降に配置） ---
        params = [
            ("🔧 フィルタ", None),
            ("filter_low", "ハイパス (Hz)", 300.0),
            ("filter_high", "ローパス (Hz)", 3000.0),
            ("filter_order", "フィルタ次数", 4),
            ("⚡ 検出", None),
            ("threshold_std", "閾値 (σ倍数)", 4.0),
            ("min_spike_interval_ms", "最小間隔 (ms)", 1.0),
            ("artifact_threshold_std", "アーティファクト閾値", 10.0),
            ("✂ 波形", None),
            ("pre_spike_ms", "スパイク前 (ms)", 0.5),
            ("post_spike_ms", "スパイク後 (ms)", 1.0),
            ("📊 PCA / クラスタ", None),
            ("n_pca_components", "PCA成分数", 3),
            ("max_clusters", "最大クラスタ数", 5),
            ("min_cluster_size", "最小クラスタサイズ", 20),
            ("isi_violation_threshold_ms", "ISI不応期 (ms)", 2.0),
            ("🏷 MUA自動判定", None),
            ("mua_isi_threshold", "MUA ISI違反率閾値 (%)", 5.0),
            ("mua_snr_threshold", "MUA SNR閾値", 2.0),
            ("🔬 KiloSort4 設定", None),
            ("channel_spacing_um", "チャンネル間隔 (μm)", 25.0),
            ("n_templates", "テンプレート数", 6),
        ]

        row = row_method
        for item in params:
            if item[1] is None:
                # セクションヘッダー
                ttk.Label(lf, text=item[0], font=('', 9, 'bold')).grid(
                    row=row, column=0, columnspan=2, sticky='w', pady=(8, 2), padx=5)
                row += 1
                continue

            key, label, default = item
            ttk.Label(lf, text=label, font=('', 8)).grid(
                row=row, column=0, sticky='w', padx=10, pady=1)
            var = tk.StringVar(master=self.root, value=str(default))
            self.cfg_vars[key] = var
            ttk.Entry(lf, textvariable=var, width=8).grid(
                row=row, column=1, sticky='w', pady=1)
            row += 1

        # ボタン
        bf = ttk.Frame(lf)
        bf.grid(row=row, column=0, columnspan=2, pady=10)
        ttk.Button(bf, text="▶ 全ch ソート", command=self._run_sorting).pack(fill='x', pady=2)
        ttk.Button(bf, text="💾 設定保存", command=self._save_cfg).pack(fill='x', pady=2)
        ttk.Button(bf, text="📂 設定読込", command=self._load_cfg).pack(fill='x', pady=2)

        self._load_cfg()

    def _build_main_panel(self, parent):
        """メインパネル（プロット + ユニット操作）"""
        # 上部: チャンネル選択
        ctrl = ttk.Frame(parent)
        ctrl.pack(fill='x', padx=5, pady=3)

        ttk.Label(ctrl, text="Channel:").pack(side='left', padx=3)
        self.ch_var = tk.StringVar(master=self.root)
        self.ch_combo = ttk.Combobox(ctrl, textvariable=self.ch_var,
                                      state='readonly', width=10)
        self.ch_combo.pack(side='left', padx=3)
        self.ch_combo.bind('<<ComboboxSelected>>', self._on_ch_change)
        ttk.Button(ctrl, text="<", command=self._prev_ch, width=3).pack(side='left')
        ttk.Button(ctrl, text=">", command=self._next_ch, width=3).pack(side='left')

        ttk.Separator(ctrl, orient='vertical').pack(side='left', fill='y', padx=8)
        ttk.Label(ctrl, text="Re-cluster:").pack(side='left')
        self.nclust_var = tk.StringVar(master=self.root, value="4")
        ttk.Spinbox(ctrl, from_=1, to=10, width=4,
                     textvariable=self.nclust_var).pack(side='left', padx=2)
        ttk.Button(ctrl, text="Re-cluster", command=self._recluster).pack(side='left', padx=3)

        ttk.Separator(ctrl, orient='vertical').pack(side='left', fill='y', padx=8)
        ttk.Button(ctrl, text="📊 Grand Summary", command=self._plot_grand_summary).pack(side='left', padx=3)
        ttk.Button(ctrl, text="Save NPZ", command=self._save_npz).pack(side='right', padx=3)
        ttk.Button(ctrl, text="Export CSV", command=self._export_csv).pack(side='right', padx=3)
        ttk.Button(ctrl, text="✅ 完了", command=self._finish,
                   style="Run.TButton").pack(side='right', padx=5)

        # 中央: プロット
        self.fig = Figure(figsize=(13, 7), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # 下部: ユニットリスト + 操作
        bottom = ttk.Frame(parent)
        bottom.pack(fill='x', padx=5, pady=3)

        lf = ttk.LabelFrame(bottom, text="Units")
        lf.pack(side='left', fill='both', expand=True)

        cols = ('unit', 'n_spikes', 'amp', 'snr', 'isi', 'status')
        self.tree = ttk.Treeview(lf, columns=cols, show='headings', height=5,
                                  selectmode='extended')
        for c, w in zip(cols, [60, 70, 90, 50, 70, 80]):
            self.tree.heading(c, text=c.replace('_', ' ').title())
            self.tree.column(c, width=w, anchor='center')
        self.tree.pack(side='left', fill='both', expand=True)
        self.tree.bind('<<TreeviewSelect>>', self._on_sel)

        af = ttk.LabelFrame(bottom, text="Actions")
        af.pack(side='right', fill='y', padx=5)
        for text, cmd in [("Merge", self._merge), ("Delete", self._delete),
                          ("Undelete", self._undelete), ("Mark MUA", self._mark_mua),
                          ("Unmark MUA", self._unmark_mua)]:
            ttk.Button(af, text=text, command=cmd, width=10).pack(pady=1, padx=3)

        self.status_var = tk.StringVar(master=self.root, value="Ready")
        ttk.Label(parent, textvariable=self.status_var, relief='sunken').pack(
            fill='x', side='bottom')

    # ============================
    # ソーティング実行
    # ============================
    def _get_sorting_config(self) -> SortingConfig:
        cfg = SortingConfig()
        for key, var in self.cfg_vars.items():
            try:
                val = var.get()
                if hasattr(cfg, key):
                    field_type = type(getattr(cfg, key))
                    setattr(cfg, key, field_type(val))
            except:
                pass
        return cfg

    def _run_sorting(self):
        if self.wideband_data is None:
            messagebox.showwarning("Warning", "Widebandデータがありません")
            return

        cfg = self._get_sorting_config()
        method = self.method_var.get()
        method_label = method

        msg = (f"全チャンネルのスパイクソーティングを実行します\n\n"
               f"手法: {method_label}\n"
               f"フィルタ: {cfg.filter_low}-{cfg.filter_high} Hz\n"
               f"閾値: {cfg.threshold_std}σ\n"
               f"最大クラスタ: {cfg.max_clusters}\n"
               f"チャンネル数: {self.wideband_data.shape[1]}")

        if not messagebox.askyesno("確認", msg):
            return

        self.status_var.set(f"ソーティング実行中 ({method_label})...")
        self.root.update()

        try:
            if method == 'KiloSort4':
                from kilosort_wrapper import run_kilosort_sorting
                try:
                    spacing = float(self.cfg_vars['channel_spacing_um'].get())
                except (KeyError, ValueError):
                    spacing = 25.0
                try:
                    n_tmpl = int(self.cfg_vars['n_templates'].get())
                except (KeyError, ValueError):
                    n_tmpl = 6
                self.results = run_kilosort_sorting(
                    self.wideband_data, self.fs, cfg,
                    output_dir=self.output_dir,
                    channel_spacing_um=spacing,
                    n_templates=n_tmpl, verbose=True)
            else:
                self.results = sort_all_channels(
                    self.wideband_data, self.fs, cfg, verbose=True)
        except Exception as e:
            messagebox.showerror("エラー", f"ソーティング中にエラーが発生しました:\n{e}")
            self.status_var.set("エラー")
            return

        self._update_channel_list()
        self._update_display()
        total_units = sum(len(r.units) for r in self.results.values())
        self.status_var.set(f"完了 ({method_label}) - {total_units} units")

    # ============================
    # 表示更新
    # ============================
    def _update_channel_list(self):
        chs = sorted(self.results.keys())
        self.ch_combo['values'] = [f"Ch {ch}" for ch in chs]
        if chs:
            self.current_channel = chs[0]
            self.ch_combo.set(f"Ch {self.current_channel}")

    def _on_ch_change(self, e=None):
        sel = self.ch_var.get()
        if sel:
            self.current_channel = int(sel.replace("Ch ", ""))
            self.selected_units = []
            self._update_display()

    def _prev_ch(self):
        chs = sorted(self.results.keys())
        if not chs:
            return
        idx = chs.index(self.current_channel) if self.current_channel in chs else 0
        self.current_channel = chs[(idx - 1) % len(chs)]
        self.ch_combo.set(f"Ch {self.current_channel}")
        self.selected_units = []
        self._update_display()

    def _next_ch(self):
        chs = sorted(self.results.keys())
        if not chs:
            return
        idx = chs.index(self.current_channel) if self.current_channel in chs else 0
        self.current_channel = chs[(idx + 1) % len(chs)]
        self.ch_combo.set(f"Ch {self.current_channel}")
        self.selected_units = []
        self._update_display()

    def _update_display(self):
        if self.current_channel not in self.results:
            return
        result = self.results[self.current_channel]
        self._update_tree(result)
        self._update_plots(result)

    def _update_tree(self, result):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for u in result.units:
            status = ("NOISE" if u.is_noise else "MUA" if u.is_mua else
                      "Good" if u.isi_violation_rate < 2 else
                      "Fair" if u.isi_violation_rate < 5 else "Poor")
            self.tree.insert('', 'end', iid=str(u.unit_id),
                             values=(f'U{u.unit_id}', u.n_spikes,
                                     f'{u.mean_amplitude:.4f}', f'{u.snr:.1f}',
                                     f'{u.isi_violation_rate:.1f}%', status))

    def _on_sel(self, e=None):
        self.selected_units = [int(i) for i in self.tree.selection()]
        self._update_plots(self.results.get(self.current_channel))

    def _update_plots(self, result):
        self.fig.clear()
        if result is None or not result.units:
            self.canvas.draw()
            return

        gs = self.fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
        ax_wf = self.fig.add_subplot(gs[0, 0])
        ax_pca = self.fig.add_subplot(gs[0, 1])
        ax_pca2 = self.fig.add_subplot(gs[0, 2])
        ax_isi = self.fig.add_subplot(gs[1, 0])
        ax_acg = self.fig.add_subplot(gs[1, 1])
        ax_info = self.fig.add_subplot(gs[1, 2])

        time_ms = result.waveform_time_ms if result.waveform_time_ms is not None \
            else np.linspace(-0.5, 1.0, 60)

        for unit in result.units:
            alpha = 0.9 if (not self.selected_units or
                            unit.unit_id in self.selected_units) else 0.15
            c = 'gray' if unit.is_noise else unit.color
            if unit.is_noise:
                alpha = 0.1

            if len(unit.waveforms) > 0:
                m = np.mean(unit.waveforms, axis=0)
                s = np.std(unit.waveforms, axis=0)
                ax_wf.plot(time_ms, m, color=c, lw=1.5, alpha=alpha)
                ax_wf.fill_between(time_ms, m - s, m + s, color=c, alpha=alpha * 0.2)

            if len(unit.pca_features) > 0:
                ax_pca.scatter(unit.pca_features[:, 0], unit.pca_features[:, 1],
                               c=c, s=5, alpha=alpha * 0.5)
                if unit.pca_features.shape[1] > 2:
                    ax_pca2.scatter(unit.pca_features[:, 0], unit.pca_features[:, 2],
                                   c=c, s=5, alpha=alpha * 0.5)

            if not unit.is_noise and len(unit.spike_times) > 1:
                bins, hist = compute_isi_histogram(unit.spike_times)
                if len(bins) > 0:
                    ax_isi.bar(bins, hist, width=1.0, color=c, alpha=alpha * 0.6)
                bins_ac, acg = compute_autocorrelogram(unit.spike_times)
                if len(bins_ac) > 0:
                    ax_acg.bar(bins_ac, acg, width=1.0, color=c, alpha=alpha * 0.6)

        ax_wf.axhline(0, color='gray', ls='--', alpha=0.4)
        ax_wf.set_title('Waveforms', fontsize=9)
        ax_pca.set_title('PC1 vs PC2', fontsize=9)
        ax_pca2.set_title('PC1 vs PC3', fontsize=9)
        ax_isi.axvline(2, color='red', ls='--', alpha=0.5)
        ax_isi.set_xlim(0, 100)
        ax_isi.set_title('ISI', fontsize=9)
        ax_acg.set_title('Autocorrelogram', fontsize=9)

        ax_info.axis('off')
        info = f"Ch{self.current_channel}\nσ={result.sigma:.4f}\n\n"
        for u in result.units:
            if not u.is_noise:
                info += (f"U{u.unit_id}: n={u.n_spikes}, "
                         f"SNR={u.snr:.1f}, ISI={u.isi_violation_rate:.1f}%"
                         f"{' [MUA]' if u.is_mua else ''}\n")
        ax_info.text(0.05, 0.95, info, transform=ax_info.transAxes,
                     fontsize=8, va='top', fontfamily='monospace')

        self.fig.suptitle(f'Channel {self.current_channel}', fontsize=11)
        self.canvas.draw()

    # ============================
    # 操作
    # ============================
    def _merge(self):
        if len(self.selected_units) < 2:
            return
        r = self.results.get(self.current_channel)
        if r:
            self.results[self.current_channel] = merge_units(r, self.selected_units)
            self.selected_units = []
            self._update_display()

    def _delete(self):
        for uid in self.selected_units:
            delete_unit(self.results[self.current_channel], uid)
        self.selected_units = []
        self._update_display()

    def _undelete(self):
        for uid in self.selected_units:
            undelete_unit(self.results[self.current_channel], uid)
        self.selected_units = []
        self._update_display()

    def _mark_mua(self):
        for uid in self.selected_units:
            mark_as_mua(self.results[self.current_channel], uid)
        self._update_display()

    def _unmark_mua(self):
        for uid in self.selected_units:
            unmark_mua(self.results[self.current_channel], uid)
        self._update_display()

    def _recluster(self):
        try:
            n = int(self.nclust_var.get())
        except:
            return
        r = self.results.get(self.current_channel)
        if r:
            self.results[self.current_channel] = recluster(r, n)
            self.selected_units = []
            self._update_display()

    def _plot_grand_summary(self):
        from spike_plotting import plot_spike_grand_summary, plot_quality_table
        if self.results:
            plot_spike_grand_summary(self.results, self.output_dir, self.basename)
            plot_quality_table(self.results, self.output_dir, self.basename)

    def _save_npz(self):
        fp = filedialog.asksaveasfilename(defaultextension=".npz",
                                           initialfile=f"{self.basename}_spike_sorting.npz")
        if fp:
            save_sorting_results(self.results, fp)

    def _export_csv(self):
        fp = filedialog.asksaveasfilename(defaultextension=".csv",
                                           initialfile=f"{self.basename}_spike_times.csv")
        if fp:
            export_spike_times_csv(self.results, fp)

    def _finish(self):
        if messagebox.askyesno("確認", "スパイクソーティングを完了しますか?"):
            self.root.quit()
            self.root.destroy()
            if self.on_done:
                self.on_done(self.results)

    # ============================
    # 設定保存/読込
    # ============================
    def _save_cfg(self):
        d = {k: v.get() for k, v in self.cfg_vars.items()}
        d['_method'] = self.method_var.get()
        try:
            with open(SPIKE_CONFIG_FILE, 'w') as f:
                json.dump(d, f, indent=2)
        except:
            pass

    def _load_cfg(self):
        if not os.path.exists(SPIKE_CONFIG_FILE):
            return
        try:
            with open(SPIKE_CONFIG_FILE, 'r') as f:
                d = json.load(f)
            for k, v in d.items():
                if k in self.cfg_vars:
                    self.cfg_vars[k].set(v)
            # 手法の復元（KiloSort4は利用可能な場合のみ）
            if '_method' in d and d['_method'] == 'KiloSort4':
                try:
                    from kilosort_wrapper import is_kilosort_available
                    if is_kilosort_available():
                        self.method_var.set('KiloSort4')
                except Exception:
                    pass
        except:
            pass

    def run(self):
        self.root.mainloop()


def launch_spike_gui(wideband_data=None, fs=None, results=None,
                     output_dir="", basename="", on_done=None):
    gui = SpikeSortGUI(wideband_data=wideband_data, fs=fs, results=results,
                       output_dir=output_dir, basename=basename, on_done=on_done)
    gui.run()
