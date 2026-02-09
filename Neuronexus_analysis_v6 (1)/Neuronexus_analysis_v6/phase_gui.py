"""
phase_gui.py - 位相ロック解析GUI

LFPデータとスパイクデータを統合し、位相ロック解析を対話的に実行。
設定の調整 → 解析実行 → 結果プレビュー → プロット保存まで一貫操作。
"""
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import json
import os
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from typing import Dict, Optional, Any
from dataclasses import dataclass, field


PHASE_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "phase_config.json")


@dataclass
class PhaseConfig:
    """位相ロック解析の設定"""
    # 周波数帯域
    bands: Dict[str, tuple] = field(default_factory=lambda: {
        'theta': (4, 12),
        'gamma': (30, 80)
    })

    # 解析パラメータ
    min_spikes: int = 50
    lfp_channel: int = 0
    condition_analysis: bool = True
    stim_artifact_window: float = 0.005

    # スパイクソース
    spike_source: str = 'plx'  # 'plx' or 'sorted'
    spike_quality_filter: str = 'sorted_only'

    # 出力
    save_plots: bool = True
    save_csv: bool = True
    show_plots: bool = True


class PhaseGUI:
    """位相ロック解析GUI"""

    def __init__(self, lfp_cleaned=None, lfp_times=None, fs=None,
                 segment=None, spike_results=None,
                 stim_times=None, condition_masks=None,
                 original_ch_numbers=None,
                 output_dir="", basename="",
                 on_done=None):
        """
        Parameters
        ----------
        lfp_cleaned : ndarray (n_samples, n_channels)
        lfp_times : ndarray
        fs : int
        segment : neo.Segment (PLXスパイク用)
        spike_results : dict (スパイクソーティング結果)
        stim_times : ndarray
        condition_masks : dict {'baseline': mask, 'stim': mask, 'post': mask}
        original_ch_numbers : list
        output_dir, basename : str
        on_done : callable
        """
        self.lfp_cleaned = lfp_cleaned
        self.lfp_times = lfp_times
        self.fs = fs
        self.segment = segment
        self.spike_results = spike_results
        self.stim_times = stim_times
        self.condition_masks = condition_masks
        self.original_ch_numbers = original_ch_numbers or []
        self.output_dir = output_dir
        self.basename = basename
        self.on_done = on_done

        # 解析結果
        self.phase_results = {}
        self.condition_results = {}
        self.spike_data = None

        self.root = tk.Tk()
        self.root.title("Phase Locking Analysis GUI")
        self.root.geometry("1200x800")
        self.vars = {}
        self._build_gui()
        self._load_config()

    def _build_gui(self):
        # 左: 設定
        pane = ttk.PanedWindow(self.root, orient='horizontal')
        pane.pack(fill='both', expand=True, padx=3, pady=3)

        left = ttk.Frame(pane, width=320)
        pane.add(left, weight=0)
        self._build_settings(left)

        right = ttk.Frame(pane)
        pane.add(right, weight=1)
        self._build_results(right)

    def _build_settings(self, parent):
        """設定パネル"""
        canvas = tk.Canvas(parent)
        sb = ttk.Scrollbar(parent, orient='vertical', command=canvas.yview)
        sf = ttk.Frame(canvas)
        sf.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=sf, anchor='nw')
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)

        row = 0

        # --- スパイクソース ---
        ttk.Label(sf, text="⚡ スパイクデータ", font=('', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky='w', padx=5, pady=(10, 3))
        row += 1

        var_src = tk.StringVar(value='plx')
        self.vars['spike_source'] = var_src
        ttk.Label(sf, text="ソース:").grid(row=row, column=0, sticky='w', padx=15)
        src_combo = ttk.Combobox(sf, textvariable=var_src,
                                  values=['plx', 'sorted'], state='readonly', width=10)
        src_combo.grid(row=row, column=1, sticky='w')
        row += 1

        var_qf = tk.StringVar(value='sorted_only')
        self.vars['spike_quality_filter'] = var_qf
        ttk.Label(sf, text="品質フィルタ:").grid(row=row, column=0, sticky='w', padx=15)
        ttk.Combobox(sf, textvariable=var_qf,
                     values=['all', 'sorted_only', 'mua'],
                     state='readonly', width=10).grid(row=row, column=1, sticky='w')
        row += 1

        # --- 位相ロック帯域 ---
        ttk.Label(sf, text="📊 周波数帯域", font=('', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky='w', padx=5, pady=(10, 3))
        row += 1

        # Theta
        self.theta_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(sf, text="Theta", variable=self.theta_enabled).grid(
            row=row, column=0, sticky='w', padx=15)
        theta_f = ttk.Frame(sf)
        theta_f.grid(row=row, column=1, sticky='w')
        self.theta_lo = tk.StringVar(value="4")
        self.theta_hi = tk.StringVar(value="12")
        ttk.Entry(theta_f, textvariable=self.theta_lo, width=4).pack(side='left')
        ttk.Label(theta_f, text="-").pack(side='left')
        ttk.Entry(theta_f, textvariable=self.theta_hi, width=4).pack(side='left')
        ttk.Label(theta_f, text="Hz").pack(side='left')
        row += 1

        # Gamma
        self.gamma_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(sf, text="Gamma", variable=self.gamma_enabled).grid(
            row=row, column=0, sticky='w', padx=15)
        gamma_f = ttk.Frame(sf)
        gamma_f.grid(row=row, column=1, sticky='w')
        self.gamma_lo = tk.StringVar(value="30")
        self.gamma_hi = tk.StringVar(value="80")
        ttk.Entry(gamma_f, textvariable=self.gamma_lo, width=4).pack(side='left')
        ttk.Label(gamma_f, text="-").pack(side='left')
        ttk.Entry(gamma_f, textvariable=self.gamma_hi, width=4).pack(side='left')
        ttk.Label(gamma_f, text="Hz").pack(side='left')
        row += 1

        # Beta (optional)
        self.beta_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(sf, text="Beta", variable=self.beta_enabled).grid(
            row=row, column=0, sticky='w', padx=15)
        beta_f = ttk.Frame(sf)
        beta_f.grid(row=row, column=1, sticky='w')
        self.beta_lo = tk.StringVar(value="14")
        self.beta_hi = tk.StringVar(value="30")
        ttk.Entry(beta_f, textvariable=self.beta_lo, width=4).pack(side='left')
        ttk.Label(beta_f, text="-").pack(side='left')
        ttk.Entry(beta_f, textvariable=self.beta_hi, width=4).pack(side='left')
        ttk.Label(beta_f, text="Hz").pack(side='left')
        row += 1

        # --- 解析パラメータ ---
        ttk.Label(sf, text="🔧 パラメータ", font=('', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky='w', padx=5, pady=(10, 3))
        row += 1

        params = [
            ("min_spikes", "最小スパイク数", "50"),
            ("lfp_channel", "参照LFPチャンネル", "0"),
            ("stim_artifact_window", "アーティファクト除外 (秒)", "0.005"),
        ]
        for key, label, default in params:
            var = tk.StringVar(value=default)
            self.vars[key] = var
            ttk.Label(sf, text=label, font=('', 8)).grid(row=row, column=0, sticky='w', padx=15)
            ttk.Entry(sf, textvariable=var, width=8).grid(row=row, column=1, sticky='w')
            row += 1

        self.cond_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(sf, text="条件別解析 (base/stim/post)",
                        variable=self.cond_var).grid(
            row=row, column=0, columnspan=2, sticky='w', padx=15, pady=3)
        row += 1

        # --- 出力設定 ---
        ttk.Label(sf, text="💾 出力", font=('', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky='w', padx=5, pady=(10, 3))
        row += 1

        self.save_plots_var = tk.BooleanVar(value=True)
        self.save_csv_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(sf, text="プロット保存", variable=self.save_plots_var).grid(
            row=row, column=0, columnspan=2, sticky='w', padx=15)
        row += 1
        ttk.Checkbutton(sf, text="CSV保存", variable=self.save_csv_var).grid(
            row=row, column=0, columnspan=2, sticky='w', padx=15)
        row += 1

        # --- 実行ボタン ---
        bf = ttk.Frame(sf)
        bf.grid(row=row, column=0, columnspan=2, pady=15)

        style = ttk.Style()
        style.configure("Run.TButton", font=("", 12, "bold"), padding=6)
        ttk.Button(bf, text="▶ 解析実行", command=self._run_analysis,
                   style="Run.TButton").pack(fill='x', pady=3)
        ttk.Button(bf, text="📊 グランドサマリー", command=self._plot_grand_summary).pack(fill='x', pady=3)
        ttk.Button(bf, text="💾 設定保存", command=self._save_config).pack(fill='x', pady=3)
        ttk.Button(bf, text="✅ 完了", command=self._finish).pack(fill='x', pady=3)

    def _build_results(self, parent):
        """結果プレビューパネル"""
        self.result_fig = Figure(figsize=(8, 6), dpi=100)
        self.result_canvas = FigureCanvasTkAgg(self.result_fig, master=parent)
        self.result_canvas.get_tk_widget().pack(fill='both', expand=True)

        tb = ttk.Frame(parent)
        tb.pack(fill='x')
        NavigationToolbar2Tk(self.result_canvas, tb)

        self.result_status = tk.StringVar(value="解析未実行")
        ttk.Label(parent, textvariable=self.result_status, relief='sunken').pack(
            fill='x', side='bottom')

    # ============================
    # 帯域取得
    # ============================
    def _get_bands(self):
        bands = {}
        if self.theta_enabled.get():
            try:
                bands['theta'] = (float(self.theta_lo.get()), float(self.theta_hi.get()))
            except:
                bands['theta'] = (4, 12)
        if self.gamma_enabled.get():
            try:
                bands['gamma'] = (float(self.gamma_lo.get()), float(self.gamma_hi.get()))
            except:
                bands['gamma'] = (30, 80)
        if self.beta_enabled.get():
            try:
                bands['beta'] = (float(self.beta_lo.get()), float(self.beta_hi.get()))
            except:
                bands['beta'] = (14, 30)
        return bands

    # ============================
    # 解析実行
    # ============================
    def _run_analysis(self):
        if self.lfp_cleaned is None:
            messagebox.showwarning("Warning", "LFPデータがありません")
            return

        bands = self._get_bands()
        if not bands:
            messagebox.showwarning("Warning", "帯域を1つ以上選択してください")
            return

        min_spikes = int(self.vars['min_spikes'].get())
        lfp_ch = int(self.vars['lfp_channel'].get())
        artifact_window = float(self.vars['stim_artifact_window'].get())
        spike_source = self.vars['spike_source'].get()
        quality_filter = self.vars['spike_quality_filter'].get()

        self.result_status.set("解析中...")
        self.root.update()

        # スパイクデータ取得
        if spike_source == 'plx' and self.segment is not None:
            from spike_processing import load_spike_data, exclude_stimulus_artifact
            self.spike_data = load_spike_data(
                self.segment, quality_filter=quality_filter, verbose=True)
        elif spike_source == 'sorted' and self.spike_results is not None:
            self.spike_data = self._convert_sorting_to_spike_data()
        else:
            messagebox.showwarning("Warning", "スパイクデータが利用不可")
            return

        if not self.spike_data or len(self.spike_data['unit_info']) == 0:
            self.result_status.set("スパイクデータなし")
            return

        # 位相ロック解析
        from phase_locking import (analyze_spike_lfp_coupling,
                                    analyze_phase_locking_by_condition)
        from spike_processing import exclude_stimulus_artifact

        self.phase_results = {}
        self.condition_results = {}

        for unit_info in self.spike_data['unit_info']:
            unit_key = unit_info.unit_key
            spike_times = self.spike_data['spike_times'][unit_key]
            print(f"  ユニット: {unit_key} ({unit_info.n_spikes} spikes)")

            # アーティファクト除外
            if artifact_window > 0 and self.stim_times is not None:
                spike_times = exclude_stimulus_artifact(
                    spike_times, self.stim_times, artifact_window)

            # 全体解析
            results = analyze_spike_lfp_coupling(
                spike_times, self.lfp_cleaned, self.lfp_times, self.fs,
                freq_bands=bands, min_spikes=min_spikes, verbose=True)
            self.phase_results[unit_key] = results

            # 条件別解析
            if self.cond_var.get() and self.condition_masks:
                for band_name in bands:
                    cond = analyze_phase_locking_by_condition(
                        spike_times, self.lfp_cleaned, self.lfp_times, self.fs,
                        self.condition_masks,
                        freq_band=bands[band_name],
                        lfp_channel=lfp_ch,
                        min_spikes=min_spikes // 2,
                        verbose=True)
                    if unit_key not in self.condition_results:
                        self.condition_results[unit_key] = {}
                    self.condition_results[unit_key][band_name] = cond

        n_units = len(self.phase_results)
        self.result_status.set(f"完了: {n_units} ユニット解析")

        # プレビュー表示
        self._show_preview()

    def _convert_sorting_to_spike_data(self):
        """スパイクソーティング結果を spike_processing 形式に変換"""
        from spike_processing import UnitInfo
        unit_info_list = []
        spike_times_dict = {}

        for ch, result in self.spike_results.items():
            for unit in result.units:
                if unit.is_noise:
                    continue
                key = f"ch{ch}_unit{unit.unit_id}"
                spike_times_dict[key] = unit.spike_times
                ui = UnitInfo(
                    channel=ch, unit_id=unit.unit_id,
                    n_spikes=unit.n_spikes,
                    t_start=float(unit.spike_times[0]) if len(unit.spike_times) > 0 else 0,
                    t_stop=float(unit.spike_times[-1]) if len(unit.spike_times) > 0 else 0,
                    unit_key=key)
                unit_info_list.append(ui)

        return {
            'spike_trains': [],
            'unit_info': unit_info_list,
            'spike_times': spike_times_dict
        }

    def _show_preview(self):
        """解析結果のプレビュー (最初のユニットのサマリー)"""
        self.result_fig.clear()

        if not self.phase_results:
            self.result_canvas.draw()
            return

        first_key = list(self.phase_results.keys())[0]
        results = self.phase_results[first_key]
        bands = self._get_bands()
        band_names = list(bands.keys())
        n_bands = len(band_names)
        n_ch = self.lfp_cleaned.shape[1] if self.lfp_cleaned.ndim > 1 else 1

        # MRLヒートマップ
        ax = self.result_fig.add_subplot(111)
        mrl_mat = np.zeros((n_bands, n_ch))
        for i, band in enumerate(band_names):
            if band in results:
                for j in range(n_ch):
                    if j in results[band] and results[band][j] is not None:
                        mrl_mat[i, j] = results[band][j].mrl

        ch_labels = [f'Ch{c}' for c in self.original_ch_numbers[:n_ch]]
        im = ax.imshow(mrl_mat, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.5)
        ax.set_xticks(range(n_ch))
        ax.set_xticklabels(ch_labels, rotation=45, fontsize=7)
        ax.set_yticks(range(n_bands))
        ax.set_yticklabels(band_names)
        ax.set_title(f'{first_key} - MRL Heatmap', fontsize=10)
        self.result_fig.colorbar(im, ax=ax, label='MRL')

        # 有意性マーカー
        for i, band in enumerate(band_names):
            if band in results:
                for j in range(n_ch):
                    if j in results[band] and results[band][j] is not None:
                        r = results[band][j]
                        if r.p_value < 0.01:
                            ax.text(j, i, '**', ha='center', va='center',
                                    fontsize=12, color='white', fontweight='bold')
                        elif r.p_value < 0.05:
                            ax.text(j, i, '*', ha='center', va='center',
                                    fontsize=14, color='white', fontweight='bold')

        self.result_fig.tight_layout()
        self.result_canvas.draw()

    # ============================
    # グランドサマリー
    # ============================
    def _plot_grand_summary(self):
        if not self.phase_results:
            messagebox.showwarning("Warning", "先に解析を実行してください")
            return

        from phase_plotting_v6 import plot_phase_grand_summary
        plot_phase_grand_summary(
            self.phase_results,
            self.condition_results,
            self._get_bands(),
            self.original_ch_numbers,
            self.spike_data,
            self.output_dir,
            self.basename,
            show=True,
            save=self.save_plots_var.get()
        )

        # CSV保存
        if self.save_csv_var.get() and self.spike_data:
            from phase_plotting import save_phase_locking_csv
            save_phase_locking_csv(
                self.phase_results,
                self.spike_data['unit_info'],
                {uk: cr.get('theta', {}) for uk, cr in self.condition_results.items()},
                self.output_dir, self.basename)

    def _finish(self):
        self.root.destroy()
        if self.on_done:
            self.on_done(self.phase_results, self.condition_results)

    # ============================
    # 設定保存/読込
    # ============================
    def _save_config(self):
        d = {k: v.get() for k, v in self.vars.items()}
        d['theta_enabled'] = self.theta_enabled.get()
        d['theta_lo'] = self.theta_lo.get()
        d['theta_hi'] = self.theta_hi.get()
        d['gamma_enabled'] = self.gamma_enabled.get()
        d['gamma_lo'] = self.gamma_lo.get()
        d['gamma_hi'] = self.gamma_hi.get()
        d['beta_enabled'] = self.beta_enabled.get()
        d['beta_lo'] = self.beta_lo.get()
        d['beta_hi'] = self.beta_hi.get()
        d['condition_analysis'] = self.cond_var.get()
        try:
            with open(PHASE_CONFIG_FILE, 'w') as f:
                json.dump(d, f, indent=2)
        except:
            pass

    def _load_config(self):
        if not os.path.exists(PHASE_CONFIG_FILE):
            return
        try:
            with open(PHASE_CONFIG_FILE, 'r') as f:
                d = json.load(f)
            for k, v in d.items():
                if k in self.vars:
                    self.vars[k].set(v)
            if 'theta_enabled' in d:
                self.theta_enabled.set(d['theta_enabled'])
            if 'theta_lo' in d:
                self.theta_lo.set(d['theta_lo'])
            if 'theta_hi' in d:
                self.theta_hi.set(d['theta_hi'])
            if 'gamma_enabled' in d:
                self.gamma_enabled.set(d['gamma_enabled'])
            if 'gamma_lo' in d:
                self.gamma_lo.set(d['gamma_lo'])
            if 'gamma_hi' in d:
                self.gamma_hi.set(d['gamma_hi'])
            if 'beta_enabled' in d:
                self.beta_enabled.set(d['beta_enabled'])
            if 'beta_lo' in d:
                self.beta_lo.set(d['beta_lo'])
            if 'beta_hi' in d:
                self.beta_hi.set(d['beta_hi'])
            if 'condition_analysis' in d:
                self.cond_var.set(d['condition_analysis'])
        except:
            pass

    def run(self):
        self.root.mainloop()


def launch_phase_gui(**kwargs):
    gui = PhaseGUI(**kwargs)
    gui.run()
