"""
step_phase.py - Step 3: 位相ロック解析パネル

phase_gui.py の設定+プレビューUIを統合GUI用Frameとして移植。
設定パネル(左) + 結果プレビュー(右: PPCヒートマップ + 極座標)。
"""
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import os
import threading
from typing import Dict, Any, List

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from main_gui import StepPanel


# 品質フィルタの表示ラベル - 内部値
QUALITY_FILTER_OPTIONS = {
    'Sortedユニットのみ': 'sorted_only',
    'Sorted + MUA（全ユニット）': 'all',
    'MUAのみ': 'mua',
}
QUALITY_FILTER_LABELS = list(QUALITY_FILTER_OPTIONS.keys())
QUALITY_FILTER_REVERSE = {v: k for k, v in QUALITY_FILTER_OPTIONS.items()}


class Panel(StepPanel):
    """位相ロック解析パネル"""

    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.vars = {}
        self.band_widgets = []
        self.phase_results = {}
        self.condition_results = {}
        self.fdr_significant = {}
        self.spike_data = None
        self._build_ui()

    # ===========================================================
    # UI構築
    # ===========================================================

    def _build_ui(self):
        pane = ttk.PanedWindow(self, orient='horizontal')
        pane.pack(fill='both', expand=True, padx=3, pady=3)

        left = ttk.Frame(pane, width=350)
        pane.add(left, weight=0)
        self._build_settings(left)

        right = ttk.Frame(pane)
        pane.add(right, weight=1)
        self._build_results(right)

    def _build_settings(self, parent):
        """設定パネル（左側、スクロール可能）"""
        canvas = tk.Canvas(parent)
        sb = ttk.Scrollbar(parent, orient='vertical', command=canvas.yview)
        sf = ttk.Frame(canvas)
        sf.bind('<Configure>',
                lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=sf, anchor='nw')
        canvas.configure(yscrollcommand=sb.set)

        def _wheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')
        canvas.bind('<MouseWheel>', _wheel)
        sf.bind('<MouseWheel>', _wheel)
        sb.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)

        row = 0

        # === スパイクデータソース ===
        ttk.Label(sf, text="スパイクデータ", font=('', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky='w', padx=5, pady=(10, 3))
        row += 1

        top = self.winfo_toplevel()
        default_src = 'sorted' if self.app.spike_results else 'plx'
        var_src = tk.StringVar(master=top, value=default_src)
        self.vars['spike_source'] = var_src
        ttk.Label(sf, text="ソース:").grid(
            row=row, column=0, sticky='w', padx=15)
        ttk.Combobox(sf, textvariable=var_src,
                     values=['plx', 'sorted'], state='readonly', width=10
                     ).grid(row=row, column=1, sticky='w')
        row += 1

        # 品質フィルタ
        self._qf_label_var = tk.StringVar(master=top, value=QUALITY_FILTER_LABELS[0])
        self.vars['spike_quality_filter'] = tk.StringVar(
            master=top, value='sorted_only')
        ttk.Label(sf, text="対象ユニット:").grid(
            row=row, column=0, sticky='w', padx=15)
        qf_combo = ttk.Combobox(sf, textvariable=self._qf_label_var,
                                values=QUALITY_FILTER_LABELS,
                                state='readonly', width=22)
        qf_combo.grid(row=row, column=1, sticky='w')
        qf_combo.bind('<<ComboboxSelected>>', self._on_qf_change)
        row += 1

        # === 周波数帯域（LFP configから動的生成）===
        ttk.Label(sf, text="周波数帯域", font=('', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky='w', padx=5, pady=(10, 3))
        row += 1

        bands = self._get_lfp_bands()
        self.band_widgets = []
        for band_name, (lo, hi) in bands.items():
            enabled_var = tk.BooleanVar(master=top, value=True)
            lo_var = tk.StringVar(master=top, value=str(lo))
            hi_var = tk.StringVar(master=top, value=str(hi))

            ttk.Checkbutton(sf, text=band_name, variable=enabled_var).grid(
                row=row, column=0, sticky='w', padx=15)
            freq_f = ttk.Frame(sf)
            freq_f.grid(row=row, column=1, sticky='w')
            ttk.Entry(freq_f, textvariable=lo_var, width=5).pack(side='left')
            ttk.Label(freq_f, text=" - ").pack(side='left')
            ttk.Entry(freq_f, textvariable=hi_var, width=5).pack(side='left')
            ttk.Label(freq_f, text=" Hz").pack(side='left')

            self.band_widgets.append({
                'name': band_name, 'enabled': enabled_var,
                'lo': lo_var, 'hi': hi_var,
            })
            row += 1

        # === 解析パラメータ ===
        ttk.Label(sf, text="パラメータ", font=('', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky='w', padx=5, pady=(10, 3))
        row += 1

        params = [
            ("min_spikes", "最小スパイク数", "50"),
            ("stim_artifact_window", "アーティファクト除外 (秒)", "0.005"),
            ("channel_spacing_um", "チャンネル間隔 (um)", "50.0"),
        ]
        for key, label, default in params:
            var = tk.StringVar(master=top, value=default)
            self.vars[key] = var
            ttk.Label(sf, text=label, font=('', 8)).grid(
                row=row, column=0, sticky='w', padx=15)
            ttk.Entry(sf, textvariable=var, width=8).grid(
                row=row, column=1, sticky='w')
            row += 1

        self.cond_var = tk.BooleanVar(master=top, value=True)
        ttk.Checkbutton(sf, text="条件別解析 (base/stim/post)",
                        variable=self.cond_var).grid(
            row=row, column=0, columnspan=2, sticky='w', padx=15, pady=3)
        row += 1

        # === 出力設定 ===
        ttk.Label(sf, text="出力", font=('', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky='w', padx=5, pady=(10, 3))
        row += 1

        self.save_plots_var = tk.BooleanVar(master=top, value=True)
        self.save_csv_var = tk.BooleanVar(master=top, value=True)
        ttk.Checkbutton(sf, text="プロット保存",
                        variable=self.save_plots_var).grid(
            row=row, column=0, columnspan=2, sticky='w', padx=15)
        row += 1
        ttk.Checkbutton(sf, text="CSV保存",
                        variable=self.save_csv_var).grid(
            row=row, column=0, columnspan=2, sticky='w', padx=15)
        row += 1

        # === 実行ボタン ===
        bf = ttk.Frame(sf)
        bf.grid(row=row, column=0, columnspan=2, pady=15)

        ttk.Button(bf, text="解析実行", command=self._run_analysis,
                   style='Run.TButton').pack(fill='x', pady=3)
        ttk.Button(bf, text="グランドサマリー",
                   command=self._plot_grand_summary).pack(fill='x', pady=3)
        ttk.Button(bf, text="完了",
                   command=self._finish).pack(fill='x', pady=3)

    def _build_results(self, parent):
        """結果プレビューパネル（右側）"""
        ch_bar = ttk.Frame(parent)
        ch_bar.pack(fill='x', padx=5, pady=3)
        ttk.Label(ch_bar, text="極座標 LFP Ch:").pack(side='left', padx=3)
        self.preview_ch_var = tk.StringVar(master=self.winfo_toplevel(), value='0')
        self.preview_ch_combo = ttk.Combobox(
            ch_bar, textvariable=self.preview_ch_var,
            state='readonly', width=10)
        self.preview_ch_combo.pack(side='left', padx=3)
        self.preview_ch_combo.bind('<<ComboboxSelected>>',
                                    self._on_preview_ch_change)

        self.result_fig = Figure(figsize=(8, 6), dpi=100)
        self.result_canvas = FigureCanvasTkAgg(self.result_fig, master=parent)
        self.result_canvas.get_tk_widget().pack(fill='both', expand=True)

        tb = ttk.Frame(parent)
        tb.pack(fill='x')
        NavigationToolbar2Tk(self.result_canvas, tb)

        self.result_status = tk.StringVar(master=self.winfo_toplevel(), value="解析未実行")
        ttk.Label(parent, textvariable=self.result_status,
                  relief='sunken').pack(fill='x', side='bottom')

    # ===========================================================
    # ヘルパー
    # ===========================================================

    def _get_lfp_bands(self) -> Dict[str, tuple]:
        """LFP configから帯域設定を取得"""
        lfp_cfg = self.state.lfp_config
        if lfp_cfg and 'bands' in lfp_cfg and lfp_cfg['bands']:
            bands = {}
            for name, freq in lfp_cfg['bands'].items():
                try:
                    if isinstance(freq, (list, tuple)) and len(freq) >= 2:
                        bands[name] = (float(freq[0]), float(freq[1]))
                except Exception:
                    pass
            if bands:
                return bands
        return {'theta': (4, 12), 'gamma': (30, 80)}

    def _get_bands(self) -> Dict[str, tuple]:
        """GUI上の帯域設定を辞書で返す（有効なもののみ）"""
        bands = {}
        for bw in self.band_widgets:
            if bw['enabled'].get():
                try:
                    lo = float(bw['lo'].get())
                    hi = float(bw['hi'].get())
                    bands[bw['name']] = (lo, hi)
                except ValueError:
                    pass
        return bands

    def _on_qf_change(self, e=None):
        label = self._qf_label_var.get()
        internal = QUALITY_FILTER_OPTIONS.get(label, 'sorted_only')
        self.vars['spike_quality_filter'].set(internal)

    def _on_preview_ch_change(self, e=None):
        self._show_preview()

    def _get_lfp_data(self):
        """LFP results から必要なデータを取得"""
        lr = self.app.lfp_results
        if lr is None:
            return None
        return {
            'lfp_cleaned': lr.get('lfp_cleaned'),
            'lfp_times': lr.get('lfp_times'),
            'fs': lr.get('fs'),
            'stim_times': lr.get('stim_times'),
            'original_ch_numbers': lr.get('original_ch_numbers', []),
        }

    def _get_condition_masks(self):
        """LFP results から条件マスクを取得"""
        lr = self.app.lfp_results
        if lr is None:
            return None
        if all(k in lr for k in ['clean_baseline', 'clean_stim',
                                  'clean_post']):
            return {
                'baseline': lr['clean_baseline'],
                'stim': lr['clean_stim'],
                'post': lr['clean_post']
            }
        return None

    def _convert_sorting_to_spike_data(self, quality_filter='all'):
        """スパイクソーティング結果を spike_processing 形式に変換"""
        from spike_processing import UnitInfo
        unit_info_list = []
        spike_times_dict = {}

        for ch, result in self.app.spike_results.items():
            for unit in result.units:
                if unit.is_noise:
                    continue
                if quality_filter == 'sorted_only' and unit.is_mua:
                    continue
                if quality_filter == 'mua' and not unit.is_mua:
                    continue
                key = f"ch{ch}_unit{unit.unit_id}"
                spike_times_dict[key] = unit.spike_times
                ui = UnitInfo(
                    channel=ch, unit_id=unit.unit_id,
                    n_spikes=unit.n_spikes,
                    t_start=(float(unit.spike_times[0])
                             if len(unit.spike_times) > 0 else 0),
                    t_stop=(float(unit.spike_times[-1])
                            if len(unit.spike_times) > 0 else 0),
                    unit_key=key)
                unit_info_list.append(ui)

        return {
            'spike_trains': [],
            'unit_info': unit_info_list,
            'spike_times': spike_times_dict
        }

    # ===========================================================
    # 解析実行
    # ===========================================================

    def _run_analysis(self):
        lfp_data = self._get_lfp_data()
        if lfp_data is None or lfp_data['lfp_cleaned'] is None:
            messagebox.showwarning(
                "Warning",
                "LFPデータがありません。先にStep 1を実行してください。")
            return

        bands = self._get_bands()
        if not bands:
            messagebox.showwarning("Warning",
                                   "帯域を1つ以上有効にしてください")
            return

        min_spikes = int(self.vars['min_spikes'].get())
        artifact_window = float(self.vars['stim_artifact_window'].get())
        spike_source = self.vars['spike_source'].get()
        quality_filter = self.vars['spike_quality_filter'].get()

        # スパイクデータ取得
        if spike_source == 'sorted' and self.app.spike_results is not None:
            self.spike_data = self._convert_sorting_to_spike_data(
                quality_filter)
        elif spike_source == 'plx' and self.app.plx_data is not None:
            from spike_processing import load_spike_data
            segment = self.app.plx_data.segment
            self.spike_data = load_spike_data(
                segment, quality_filter=quality_filter, verbose=True)
        else:
            messagebox.showwarning("Warning", "スパイクデータが利用不可")
            return

        if (not self.spike_data
                or len(self.spike_data['unit_info']) == 0):
            messagebox.showwarning("Warning",
                                   "スパイクデータが見つかりません")
            return

        self.result_status.set("解析中...")
        self.app.set_status("位相ロック解析実行中...")

        condition_masks = self._get_condition_masks()

        def _execute():
            try:
                from phase_locking import (
                    analyze_spike_lfp_coupling,
                    analyze_phase_locking_by_condition,
                    apply_fdr_to_phase_results)
                from spike_processing import exclude_stimulus_artifact

                lfp_cleaned = lfp_data['lfp_cleaned']
                lfp_times = lfp_data['lfp_times']
                fs = lfp_data['fs']
                stim_times = lfp_data['stim_times']

                phase_results = {}
                condition_results = {}

                for unit_info in self.spike_data['unit_info']:
                    unit_key = unit_info.unit_key
                    spike_times = self.spike_data['spike_times'][unit_key]
                    print(f"  ユニット: {unit_key} "
                          f"({unit_info.n_spikes} spikes)")

                    # アーティファクト除外
                    if artifact_window > 0 and stim_times is not None:
                        spike_times = exclude_stimulus_artifact(
                            spike_times, stim_times, artifact_window)

                    # 全体解析（全LFPチャンネル x 全帯域）
                    results = analyze_spike_lfp_coupling(
                        spike_times, lfp_cleaned, lfp_times, fs,
                        freq_bands=bands, min_spikes=min_spikes,
                        verbose=True)
                    phase_results[unit_key] = results

                    # 条件別解析
                    if self.cond_var.get() and condition_masks:
                        for band_name in bands:
                            cond = analyze_phase_locking_by_condition(
                                spike_times, lfp_cleaned, lfp_times, fs,
                                condition_masks,
                                freq_band=bands[band_name],
                                lfp_channel=0,
                                min_spikes=min_spikes // 2,
                                verbose=True)
                            if unit_key not in condition_results:
                                condition_results[unit_key] = {}
                            condition_results[unit_key][band_name] = cond

                # FDR補正
                fdr_significant, n_total, n_sig = \
                    apply_fdr_to_phase_results(phase_results)
                print(f"  FDR補正: {n_sig}/{n_total} 有意")

                self.phase_results = phase_results
                self.condition_results = condition_results
                self.fdr_significant = fdr_significant
                self.app.phase_results = phase_results

                self.after(0, lambda: self._on_analysis_done(
                    len(phase_results), n_sig, n_total))
            except Exception as e:
                self.after(0, lambda: self._on_analysis_error(str(e)))

        t = threading.Thread(target=_execute, daemon=True)
        t.start()

    def _on_analysis_done(self, n_units, n_sig, n_total):
        self.result_status.set(
            f"完了: {n_units} ユニット, FDR {n_sig}/{n_total} 有意")
        self.app.set_status(
            f"位相ロック解析完了 ({n_units} units, "
            f"FDR {n_sig}/{n_total})")
        self._show_preview()

    def _on_analysis_error(self, msg: str):
        self.result_status.set("エラー")
        messagebox.showerror("Error", f"位相ロック解析エラー:\n{msg}")
        self.app.set_status("位相ロック解析エラー")

    # ===========================================================
    # プレビュー
    # ===========================================================

    def _show_preview(self):
        """解析結果のプレビュー: 左=PPCヒートマップ, 右=全帯域極座標"""
        self.result_fig.clear()

        if not self.phase_results:
            self.result_canvas.draw()
            return

        lfp_data = self._get_lfp_data()
        if lfp_data is None:
            return

        bands = self._get_bands()
        band_names = list(bands.keys())
        n_bands = len(band_names)
        lfp_cleaned = lfp_data['lfp_cleaned']
        n_ch = lfp_cleaned.shape[1] if lfp_cleaned.ndim > 1 else 1
        original_ch = lfp_data['original_ch_numbers']

        ch_labels = ([f'Ch{c}' for c in original_ch[:n_ch]]
                     if original_ch
                     else [f'Ch{i}' for i in range(n_ch)])

        # チャンネルコンボ更新
        self.preview_ch_combo['values'] = ch_labels
        if self.preview_ch_var.get() not in ch_labels:
            self.preview_ch_var.set(ch_labels[0])
        sel_ch_idx = (ch_labels.index(self.preview_ch_var.get())
                      if self.preview_ch_var.get() in ch_labels else 0)

        # --- 全ユニット平均 PPC 行列 (LFP ch x band) ---
        ppc_mat = np.zeros((n_ch, n_bands))
        count_mat = np.zeros((n_ch, n_bands))
        fdr_any = np.zeros((n_ch, n_bands), dtype=bool)

        for unit_key, results in self.phase_results.items():
            for bi, band in enumerate(band_names):
                if band not in results:
                    continue
                for ch in range(n_ch):
                    r = results[band].get(ch)
                    if r is not None:
                        ppc_mat[ch, bi] += r.ppc
                        count_mat[ch, bi] += 1
                    if (self.fdr_significant
                            .get(unit_key, {})
                            .get(band, {})
                            .get(ch, False)):
                        fdr_any[ch, bi] = True

        with np.errstate(divide='ignore', invalid='ignore'):
            ppc_mean = np.where(count_mat > 0, ppc_mat / count_mat, 0)

        # レイアウト: 左=ヒートマップ, 右=帯域数分の極座標
        gs = self.result_fig.add_gridspec(
            1, 1 + n_bands, wspace=0.3,
            width_ratios=[1.5] + [1] * n_bands)

        # --- 左: PPC ヒートマップ ---
        ax_heat = self.result_fig.add_subplot(gs[0, 0])
        vmax = max(0.05, np.max(ppc_mean))
        im = ax_heat.imshow(ppc_mean, aspect='auto', cmap='YlGnBu',
                            vmin=0, vmax=vmax)
        ax_heat.set_yticks(range(n_ch))
        ax_heat.set_yticklabels(ch_labels, fontsize=6)
        ax_heat.set_xticks(range(n_bands))
        ax_heat.set_xticklabels(band_names, fontsize=7)
        ax_heat.set_ylabel('LFP Channel (depth)')
        ax_heat.set_xlabel('Frequency Band')
        ax_heat.set_title('PPC (all units avg)', fontsize=9)
        self.result_fig.colorbar(im, ax=ax_heat, label='PPC', shrink=0.7)

        # FDR有意性マーカー + 非有意グレーアウト
        for ch in range(n_ch):
            for bi in range(n_bands):
                if fdr_any[ch, bi]:
                    ax_heat.text(bi, ch, '*', ha='center', va='center',
                                fontsize=10, color='white',
                                fontweight='bold')
                else:
                    ax_heat.add_patch(matplotlib.patches.Rectangle(
                        (bi - 0.5, ch - 0.5), 1, 1,
                        fill=True, facecolor='gray', alpha=0.35,
                        edgecolor='none'))

        # 選択中のチャンネルを枠で強調
        for bi in range(n_bands):
            ax_heat.add_patch(matplotlib.patches.Rectangle(
                (bi - 0.5, sel_ch_idx - 0.5), 1, 1,
                fill=False, edgecolor='red', lw=2))

        # --- 右: 全帯域の極座標ヒストグラム（選択LFP ch）---
        cmap_band = matplotlib.colormaps.get_cmap('Set1')
        for bi, band in enumerate(band_names):
            ax = self.result_fig.add_subplot(
                gs[0, 1 + bi], projection='polar')

            # 全ユニットのspike_phasesを結合
            all_phases = []
            for unit_key, results in self.phase_results.items():
                if band in results:
                    r = results[band].get(sel_ch_idx)
                    if r is not None and len(r.spike_phases) > 0:
                        all_phases.append(r.spike_phases)

            if all_phases:
                phases = np.concatenate(all_phases)
                bins = np.linspace(-np.pi, np.pi, 37)
                counts, _ = np.histogram(phases, bins=bins)
                bw = 2 * np.pi / 36
                counts_norm = counts / (len(phases) * bw)
                bc = (bins[:-1] + bins[1:]) / 2

                c = cmap_band(bi / max(n_bands, 1))
                ax.bar(bc, counts_norm, width=bw * 0.9,
                       color=c, alpha=0.7, edgecolor='white', lw=0.3)

                # 平均ベクトル
                mean_vec = np.mean(np.exp(1j * phases))
                mrl = np.abs(mean_vec)
                pref = np.angle(mean_vec)
                mc = counts_norm.max() if counts_norm.max() > 0 else 1
                ax.annotate('', xy=(pref, mrl * mc * 1.3),
                            xytext=(0, 0),
                            arrowprops=dict(arrowstyle='->',
                                            color='red', lw=1.5))

                ax.set_title(f'{band}\nMRL={mrl:.3f} n={len(phases)}',
                             fontsize=7, pad=8)
            else:
                ax.set_title(f'{band}\n(no data)', fontsize=7, pad=8)

            ax.tick_params(labelsize=4)
            ax.set_rticks([])

        n_units = len(self.phase_results)
        self.result_fig.suptitle(
            f'PPC Heatmap + Phase Distribution @ {ch_labels[sel_ch_idx]} '
            f'({n_units} units pooled)', fontsize=10)
        self.result_canvas.draw()

    # ===========================================================
    # グランドサマリー
    # ===========================================================

    def _plot_grand_summary(self):
        if not self.phase_results:
            messagebox.showwarning("Warning", "先に解析を実行してください")
            return

        lfp_data = self._get_lfp_data()
        if lfp_data is None:
            return

        channel_spacing = float(self.vars['channel_spacing_um'].get())

        from phase_plotting_v6 import plot_phase_grand_summary
        plot_phase_grand_summary(
            phase_results=self.phase_results,
            condition_results=self.condition_results,
            bands=self._get_bands(),
            original_ch_numbers=lfp_data['original_ch_numbers'],
            spike_data=self.spike_data,
            output_dir=self.state.output_dir,
            basename=self.state.basename,
            lfp_cleaned=lfp_data['lfp_cleaned'],
            lfp_times=lfp_data['lfp_times'],
            fs=lfp_data['fs'],
            stim_times=lfp_data['stim_times'],
            channel_spacing_um=channel_spacing,
            fdr_significant=self.fdr_significant,
            show=True,
            save=self.save_plots_var.get()
        )

        if self.save_csv_var.get() and self.spike_data:
            from phase_plotting import save_phase_locking_csv
            bands = self._get_bands()
            band_list = list(bands.keys())
            save_phase_locking_csv(
                self.phase_results,
                self.spike_data['unit_info'],
                {uk: cr.get(band_list[0], {})
                 for uk, cr in self.condition_results.items()}
                if band_list else {},
                self.state.output_dir, self.state.basename)

    def _finish(self):
        """解析完了ボタン"""
        if not self.phase_results:
            messagebox.showwarning("Warning", "先に解析を実行してください")
            return
        self.app.phase_results = self.phase_results
        self.app.complete_step("phase")
        self.app.set_status("位相ロック解析完了")

    # ===========================================================
    # 設定取得 / 反映
    # ===========================================================

    def get_config(self) -> dict:
        d = {}
        for k, v in self.vars.items():
            d[k] = v.get()
        d['condition_analysis'] = self.cond_var.get()
        band_cfg = {}
        for bw in self.band_widgets:
            band_cfg[bw['name']] = {
                'enabled': bw['enabled'].get(),
                'lo': bw['lo'].get(),
                'hi': bw['hi'].get(),
            }
        d['bands'] = band_cfg
        return d

    def set_config(self, config: dict):
        if not config:
            return
        for k, v in config.items():
            if k in self.vars:
                if k == 'spike_source' and self.app.spike_results:
                    continue  # sorted がある場合はデフォルト優先
                self.vars[k].set(v)
        # 品質フィルタ: 内部値 -> ラベル
        qf_val = self.vars.get('spike_quality_filter')
        if qf_val:
            label = QUALITY_FILTER_REVERSE.get(
                qf_val.get(), QUALITY_FILTER_LABELS[0])
            self._qf_label_var.set(label)
        # 帯域設定
        if 'bands' in config:
            for bw in self.band_widgets:
                if bw['name'] in config['bands']:
                    bc = config['bands'][bw['name']]
                    bw['enabled'].set(bc.get('enabled', True))
                    bw['lo'].set(bc.get('lo', bw['lo'].get()))
                    bw['hi'].set(bc.get('hi', bw['hi'].get()))
        if 'condition_analysis' in config:
            self.cond_var.set(config['condition_analysis'])

    # ===========================================================
    # ライフサイクル
    # ===========================================================

    def on_show(self):
        if self.state.phase_config:
            self.set_config(self.state.phase_config)
        # 既存結果の復元
        if self.app.phase_results and not self.phase_results:
            self.phase_results = self.app.phase_results
            self._show_preview()
