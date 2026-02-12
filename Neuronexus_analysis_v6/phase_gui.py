"""
phase_gui.py - 位相ロック解析GUI

LFPデータとスパイクデータを統合し、位相ロック解析を対話的に実行。
設定の調整 → 解析実行 → 結果プレビュー → プロット保存まで一貫操作。
帯域設定は lfp_config.json から自動取得。
"""
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import json
import os
import matplotlib
if matplotlib.get_backend() == '' or matplotlib.get_backend().lower() == 'agg':
    matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field


LFP_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "lfp_config.json")
PHASE_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "phase_config.json")

# 品質フィルタの表示ラベル → 内部値
QUALITY_FILTER_OPTIONS = {
    'Sortedユニットのみ': 'sorted_only',
    'Sorted + MUA（全ユニット）': 'all',
    'MUAのみ': 'mua',
}
QUALITY_FILTER_LABELS = list(QUALITY_FILTER_OPTIONS.keys())
QUALITY_FILTER_REVERSE = {v: k for k, v in QUALITY_FILTER_OPTIONS.items()}


def _load_lfp_bands() -> Dict[str, tuple]:
    """lfp_config.json から帯域設定を読み込む"""
    if not os.path.exists(LFP_CONFIG_FILE):
        return {'theta': (4, 12), 'gamma': (30, 80)}
    try:
        with open(LFP_CONFIG_FILE, 'r') as f:
            cfg = json.load(f)
        bands = cfg.get('bands', {})
        return {name: tuple(freq) for name, freq in bands.items()}
    except Exception:
        return {'theta': (4, 12), 'gamma': (30, 80)}


def _load_channel_spacing() -> float:
    """lfp_config.json または spike_sort_config から channel_spacing を取得"""
    for path in [LFP_CONFIG_FILE,
                 os.path.join(os.path.dirname(__file__), "spike_sort_config.json")]:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    cfg = json.load(f)
                if 'channel_spacing_um' in cfg:
                    return float(cfg['channel_spacing_um'])
            except Exception:
                pass
    return 50.0


class PhaseGUI:
    """位相ロック解析GUI"""

    def __init__(self, lfp_cleaned=None, lfp_times=None, fs=None,
                 segment=None, spike_results=None,
                 stim_times=None, condition_masks=None,
                 original_ch_numbers=None,
                 output_dir="", basename="",
                 on_done=None):
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

        # spike_resultsがあればsortedをデフォルトに
        self._default_spike_source = 'sorted' if spike_results else 'plx'

        # LFP configから帯域を取得
        self._lfp_bands = _load_lfp_bands()
        self.channel_spacing_um = _load_channel_spacing()

        self.root = tk.Tk()
        self.root.title("Phase Locking Analysis GUI")
        self.root.geometry("1200x800")
        self.vars = {}

        # 帯域ウィジェット（動的生成）
        self.band_widgets = []  # [{name, enabled_var, lo_var, hi_var}, ...]

        self._build_gui()
        self._load_config()

    def _build_gui(self):
        pane = ttk.PanedWindow(self.root, orient='horizontal')
        pane.pack(fill='both', expand=True, padx=3, pady=3)

        left = ttk.Frame(pane, width=350)
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

        # === スパイクデータ ===
        ttk.Label(sf, text="スパイクデータ", font=('', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky='w', padx=5, pady=(10, 3))
        row += 1

        var_src = tk.StringVar(master=self.root, value=self._default_spike_source)
        self.vars['spike_source'] = var_src
        ttk.Label(sf, text="ソース:").grid(row=row, column=0, sticky='w', padx=15)
        ttk.Combobox(sf, textvariable=var_src,
                     values=['plx', 'sorted'], state='readonly', width=10
                     ).grid(row=row, column=1, sticky='w')
        row += 1

        # 品質フィルタ（日本語ラベル）
        self._qf_label_var = tk.StringVar(master=self.root,
                                           value=QUALITY_FILTER_LABELS[0])
        self.vars['spike_quality_filter'] = tk.StringVar(
            master=self.root, value='sorted_only')
        ttk.Label(sf, text="対象ユニット:").grid(row=row, column=0, sticky='w', padx=15)
        qf_combo = ttk.Combobox(sf, textvariable=self._qf_label_var,
                                values=QUALITY_FILTER_LABELS,
                                state='readonly', width=22)
        qf_combo.grid(row=row, column=1, sticky='w')
        qf_combo.bind('<<ComboboxSelected>>', self._on_qf_change)
        row += 1

        # === 周波数帯域（LFP configから動的生成）===
        ttk.Label(sf, text="周波数帯域 (LFP設定から取得)",
                  font=('', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky='w', padx=5, pady=(10, 3))
        row += 1

        self.band_widgets = []
        for band_name, (lo, hi) in self._lfp_bands.items():
            enabled_var = tk.BooleanVar(master=self.root, value=True)
            lo_var = tk.StringVar(master=self.root, value=str(lo))
            hi_var = tk.StringVar(master=self.root, value=str(hi))

            ttk.Checkbutton(sf, text=band_name, variable=enabled_var).grid(
                row=row, column=0, sticky='w', padx=15)
            freq_f = ttk.Frame(sf)
            freq_f.grid(row=row, column=1, sticky='w')
            ttk.Entry(freq_f, textvariable=lo_var, width=5).pack(side='left')
            ttk.Label(freq_f, text=" - ").pack(side='left')
            ttk.Entry(freq_f, textvariable=hi_var, width=5).pack(side='left')
            ttk.Label(freq_f, text=" Hz").pack(side='left')

            self.band_widgets.append({
                'name': band_name,
                'enabled': enabled_var,
                'lo': lo_var,
                'hi': hi_var,
            })
            row += 1

        # === 解析パラメータ ===
        ttk.Label(sf, text="パラメータ", font=('', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky='w', padx=5, pady=(10, 3))
        row += 1

        params = [
            ("min_spikes", "最小スパイク数", "50"),
            ("stim_artifact_window", "アーティファクト除外 (秒)", "0.005"),
            ("channel_spacing_um", "チャンネル間隔 (um)", str(self.channel_spacing_um)),
        ]
        for key, label, default in params:
            var = tk.StringVar(master=self.root, value=default)
            self.vars[key] = var
            ttk.Label(sf, text=label, font=('', 8)).grid(
                row=row, column=0, sticky='w', padx=15)
            ttk.Entry(sf, textvariable=var, width=8).grid(
                row=row, column=1, sticky='w')
            row += 1

        self.cond_var = tk.BooleanVar(master=self.root, value=True)
        ttk.Checkbutton(sf, text="条件別解析 (base/stim/post)",
                        variable=self.cond_var).grid(
            row=row, column=0, columnspan=2, sticky='w', padx=15, pady=3)
        row += 1

        # === 出力設定 ===
        ttk.Label(sf, text="出力", font=('', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky='w', padx=5, pady=(10, 3))
        row += 1

        self.save_plots_var = tk.BooleanVar(master=self.root, value=True)
        self.save_csv_var = tk.BooleanVar(master=self.root, value=True)
        ttk.Checkbutton(sf, text="プロット保存", variable=self.save_plots_var).grid(
            row=row, column=0, columnspan=2, sticky='w', padx=15)
        row += 1
        ttk.Checkbutton(sf, text="CSV保存", variable=self.save_csv_var).grid(
            row=row, column=0, columnspan=2, sticky='w', padx=15)
        row += 1

        # === 実行ボタン ===
        bf = ttk.Frame(sf)
        bf.grid(row=row, column=0, columnspan=2, pady=15)

        style = ttk.Style()
        style.configure("Run.TButton", font=("", 12, "bold"), padding=6)
        ttk.Button(bf, text="解析実行", command=self._run_analysis,
                   style="Run.TButton").pack(fill='x', pady=3)
        ttk.Button(bf, text="グランドサマリー",
                   command=self._plot_grand_summary).pack(fill='x', pady=3)
        ttk.Button(bf, text="設定保存", command=self._save_config).pack(fill='x', pady=3)
        ttk.Button(bf, text="完了", command=self._finish).pack(fill='x', pady=3)

    def _build_results(self, parent):
        """結果プレビューパネル"""
        # LFPチャンネル選択バー（極座標表示用）
        ch_bar = ttk.Frame(parent)
        ch_bar.pack(fill='x', padx=5, pady=3)
        ttk.Label(ch_bar, text="極座標 LFP Ch:").pack(side='left', padx=3)
        self.preview_ch_var = tk.StringVar(master=self.root, value='0')
        self.preview_ch_combo = ttk.Combobox(
            ch_bar, textvariable=self.preview_ch_var,
            state='readonly', width=10)
        self.preview_ch_combo.pack(side='left', padx=3)
        self.preview_ch_combo.bind('<<ComboboxSelected>>', self._on_preview_ch_change)

        self.result_fig = Figure(figsize=(8, 6), dpi=100)
        self.result_canvas = FigureCanvasTkAgg(self.result_fig, master=parent)
        self.result_canvas.get_tk_widget().pack(fill='both', expand=True)

        tb = ttk.Frame(parent)
        tb.pack(fill='x')
        NavigationToolbar2Tk(self.result_canvas, tb)

        self.result_status = tk.StringVar(master=self.root, value="解析未実行")
        ttk.Label(parent, textvariable=self.result_status, relief='sunken').pack(
            fill='x', side='bottom')

    # ============================
    # コールバック
    # ============================

    def _on_qf_change(self, e=None):
        """品質フィルタ変更時: ラベル → 内部値"""
        label = self._qf_label_var.get()
        internal = QUALITY_FILTER_OPTIONS.get(label, 'sorted_only')
        self.vars['spike_quality_filter'].set(internal)

    def _on_preview_ch_change(self, e=None):
        self._show_preview()

    # ============================
    # 帯域取得
    # ============================

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

    # ============================
    # 解析実行
    # ============================

    def _run_analysis(self):
        if self.lfp_cleaned is None:
            messagebox.showwarning("Warning", "LFPデータがありません")
            return

        bands = self._get_bands()
        if not bands:
            messagebox.showwarning("Warning", "帯域を1つ以上有効にしてください")
            return

        min_spikes = int(self.vars['min_spikes'].get())
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
            self.spike_data = self._convert_sorting_to_spike_data(quality_filter)
        else:
            messagebox.showwarning("Warning", "スパイクデータが利用不可")
            return

        if not self.spike_data or len(self.spike_data['unit_info']) == 0:
            self.result_status.set("スパイクデータなし")
            return

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

            # 全体解析（全LFPチャンネル × 全帯域）
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
                        lfp_channel=0,
                        min_spikes=min_spikes // 2,
                        verbose=True)
                    if unit_key not in self.condition_results:
                        self.condition_results[unit_key] = {}
                    self.condition_results[unit_key][band_name] = cond

        # FDR補正
        from phase_locking import apply_fdr_to_phase_results
        self.fdr_significant, n_total, n_sig = apply_fdr_to_phase_results(
            self.phase_results)
        print(f"  FDR補正: {n_sig}/{n_total} 有意")

        n_units = len(self.phase_results)
        self.result_status.set(
            f"完了: {n_units} ユニット, FDR {n_sig}/{n_total} 有意")

        self._show_preview()

    def _convert_sorting_to_spike_data(self, quality_filter='all'):
        """スパイクソーティング結果を spike_processing 形式に変換"""
        from spike_processing import UnitInfo
        unit_info_list = []
        spike_times_dict = {}

        for ch, result in self.spike_results.items():
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
                    t_start=float(unit.spike_times[0]) if len(unit.spike_times) > 0 else 0,
                    t_stop=float(unit.spike_times[-1]) if len(unit.spike_times) > 0 else 0,
                    unit_key=key)
                unit_info_list.append(ui)

        return {
            'spike_trains': [],
            'unit_info': unit_info_list,
            'spike_times': spike_times_dict
        }

    # ============================
    # プレビュー
    # ============================

    def _show_preview(self):
        """解析結果のプレビュー: 左=PPCヒートマップ(1枚)、右=全帯域極座標"""
        self.result_fig.clear()

        if not self.phase_results:
            self.result_canvas.draw()
            return

        bands = self._get_bands()
        band_names = list(bands.keys())
        n_bands = len(band_names)
        n_ch = self.lfp_cleaned.shape[1] if self.lfp_cleaned.ndim > 1 else 1

        # チャンネルコンボ更新
        ch_labels = ([f'Ch{c}' for c in self.original_ch_numbers[:n_ch]]
                     if self.original_ch_numbers
                     else [f'Ch{i}' for i in range(n_ch)])
        self.preview_ch_combo['values'] = ch_labels
        sel_ch_str = self.preview_ch_var.get()
        if sel_ch_str not in ch_labels:
            self.preview_ch_var.set(ch_labels[0])
        sel_ch_idx = ch_labels.index(self.preview_ch_var.get()) if self.preview_ch_var.get() in ch_labels else 0

        # --- 全ユニット平均 PPC/MRL 行列 (LFP ch × band) ---
        ppc_mat = np.zeros((n_ch, n_bands))
        count_mat = np.zeros((n_ch, n_bands))
        fdr_any = np.zeros((n_ch, n_bands), dtype=bool)
        fdr_sig = getattr(self, 'fdr_significant', {})

        for unit_key, results in self.phase_results.items():
            for bi, band in enumerate(band_names):
                if band not in results:
                    continue
                for ch in range(n_ch):
                    r = results[band].get(ch)
                    if r is not None:
                        ppc_mat[ch, bi] += r.ppc
                        count_mat[ch, bi] += 1
                    # FDR: いずれかのユニットで有意なら有意
                    if fdr_sig.get(unit_key, {}).get(band, {}).get(ch, False):
                        fdr_any[ch, bi] = True

        # 平均
        with np.errstate(divide='ignore', invalid='ignore'):
            ppc_mean = np.where(count_mat > 0, ppc_mat / count_mat, 0)

        # レイアウト: 左=ヒートマップ, 右=帯域数分の極座標
        gs = self.result_fig.add_gridspec(
            1, 1 + n_bands, wspace=0.3,
            width_ratios=[1.5] + [1] * n_bands)

        # --- 左: PPC ヒートマップ (縦=LFP ch, 横=帯域) ---
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
                                fontsize=10, color='white', fontweight='bold')
                else:
                    ax_heat.add_patch(matplotlib.patches.Rectangle(
                        (bi - 0.5, ch - 0.5), 1, 1,
                        fill=True, facecolor='gray', alpha=0.35, edgecolor='none'))

        # 選択中のチャンネルを枠で強調
        for bi in range(n_bands):
            ax_heat.add_patch(matplotlib.patches.Rectangle(
                (bi - 0.5, sel_ch_idx - 0.5), 1, 1,
                fill=False, edgecolor='red', lw=2))

        # --- 右: 全帯域の極座標ヒストグラム（選択LFP ch） ---
        cmap_band = matplotlib.colormaps.get_cmap('Set1')
        for bi, band in enumerate(band_names):
            ax = self.result_fig.add_subplot(gs[0, 1 + bi], projection='polar')

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
                            arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

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

    # ============================
    # グランドサマリー
    # ============================

    def _plot_grand_summary(self):
        if not self.phase_results:
            messagebox.showwarning("Warning", "先に解析を実行してください")
            return

        channel_spacing = float(self.vars['channel_spacing_um'].get())

        from phase_plotting_v6 import plot_phase_grand_summary
        plot_phase_grand_summary(
            phase_results=self.phase_results,
            condition_results=self.condition_results,
            bands=self._get_bands(),
            original_ch_numbers=self.original_ch_numbers,
            spike_data=self.spike_data,
            output_dir=self.output_dir,
            basename=self.basename,
            lfp_cleaned=self.lfp_cleaned,
            lfp_times=self.lfp_times,
            fs=self.fs,
            stim_times=self.stim_times,
            channel_spacing_um=channel_spacing,
            fdr_significant=getattr(self, 'fdr_significant', None),
            show=True,
            save=self.save_plots_var.get()
        )

        if self.save_csv_var.get() and self.spike_data:
            from phase_plotting import save_phase_locking_csv
            save_phase_locking_csv(
                self.phase_results,
                self.spike_data['unit_info'],
                {uk: cr.get(list(self._get_bands().keys())[0], {})
                 for uk, cr in self.condition_results.items()},
                self.output_dir, self.basename)

    def _finish(self):
        self.root.quit()
        self.root.destroy()
        if self.on_done:
            self.on_done(self.phase_results, self.condition_results)

    # ============================
    # 設定保存/読込
    # ============================

    def _save_config(self):
        d = {}
        # 基本パラメータ
        for k, v in self.vars.items():
            d[k] = v.get()
        d['condition_analysis'] = self.cond_var.get()
        # 帯域設定
        band_cfg = {}
        for bw in self.band_widgets:
            band_cfg[bw['name']] = {
                'enabled': bw['enabled'].get(),
                'lo': bw['lo'].get(),
                'hi': bw['hi'].get(),
            }
        d['bands'] = band_cfg
        try:
            with open(PHASE_CONFIG_FILE, 'w') as f:
                json.dump(d, f, indent=2)
        except Exception:
            pass

    def _load_config(self):
        if not os.path.exists(PHASE_CONFIG_FILE):
            return
        try:
            with open(PHASE_CONFIG_FILE, 'r') as f:
                d = json.load(f)
            for k, v in d.items():
                if k in self.vars:
                    if k == 'spike_source' and self.spike_results:
                        continue
                    self.vars[k].set(v)
            # 品質フィルタ: 内部値 → ラベル
            qf_val = self.vars.get('spike_quality_filter')
            if qf_val:
                label = QUALITY_FILTER_REVERSE.get(qf_val.get(), QUALITY_FILTER_LABELS[0])
                self._qf_label_var.set(label)
            # 帯域設定
            if 'bands' in d:
                for bw in self.band_widgets:
                    if bw['name'] in d['bands']:
                        bc = d['bands'][bw['name']]
                        bw['enabled'].set(bc.get('enabled', True))
                        bw['lo'].set(bc.get('lo', bw['lo'].get()))
                        bw['hi'].set(bc.get('hi', bw['hi'].get()))
            if 'condition_analysis' in d:
                self.cond_var.set(d['condition_analysis'])
        except Exception:
            pass

    def run(self):
        self.root.mainloop()


def launch_phase_gui(**kwargs):
    gui = PhaseGUI(**kwargs)
    gui.run()
