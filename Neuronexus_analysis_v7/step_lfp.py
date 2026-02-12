"""
step_lfp.py - Step 1: LFP フィルタリングパネル

lfp_filter_gui.py の設定UIを統合GUI用Frameとして移植。
LfpConfig, BandEditorFrame, 5タブ設定, フィルタプレビュー, パイプライン実行。
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import os
import threading
from dataclasses import dataclass, field, asdict
from typing import Dict, Tuple, Any

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from main_gui import StepPanel


# ============================================================
# LFP処理設定 (v6 lfp_filter_gui.py から移植)
# ============================================================
@dataclass
class LfpConfig:
    """LFP前処理の全設定"""
    # --- バンドパスフィルタ ---
    filter_enabled: bool = True
    filter_type: str = 'iir'
    filter_lowcut: float = 0.1
    filter_highcut: float = 100.0
    filter_order: int = 4
    filter_fir_numtaps: int = 0

    # --- ノッチフィルタ ---
    notch_enabled: bool = True
    notch_freq: float = 60.0
    notch_Q: float = 60.0

    # --- 高調波ノイズ除去 ---
    harmonic_removal_enabled: bool = True
    harmonic_fundamental: float = 10.0
    harmonic_count: int = 5
    harmonic_q: float = 50.0

    # --- 環境ノイズ ---
    noise_removal_enabled: bool = False
    noise_file: str = ""
    noise_threshold_db: float = 10.0

    # --- チャンネル処理 ---
    bad_channel_detection: bool = True
    bad_channel_threshold: float = 3.0
    manual_bad_channels: str = ""

    # --- モーション解析 ---
    motion_analysis: bool = True
    motion_roi: str = ""
    motion_percentile: float = 75.0
    motion_expand_sec: float = 0.1

    # --- ICA ---
    ica_enabled: bool = True
    ica_noise_ratio_threshold: float = 1.5
    ica_max_remove: int = 4

    # --- 解析設定 ---
    n_sessions: int = 9
    n_stim_per_session: int = 10
    baseline_pre_sec: float = 3.0
    post_duration_sec: float = 3.0

    # --- 帯域パワー ---
    bands: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 14),
        'beta': (14, 30), 'gamma': (30, 80)
    })

    # --- 表示 ---
    power_freq_min: float = 0.5
    power_freq_max: float = 100.0
    fft_freq_max: float = 300.0
    plot_t_start: float = 0.0
    plot_t_end: float = 0.0

    # --- 出力 ---
    show_plots: bool = True
    save_plots: bool = True
    output_dir: str = ""


# ============================================================
# 帯域エディタ (v6 lfp_filter_gui.py から移植)
# ============================================================
class BandEditorFrame(ttk.LabelFrame):
    """周波数帯域を編集するフレーム"""
    PRESETS = {
        'Standard': {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 14),
                     'beta': (14, 30), 'gamma': (30, 80)},
        'Rodent': {'delta': (1, 4), 'theta': (6, 10), 'alpha': (10, 14),
                   'beta': (14, 30), 'gamma': (30, 100)},
        'High Gamma': {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13),
                       'beta': (13, 30), 'low_gamma': (30, 60), 'high_gamma': (60, 120)},
        'Simple': {'low': (1, 30), 'high': (30, 100)},
    }
    COLORS = ['#3b82f6', '#22c55e', '#f59e0b', '#a855f7', '#ef4444',
              '#06b6d4', '#ec4899', '#84cc16']

    def __init__(self, parent, initial_bands=None):
        super().__init__(parent, text="周波数帯域設定")
        if initial_bands is None:
            initial_bands = self.PRESETS['Standard'].copy()
        self.band_rows = []
        self._build_ui()
        self.set_bands(initial_bands)

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill='x', padx=5, pady=3)
        ttk.Label(top, text="プリセット:").pack(side='left')
        self.preset_var = tk.StringVar(master=self._toplevel, value="Standard")
        combo = ttk.Combobox(top, textvariable=self.preset_var,
                             values=list(self.PRESETS.keys()),
                             state='readonly', width=12)
        combo.pack(side='left', padx=5)
        combo.bind('<<ComboboxSelected>>', self._on_preset)
        ttk.Button(top, text="+ 帯域追加", command=self._add_empty,
                   width=10).pack(side='right')

        self.list_frame = ttk.Frame(self)
        self.list_frame.pack(fill='both', expand=True, padx=5, pady=3)

    def _on_preset(self, event=None):
        name = self.preset_var.get()
        if name in self.PRESETS:
            self.set_bands(self.PRESETS[name])

    def set_bands(self, bands_dict):
        for r in self.band_rows:
            r['frame'].destroy()
        self.band_rows = []
        for i, (name, (lo, hi)) in enumerate(bands_dict.items()):
            self._add_row(name, lo, hi, self.COLORS[i % len(self.COLORS)])

    def _add_empty(self):
        lo, hi = 1, 10
        if self.band_rows:
            try:
                hi_prev = self.band_rows[-1]['high_var'].get()
                lo = hi_prev
                hi = lo + 20
            except Exception:
                pass
        self._add_row(f"band{len(self.band_rows)+1}", lo, hi,
                      self.COLORS[len(self.band_rows) % len(self.COLORS)])

    def _add_row(self, name, lo, hi, color):
        f = ttk.Frame(self.list_frame)
        f.pack(fill='x', pady=1)
        tk.Label(f, bg=color, width=2, height=1).pack(side='left', padx=2)
        nv = tk.StringVar(master=self._toplevel, value=name)
        ttk.Entry(f, textvariable=nv, width=10).pack(side='left', padx=2)
        lv = tk.DoubleVar(master=self._toplevel, value=lo)
        ttk.Entry(f, textvariable=lv, width=6).pack(side='left', padx=1)
        ttk.Label(f, text="~").pack(side='left')
        hv = tk.DoubleVar(master=self._toplevel, value=hi)
        ttk.Entry(f, textvariable=hv, width=6).pack(side='left', padx=1)
        ttk.Label(f, text="Hz").pack(side='left', padx=2)
        ttk.Button(f, text="x", width=2,
                   command=lambda: self._remove(f)).pack(side='left', padx=2)
        self.band_rows.append({'frame': f, 'name_var': nv,
                               'low_var': lv, 'high_var': hv, 'color': color})

    def _remove(self, frame):
        if len(self.band_rows) <= 1:
            return
        for i, r in enumerate(self.band_rows):
            if r['frame'] == frame:
                frame.destroy()
                self.band_rows.pop(i)
                break

    def get_bands(self):
        bands = {}
        for r in self.band_rows:
            try:
                n = r['name_var'].get().strip()
                lo = r['low_var'].get()
                hi = r['high_var'].get()
                if n and lo < hi:
                    bands[n] = (lo, hi)
            except Exception:
                pass
        return bands

    def get_bands_json(self):
        return {k: list(v) for k, v in self.get_bands().items()}

    def set_bands_from_json(self, d):
        valid = {}
        for k, v in d.items():
            try:
                lo, hi = float(v[0]), float(v[1])
                if k and lo < hi:
                    valid[k] = (lo, hi)
            except (ValueError, TypeError, IndexError):
                continue
        if valid:
            self.set_bands(valid)


# ============================================================
# パネル
# ============================================================
class Panel(StepPanel):
    """LFPフィルタリングパネル"""

    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.vars = {}
        self.band_editor = None
        self.preview_fig = None
        self.preview_canvas = None
        self._build_ui()

    # ===========================================================
    # UI構築
    # ===========================================================

    def _build_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill='both', expand=True, padx=5, pady=5)

        tab1 = ttk.Frame(nb)
        nb.add(tab1, text=' フィルタ ')
        self._build_tab_filter(tab1)

        tab2 = ttk.Frame(nb)
        nb.add(tab2, text=' ノイズ除去 ')
        self._build_tab_noise(tab2)

        tab3 = ttk.Frame(nb)
        nb.add(tab3, text=' 前処理 ')
        self._build_tab_preprocess(tab3)

        tab4 = ttk.Frame(nb)
        nb.add(tab4, text=' 解析設定 ')
        self._build_tab_analysis(tab4)

        tab5 = ttk.Frame(nb)
        nb.add(tab5, text=' プレビュー ')
        self._build_tab_preview(tab5)

        # 下部: ボタン
        bf = ttk.Frame(self)
        bf.pack(fill='x', padx=10, pady=8)
        ttk.Button(bf, text="リセット", command=self._reset).pack(side='left', padx=3)
        ttk.Button(bf, text="LFP Pipeline 実行", command=self._run,
                   style='Run.TButton', width=20).pack(side='right', padx=5)

    # --- Tab 1: フィルタ ---
    def _build_tab_filter(self, parent):
        f = self._scrollable(parent)
        r = 0
        r = self._section(f, "バンドパスフィルタ", r)
        r = self._check(f, "filter_enabled", "バンドパスフィルタを適用", r, True)
        r = self._combo(f, "filter_type", "フィルタ種類", r, ['iir', 'fir'], 'iir')
        r = self._num(f, "filter_lowcut", "ローカット (Hz)", r, 0.1)
        r = self._num(f, "filter_highcut", "ハイカット (Hz)", r, 100.0)
        r = self._num(f, "filter_order", "IIR次数", r, 4, is_int=True)
        r = self._num(f, "filter_fir_numtaps", "FIRタップ数 (0=自動)", r, 0, is_int=True)

        r = self._section(f, "ノッチフィルタ", r)
        r = self._check(f, "notch_enabled", "ノッチフィルタを適用", r, True)
        r = self._num(f, "notch_freq", "ノッチ周波数 (Hz)", r, 60.0)
        r = self._num(f, "notch_Q", "Q値", r, 60.0)

    # --- Tab 2: ノイズ除去 ---
    def _build_tab_noise(self, parent):
        f = self._scrollable(parent)
        r = 0
        r = self._section(f, "高調波ノイズ除去 (ピエゾ)", r)
        r = self._check(f, "harmonic_removal_enabled", "高調波ノイズ除去", r, True)
        r = self._num(f, "harmonic_fundamental", "基本周波数 (Hz)", r, 10.0)
        r = self._num(f, "harmonic_count", "高調波の数", r, 5, is_int=True)
        r = self._num(f, "harmonic_q", "Q値", r, 50.0)

        r = self._section(f, "環境ノイズ除去", r)
        r = self._check(f, "noise_removal_enabled", "環境ノイズ除去", r, False)
        r = self._file_input(f, "noise_file", "ノイズ記録ファイル (.plx)", r)
        r = self._num(f, "noise_threshold_db", "ピーク検出閾値 (dB)", r, 10.0)

    # --- Tab 3: 前処理 ---
    def _build_tab_preprocess(self, parent):
        f = self._scrollable(parent)
        r = 0
        r = self._section(f, "チャンネル処理", r)
        r = self._check(f, "bad_channel_detection", "悪いチャンネル自動検出", r, True)
        r = self._num(f, "bad_channel_threshold", "検出閾値 (MAD倍数)", r, 3.0)
        r = self._text(f, "manual_bad_channels", "手動除外 (例: 7,12)", r, "")

        r = self._section(f, "モーション解析", r)
        r = self._check(f, "motion_analysis", "モーション解析を実行", r, True)
        r = self._text(f, "motion_roi", "ROI固定 (x,y,w,h) 空=手動", r, "")
        r = self._num(f, "motion_percentile", "ノイズ閾値パーセンタイル", r, 75.0)
        r = self._num(f, "motion_expand_sec", "マスク拡張 (秒)", r, 0.1)

        r = self._section(f, "ICA", r)
        r = self._check(f, "ica_enabled", "ICAアーティファクト除去", r, True)
        r = self._num(f, "ica_noise_ratio_threshold", "ICA除去閾値", r, 1.5)
        r = self._num(f, "ica_max_remove", "最大除去成分数", r, 4, is_int=True)

        r = self._section(f, "実験設定", r)
        r = self._num(f, "n_sessions", "セッション数", r, 9, is_int=True)
        r = self._num(f, "n_stim_per_session", "セッション毎刺激数", r, 10, is_int=True)
        r = self._num(f, "baseline_pre_sec", "Baseline期間 (秒)", r, 3.0)
        r = self._num(f, "post_duration_sec", "Post期間 (秒)", r, 3.0)

    # --- Tab 4: 解析設定 ---
    def _build_tab_analysis(self, parent):
        f = self._scrollable(parent)
        r = 0
        r = self._section(f, "周波数帯域", r)
        self.band_editor = BandEditorFrame(f)
        self.band_editor.grid(row=r, column=0, columnspan=3,
                              sticky='ew', padx=5, pady=5)
        r += 1

        r = self._section(f, "表示範囲", r)
        r = self._num(f, "power_freq_min", "パワー表示 最小 (Hz)", r, 0.5)
        r = self._num(f, "power_freq_max", "パワー表示 最大 (Hz)", r, 100.0)
        r = self._num(f, "fft_freq_max", "FFT比較 最大 (Hz)", r, 300.0)
        r = self._num(f, "plot_t_start", "時間表示 開始 (秒)", r, 0.0)
        r = self._num(f, "plot_t_end", "時間表示 終了 (秒)", r, 0.0)

        r = self._section(f, "出力", r)
        r = self._check(f, "show_plots", "プロットを画面表示", r, True)
        r = self._check(f, "save_plots", "プロットを画像保存", r, True)

    # --- Tab 5: プレビュー ---
    def _build_tab_preview(self, parent):
        ctrl = ttk.Frame(parent)
        ctrl.pack(fill='x', padx=5, pady=5)
        ttk.Button(ctrl, text="フィルタ応答プレビュー",
                   command=self._preview_filter_response).pack(side='left', padx=5)
        ttk.Button(ctrl, text="PSDプレビュー (要データ)",
                   command=self._preview_psd).pack(side='left', padx=5)

        self.preview_fig = Figure(figsize=(9, 5), dpi=100)
        self.preview_canvas = FigureCanvasTkAgg(self.preview_fig, master=parent)
        self.preview_canvas.get_tk_widget().pack(fill='both', expand=True)

        tb_frame = ttk.Frame(parent)
        tb_frame.pack(fill='x')
        NavigationToolbar2Tk(self.preview_canvas, tb_frame)

    # ===========================================================
    # プレビュー機能
    # ===========================================================

    def _preview_filter_response(self):
        """フィルタの周波数応答をプレビュー"""
        from scipy import signal as sig

        self.preview_fig.clear()
        ax_mag = self.preview_fig.add_subplot(121)
        ax_phase = self.preview_fig.add_subplot(122)

        fs = 1000
        pd = self.app.plx_data
        if pd and pd.lfp_fs > 0:
            fs = pd.lfp_fs

        nyq = 0.5 * fs
        try:
            lowcut = float(self.vars['filter_lowcut']['var'].get())
            highcut = float(self.vars['filter_highcut']['var'].get())
            ftype = self.vars['filter_type']['var'].get()
        except Exception:
            return

        if ftype == 'fir':
            numtaps = int(self.vars['filter_fir_numtaps']['var'].get())
            if numtaps <= 0:
                from lfp_processing import estimate_fir_taps
                numtaps = estimate_fir_taps(fs, lowcut, highcut)
            if numtaps % 2 == 0:
                numtaps += 1
            b = sig.firwin(numtaps, [lowcut / nyq, highcut / nyq],
                           pass_zero='bandpass')
            w, h = sig.freqz(b, [1.0], worN=2048, fs=fs)
            label = f"FIR (taps={numtaps})"
        else:
            order = int(self.vars['filter_order']['var'].get())
            sos = sig.butter(order, [lowcut / nyq, highcut / nyq],
                             btype='bandpass', output='sos')
            w, h = sig.sosfreqz(sos, worN=2048, fs=fs)
            label = f"IIR Butterworth (order={order})"

        ax_mag.plot(w, 20 * np.log10(np.abs(h) + 1e-10), 'b-', label=label)
        ax_mag.set_ylabel('Magnitude (dB)')
        ax_mag.set_xlabel('Frequency (Hz)')
        ax_mag.set_title('Amplitude Response')
        ax_mag.set_xlim(0, min(highcut * 3, nyq))
        ax_mag.set_ylim(-60, 5)
        ax_mag.axhline(-3, color='gray', ls='--', alpha=0.5)
        ax_mag.legend(fontsize=8)
        ax_mag.grid(True, alpha=0.3)

        ax_phase.plot(w, np.degrees(np.angle(h)), 'r-', alpha=0.7)
        ax_phase.set_ylabel('Phase (degrees)')
        ax_phase.set_xlabel('Frequency (Hz)')
        ax_phase.set_title('Phase Response')
        ax_phase.set_xlim(0, min(highcut * 3, nyq))
        ax_phase.grid(True, alpha=0.3)

        # ノッチフィルタを重畳表示
        try:
            notch_on = self.vars['notch_enabled']['var'].get()
            if notch_on:
                nf = float(self.vars['notch_freq']['var'].get())
                nq = float(self.vars['notch_Q']['var'].get())
                bn, an = sig.iirnotch(nf, nq, fs)
                wn, hn = sig.freqz(bn, an, worN=2048, fs=fs)
                ax_mag.plot(wn, 20 * np.log10(np.abs(hn) + 1e-10),
                            'g--', alpha=0.7, label=f'Notch {nf}Hz (Q={nq})')
                ax_mag.legend(fontsize=8)
        except Exception:
            pass

        self.preview_fig.tight_layout()
        self.preview_canvas.draw()

    def _preview_psd(self):
        """ロード済みデータがあればPSDプレビュー"""
        pd = self.app.plx_data
        if pd is None or pd.lfp_raw is None:
            messagebox.showinfo("情報", "先にStep 0でデータを読み込んでください")
            return

        from scipy.signal import welch
        self.preview_fig.clear()
        ax = self.preview_fig.add_subplot(111)

        lfp = pd.lfp_raw
        fs = pd.lfp_fs
        mean_sig = lfp.mean(axis=1)
        freqs, psd = welch(mean_sig, fs,
                           nperseg=min(int(fs * 2), len(mean_sig) // 2))

        ax.semilogy(freqs, psd, 'k-', lw=0.8, label='Raw PSD (ch avg)')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD (V^2/Hz)')
        ax.set_title('Power Spectral Density (Raw Data)')

        try:
            fmax = float(self.vars['power_freq_max']['var'].get())
            ax.set_xlim(0, fmax)
        except Exception:
            ax.set_xlim(0, 100)

        # 帯域を塗り分け
        bands = self.band_editor.get_bands() if self.band_editor else {}
        colors = BandEditorFrame.COLORS
        for i, (name, (lo, hi)) in enumerate(bands.items()):
            ax.axvspan(lo, hi, alpha=0.15,
                       color=colors[i % len(colors)], label=name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        self.preview_fig.tight_layout()
        self.preview_canvas.draw()

    # ===========================================================
    # LfpConfig変換
    # ===========================================================

    def _get_lfp_config(self) -> LfpConfig:
        """GUI値からLfpConfigオブジェクトを生成"""
        kwargs = {}
        valid_keys = set(LfpConfig.__dataclass_fields__.keys())
        for key, info in self.vars.items():
            if key not in valid_keys:
                continue
            try:
                v = info['var'].get()
                t = info['type']
                if t == 'bool':
                    kwargs[key] = v
                elif t == 'int':
                    kwargs[key] = int(v)
                elif t == 'float':
                    kwargs[key] = float(v)
                else:
                    kwargs[key] = v
            except Exception:
                kwargs[key] = info['default']

        if self.band_editor:
            kwargs['bands'] = self.band_editor.get_bands()

        kwargs['output_dir'] = self.state.output_dir
        return LfpConfig(**kwargs)

    # ===========================================================
    # 実行
    # ===========================================================

    def _run(self):
        pd = self.app.plx_data
        if pd is None:
            messagebox.showwarning("Warning",
                                   "先にStep 0でデータを読み込んでください")
            return

        if not self.app.reset_downstream("lfp"):
            return

        config = self._get_lfp_config()

        # plx_data の値を優先適用
        if hasattr(pd, 'trim_start') and pd.trim_start is not None:
            config.plot_t_start = pd.trim_start
            config.plot_t_end = pd.trim_end
        if pd.session_times is not None and len(pd.session_times) > 0:
            config.n_sessions = len(pd.session_times)
            if pd.stim_times is not None and len(pd.stim_times) > 0:
                n_per = len(pd.stim_times) // len(pd.session_times)
                if n_per > 0:
                    config.n_stim_per_session = n_per
        if pd.lfp_fs > 0:
            nyq = pd.lfp_fs / 2.0
            if config.filter_highcut >= nyq:
                config.filter_highcut = nyq - 1

        # 確認ダイアログ
        bands_str = ", ".join(
            [f"{k}({v[0]}-{v[1]})" for k, v in config.bands.items()])
        if config.filter_type == 'fir':
            taps_str = ("自動" if config.filter_fir_numtaps == 0
                        else str(config.filter_fir_numtaps))
            finfo = (f"{config.filter_lowcut}-{config.filter_highcut}Hz "
                     f"[FIR taps={taps_str}]")
        else:
            finfo = (f"{config.filter_lowcut}-{config.filter_highcut}Hz "
                     f"[IIR order={config.filter_order}]")
        if config.notch_enabled:
            finfo += f", Notch {config.notch_freq}Hz"

        msg = (f"LFPパイプラインを実行しますか?\n\n"
               f"フィルタ: {finfo}\n"
               f"高調波除去: {'ON' if config.harmonic_removal_enabled else 'OFF'}\n"
               f"ICA: {'ON' if config.ica_enabled else 'OFF'}\n"
               f"帯域: {bands_str}\n")

        if not messagebox.askyesno("確認", msg):
            return

        self.app.set_status("LFPパイプライン実行中...")

        def _execute():
            try:
                from lfp_pipeline import run_lfp_pipeline
                results = run_lfp_pipeline(config, pd)
                self.app.lfp_results = results
                self.app.spike_results = None
                self.app.phase_results = None
                self.after(0, self._on_done)
            except Exception as e:
                self.after(0, lambda: self._on_error(str(e)))

        t = threading.Thread(target=_execute, daemon=True)
        t.start()

    def _on_done(self):
        self.app.complete_step("lfp")
        self.app.set_status("LFPパイプライン完了")

    def _on_error(self, msg: str):
        messagebox.showerror("Error", f"LFPパイプラインエラー:\n{msg}")
        self.app.set_status("LFPパイプラインエラー")

    def _reset(self):
        """設定をデフォルトにリセット"""
        cfg = LfpConfig()
        d = asdict(cfg)
        for k, info in self.vars.items():
            if k in d:
                info['var'].set(d[k])
        if self.band_editor:
            self.band_editor.set_bands(BandEditorFrame.PRESETS['Standard'])

    # ===========================================================
    # ライフサイクル
    # ===========================================================

    def on_show(self):
        if self.state.lfp_config:
            self.set_config(self.state.lfp_config)
        self._apply_plx_data()

    def _apply_plx_data(self):
        """plx_data の情報をGUIに反映"""
        pd = self.app.plx_data
        if pd is None:
            return
        if hasattr(pd, 'trim_start') and pd.trim_start is not None:
            if 'plot_t_start' in self.vars:
                self.vars['plot_t_start']['var'].set(str(pd.trim_start))
            if 'plot_t_end' in self.vars:
                self.vars['plot_t_end']['var'].set(str(pd.trim_end))
        if pd.session_times is not None and len(pd.session_times) > 0:
            if 'n_sessions' in self.vars:
                self.vars['n_sessions']['var'].set(str(len(pd.session_times)))
            if (pd.stim_times is not None and len(pd.stim_times) > 0):
                n_per = len(pd.stim_times) // len(pd.session_times)
                if n_per > 0 and 'n_stim_per_session' in self.vars:
                    self.vars['n_stim_per_session']['var'].set(str(n_per))
        if pd.lfp_fs > 0 and 'filter_highcut' in self.vars:
            nyq = pd.lfp_fs / 2.0
            try:
                current = float(self.vars['filter_highcut']['var'].get())
                if current >= nyq:
                    self.vars['filter_highcut']['var'].set(str(nyq - 1))
            except Exception:
                pass

    # ===========================================================
    # 設定取得 / 反映
    # ===========================================================

    def get_config(self) -> dict:
        """現在のGUI設定値を辞書で返す（ProjectState保存用）"""
        d = {}
        for k, info in self.vars.items():
            try:
                v = info['var'].get()
                if info['type'] == 'bool':
                    d[k] = v
                elif info['type'] == 'int':
                    d[k] = int(v)
                elif info['type'] == 'float':
                    d[k] = float(v)
                else:
                    d[k] = v
            except Exception:
                d[k] = info['default']
        if self.band_editor:
            d['bands'] = self.band_editor.get_bands_json()
        return d

    def set_config(self, config: dict):
        """設定値をGUIに反映"""
        if not config:
            return
        for k, v in config.items():
            if k == 'bands':
                if self.band_editor and v:
                    self.band_editor.set_bands_from_json(v)
                continue
            if k not in self.vars:
                continue
            info = self.vars[k]
            try:
                if info['type'] == 'bool':
                    info['var'].set(bool(v))
                else:
                    info['var'].set(v)
            except Exception:
                pass

    # ===========================================================
    # UIヘルパー (v6同等)
    # ===========================================================

    def _scrollable(self, parent):
        c = tk.Canvas(parent)
        sb = ttk.Scrollbar(parent, orient='vertical', command=c.yview)
        sf = ttk.Frame(c)
        sf.bind('<Configure>',
                lambda e: c.configure(scrollregion=c.bbox('all')))
        c.create_window((0, 0), window=sf, anchor='nw')
        c.configure(yscrollcommand=sb.set)

        def _wheel(event):
            c.yview_scroll(int(-1 * (event.delta / 120)), 'units')
        c.bind('<MouseWheel>', _wheel)
        sf.bind('<MouseWheel>', _wheel)
        sb.pack(side='right', fill='y')
        c.pack(side='left', fill='both', expand=True)
        return sf

    def _section(self, parent, text, row):
        ttk.Label(parent, text=text, font=('', 10, 'bold')).grid(
            row=row, column=0, columnspan=3, sticky='w', pady=(12, 3), padx=5)
        ttk.Separator(parent, orient='horizontal').grid(
            row=row + 1, column=0, columnspan=3, sticky='ew')
        return row + 2

    def _check(self, parent, key, label, row, default=True):
        var = tk.BooleanVar(master=self.winfo_toplevel(), value=default)
        self.vars[key] = {'var': var, 'type': 'bool', 'default': default}
        ttk.Checkbutton(parent, text=label, variable=var).grid(
            row=row, column=0, columnspan=2, sticky='w', padx=20, pady=2)
        return row + 1

    def _num(self, parent, key, label, row, default=0.0, is_int=False):
        var = tk.StringVar(master=self.winfo_toplevel(), value=str(default))
        self.vars[key] = {'var': var,
                          'type': 'int' if is_int else 'float',
                          'default': default}
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky='w', padx=20, pady=2)
        ttk.Entry(parent, textvariable=var, width=10).grid(
            row=row, column=1, sticky='w', pady=2)
        return row + 1

    def _text(self, parent, key, label, row, default=""):
        var = tk.StringVar(master=self.winfo_toplevel(), value=default)
        self.vars[key] = {'var': var, 'type': 'str', 'default': default}
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky='w', padx=20, pady=2)
        ttk.Entry(parent, textvariable=var, width=20).grid(
            row=row, column=1, sticky='w', pady=2)
        return row + 1

    def _combo(self, parent, key, label, row, options, default=None):
        if default is None:
            default = options[0]
        var = tk.StringVar(master=self.winfo_toplevel(), value=default)
        self.vars[key] = {'var': var, 'type': 'str', 'default': default}
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky='w', padx=20, pady=2)
        ttk.Combobox(parent, textvariable=var, values=options,
                     state='readonly', width=12).grid(
            row=row, column=1, sticky='w', pady=2)
        return row + 1

    def _file_input(self, parent, key, label, row, is_dir=False):
        var = tk.StringVar(master=self.winfo_toplevel(), value="")
        self.vars[key] = {'var': var, 'type': 'str', 'default': ''}
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky='w', padx=20, pady=2)
        ef = ttk.Frame(parent)
        ef.grid(row=row, column=1, columnspan=2, sticky='ew', pady=2)
        ttk.Entry(ef, textvariable=var, width=35).pack(
            side='left', fill='x', expand=True)
        if is_dir:
            cmd = lambda: var.set(filedialog.askdirectory() or var.get())
        else:
            cmd = lambda: var.set(
                filedialog.askopenfilename(
                    filetypes=[("PLX files", "*.plx")]
                ) or var.get())
        ttk.Button(ef, text="参照", command=cmd, width=5).pack(
            side='left', padx=3)
        return row + 1
