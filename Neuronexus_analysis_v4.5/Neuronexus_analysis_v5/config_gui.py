"""
config_gui.py - パイプライン設定GUI（タブ版）
タブで整理 + 実行ボタンは常に下部に表示
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os
from dataclasses import asdict
from pipeline import PipelineConfig, run_pipeline


CONFIG_SAVE_PATH = os.path.join(os.path.dirname(__file__), "last_config.json")


# ============================================================
# 周波数帯域エディタ
# ============================================================
class BandEditorFrame(ttk.LabelFrame):
    """周波数帯域を編集するフレーム"""
    
    PRESETS = {
        'Standard': {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 14), 'beta': (14, 30), 'gamma': (30, 80)},
        'High Gamma': {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'low_gamma': (30, 60), 'high_gamma': (60, 120)},
        'Rodent': {'delta': (1, 4), 'theta': (6, 10), 'alpha': (10, 14), 'beta': (14, 30), 'gamma': (30, 100)},
        'Simple': {'low': (1, 30), 'high': (30, 100)},
    }
    
    COLORS = ['#3b82f6', '#22c55e', '#f59e0b', '#a855f7', '#ef4444', '#06b6d4', '#ec4899', '#84cc16']
    
    def __init__(self, parent, initial_bands=None):
        super().__init__(parent, text="📊 周波数帯域設定")
        
        if initial_bands is None:
            initial_bands = self.PRESETS['Standard'].copy()
        
        self.band_rows = []
        self.max_freq = 120
        
        self._build_ui()
        self.set_bands(initial_bands)
    
    def _build_ui(self):
        preset_frame = ttk.Frame(self)
        preset_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(preset_frame, text="プリセット:").pack(side='left')
        
        self.preset_var = tk.StringVar(value="Standard")
        preset_combo = ttk.Combobox(
            preset_frame, 
            textvariable=self.preset_var,
            values=list(self.PRESETS.keys()),
            state='readonly',
            width=12
        )
        preset_combo.pack(side='left', padx=5)
        preset_combo.bind('<<ComboboxSelected>>', self._on_preset_selected)
        
        ttk.Button(preset_frame, text="+ 帯域追加", command=self._add_empty_band, width=10).pack(side='right', padx=2)
        
        self.canvas = tk.Canvas(self, height=35, bg='#1a1a2e', highlightthickness=1, highlightbackground='#444')
        self.canvas.pack(fill='x', padx=5, pady=(5, 0))
        self.canvas.bind('<Configure>', lambda e: self._update_visualization())
        
        scale_frame = ttk.Frame(self)
        scale_frame.pack(fill='x', padx=5)
        self.scale_labels = []
        for i in range(5):
            lbl = ttk.Label(scale_frame, text="", font=('', 8))
            lbl.pack(side='left', expand=True)
            self.scale_labels.append(lbl)
        self._update_scale_labels()
        
        header = ttk.Frame(self)
        header.pack(fill='x', padx=5, pady=(10, 2))
        ttk.Label(header, text="", width=3).pack(side='left')
        ttk.Label(header, text="名前", width=10).pack(side='left', padx=2)
        ttk.Label(header, text="下限(Hz)", width=8).pack(side='left', padx=2)
        ttk.Label(header, text="", width=2).pack(side='left')
        ttk.Label(header, text="上限(Hz)", width=8).pack(side='left', padx=2)
        
        self.list_frame = ttk.Frame(self)
        self.list_frame.pack(fill='both', expand=True, padx=5, pady=5)
    
    def _update_scale_labels(self):
        for i, lbl in enumerate(self.scale_labels):
            hz = int(self.max_freq * i / 4)
            lbl.config(text=f"{hz}Hz")
    
    def _on_preset_selected(self, event=None):
        preset_name = self.preset_var.get()
        if preset_name in self.PRESETS:
            self.set_bands(self.PRESETS[preset_name])
    
    def set_bands(self, bands_dict):
        for row_data in self.band_rows:
            row_data['frame'].destroy()
        self.band_rows = []
        
        if bands_dict:
            max_high = max(high for low, high in bands_dict.values())
            self.max_freq = max(100, int(max_high * 1.2))
            self._update_scale_labels()
        
        for i, (name, (low, high)) in enumerate(bands_dict.items()):
            self._add_band_row(name, low, high, self.COLORS[i % len(self.COLORS)])
        
        self._update_visualization()
    
    def _add_empty_band(self):
        if self.band_rows:
            last_high = self.band_rows[-1]['high_var'].get()
            new_low = last_high
            new_high = min(last_high + 20, self.max_freq)
        else:
            new_low = 1
            new_high = 10
        
        color = self.COLORS[len(self.band_rows) % len(self.COLORS)]
        self._add_band_row(f"band{len(self.band_rows)+1}", new_low, new_high, color)
        self._update_visualization()
    
    def _add_band_row(self, name, low, high, color):
        frame = ttk.Frame(self.list_frame)
        frame.pack(fill='x', pady=2)
        
        color_lbl = tk.Label(frame, bg=color, width=3, height=1)
        color_lbl.pack(side='left', padx=(0, 5))
        
        name_var = tk.StringVar(value=name)
        ttk.Entry(frame, textvariable=name_var, width=10).pack(side='left', padx=2)
        
        low_var = tk.DoubleVar(value=low)
        ttk.Entry(frame, textvariable=low_var, width=8).pack(side='left', padx=2)
        low_var.trace_add('write', lambda *args: self._update_visualization())
        
        ttk.Label(frame, text="〜").pack(side='left')
        
        high_var = tk.DoubleVar(value=high)
        ttk.Entry(frame, textvariable=high_var, width=8).pack(side='left', padx=2)
        high_var.trace_add('write', lambda *args: self._update_visualization())
        
        ttk.Label(frame, text="Hz").pack(side='left', padx=(0, 5))
        
        ttk.Button(frame, text="✕", width=3,
                   command=lambda: self._remove_band_row(frame)).pack(side='left', padx=2)
        
        self.band_rows.append({
            'frame': frame,
            'name_var': name_var,
            'low_var': low_var,
            'high_var': high_var,
            'color': color,
            'color_lbl': color_lbl
        })
    
    def _remove_band_row(self, frame):
        if len(self.band_rows) <= 1:
            messagebox.showwarning("警告", "最低1つの帯域が必要です")
            return
        
        for i, row in enumerate(self.band_rows):
            if row['frame'] == frame:
                frame.destroy()
                self.band_rows.pop(i)
                break
        
        for i, row in enumerate(self.band_rows):
            new_color = self.COLORS[i % len(self.COLORS)]
            row['color'] = new_color
            row['color_lbl'].config(bg=new_color)
        
        self._update_visualization()
    
    def _update_visualization(self):
        self.canvas.delete('all')
        
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        if width <= 1:
            return
        
        padding = 10
        plot_width = width - 2 * padding
        
        for row in self.band_rows:
            try:
                low = row['low_var'].get()
                high = row['high_var'].get()
                color = row['color']
                name = row['name_var'].get()
                
                if low >= high or low < 0:
                    continue
                
                x1 = padding + (low / self.max_freq) * plot_width
                x2 = padding + (high / self.max_freq) * plot_width
                
                self.canvas.create_rectangle(
                    x1, 5, x2, height - 5,
                    fill=color, outline='white', width=1
                )
                
                if x2 - x1 > 30:
                    self.canvas.create_text(
                        (x1 + x2) / 2, height / 2,
                        text=name, fill='white', font=('', 8, 'bold')
                    )
            except tk.TclError:
                pass
        
        for i in range(5):
            x = padding + (plot_width * i / 4)
            self.canvas.create_line(x, height - 3, x, height, fill='#888')
    
    def get_bands(self):
        bands = {}
        for row in self.band_rows:
            try:
                name = row['name_var'].get().strip()
                low = row['low_var'].get()
                high = row['high_var'].get()
                
                if name and low < high:
                    bands[name] = (low, high)
            except (tk.TclError, ValueError):
                pass
        return bands
    
    def get_bands_for_json(self):
        return {name: list(vals) for name, vals in self.get_bands().items()}
    
    def set_bands_from_json(self, bands_json):
        bands = {name: tuple(vals) for name, vals in bands_json.items()}
        self.set_bands(bands)


# ============================================================
# メインGUI（タブ版）
# ============================================================
class ConfigGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Neuronexus Analysis - Pipeline Config")
        self.root.geometry("700x650")
        self.root.resizable(True, True)
        
        self.vars = {}
        self.group_vars = {}
        self.band_editor = None
        
        self._build_ui()
        self._load_last_config()
    
    def _build_ui(self):
        # ================================================================
        # 全体レイアウト: 上部（タブ）+ 下部（ボタン固定）
        # ================================================================
        
        # --- 上部: タブ ---
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # タブ1: 基本設定
        tab_basic = ttk.Frame(self.notebook)
        self.notebook.add(tab_basic, text=' 📁 基本設定 ')
        self._build_tab_basic(tab_basic)
        
        # タブ2: 前処理
        tab_preprocess = ttk.Frame(self.notebook)
        self.notebook.add(tab_preprocess, text=' 🔧 前処理 ')
        self._build_tab_preprocess(tab_preprocess)
        
        # タブ3: 解析
        tab_analysis = ttk.Frame(self.notebook)
        self.notebook.add(tab_analysis, text=' 📈 解析 ')
        self._build_tab_analysis(tab_analysis)
        
        # タブ4: 出力
        tab_output = ttk.Frame(self.notebook)
        self.notebook.add(tab_output, text=' 💾 出力 ')
        self._build_tab_output(tab_output)
        
        # --- 下部: ボタン（常に表示） ---
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill='x', padx=10, pady=10)
        
        # 左側ボタン
        ttk.Button(btn_frame, text="📂 読み込み", command=self._load_last_config).pack(side="left", padx=3)
        ttk.Button(btn_frame, text="🔄 リセット", command=self._reset_to_default).pack(side="left", padx=3)
        ttk.Button(btn_frame, text="💾 保存", command=self._save_config).pack(side="left", padx=3)
        
        # 右側: 実行ボタン（大きく目立つ）
        style = ttk.Style()
        style.configure("Run.TButton", font=("", 14, "bold"), padding=10)
        ttk.Button(btn_frame, text="▶️ 実行", command=self._run_pipeline, 
                   style="Run.TButton", width=15).pack(side="right", padx=5)
    
    # ================================================================
    # タブ1: 基本設定
    # ================================================================
    def _build_tab_basic(self, parent):
        frame = self._create_scrollable_frame(parent)
        row = 0
        
        # ファイル設定
        row = self._add_section_header(frame, "📁 ファイル設定", row)
        row = self._add_file_selector(frame, "plx_file", "PLXファイル", row)
        row = self._add_file_selector(frame, "output_dir", "出力先フォルダ", row, is_dir=True)
        
        # 実験設定
        row = self._add_section_header(frame, "🔬 実験設定", row)
        row = self._add_number_input(frame, "n_sessions", "セッション数", row, default=9, is_int=True)
        row = self._add_number_input(frame, "n_stim_per_session", "セッション毎の刺激数", row, default=10, is_int=True)
        row = self._add_number_input(frame, "baseline_pre_sec", "Baseline期間 (秒)", row, default=3.0)
        row = self._add_number_input(frame, "post_duration_sec", "Post期間 (秒)", row, default=3.0)
    
    # ================================================================
    # タブ2: 前処理
    # ================================================================
    def _build_tab_preprocess(self, parent):
        frame = self._create_scrollable_frame(parent)
        row = 0
        
        # === バンドパスフィルタ設定 ===
        row = self._add_section_header(frame, "🔧 バンドパスフィルタ", row, group_key="filter")
        row = self._add_checkbox(frame, "filter_enabled", "バンドパスフィルタを適用", row, default=True, group="filter")
        row = self._add_combo(frame, "filter_type", "フィルタ種類", row, ['iir', 'fir'], default='iir')
        row = self._add_number_input(frame, "filter_lowcut", "ローカット周波数 (Hz)", row, default=0.1)
        row = self._add_number_input(frame, "filter_highcut", "ハイカット周波数 (Hz)", row, default=100.0)
        row = self._add_number_input(frame, "filter_order", "IIR次数 (Butterworth)", row, default=4, is_int=True)
        row = self._add_number_input(frame, "filter_fir_numtaps", "FIRタップ数 (0=自動)", row, default=0, is_int=True)
        
        # === ノッチフィルタ設定（IIR固定）===
        row = self._add_section_header(frame, "🔇 ノッチフィルタ (IIR)", row)
        row = self._add_checkbox(frame, "notch_enabled", "ノッチフィルタを適用", row, default=True, group="filter")
        row = self._add_number_input(frame, "notch_freq", "ノッチ周波数 (Hz)", row, default=60.0)
        row = self._add_number_input(frame, "notch_Q", "Q値 (大=狭帯域)", row, default=60.0)
        
        # 高調波ノイズ除去
        row = self._add_section_header(frame, "🎵 高調波ノイズ除去 (ピエゾ)", row)
        row = self._add_checkbox(frame, "harmonic_removal_enabled", "高調波ノイズ除去", row, default=True)
        row = self._add_number_input(frame, "harmonic_fundamental", "基本周波数 (Hz)", row, default=10.0)
        row = self._add_number_input(frame, "harmonic_count", "高調波の数", row, default=5, is_int=True)
        row = self._add_number_input(frame, "harmonic_q", "Q値", row, default=50.0)
        
        # 環境ノイズ除去
        row = self._add_section_header(frame, "🌐 環境ノイズ除去", row)
        row = self._add_checkbox(frame, "noise_removal_enabled", "環境ノイズ除去を実行", row, default=False)
        row = self._add_file_selector(frame, "noise_file", "ノイズ記録ファイル (.plx)", row)
        row = self._add_number_input(frame, "noise_threshold_db", "ピーク検出閾値 (dB)", row, default=10.0)
        
        # チャンネル処理
        row = self._add_section_header(frame, "📊 チャンネル処理", row, group_key="channel")
        row = self._add_checkbox(frame, "bad_channel_detection", "悪いチャンネル自動検出", row, default=True, group="channel")
        row = self._add_number_input(frame, "bad_channel_threshold", "検出閾値 (MAD倍数)", row, default=3.0)
        row = self._add_text_input(frame, "manual_bad_channels", "手動除外 (例: 7,12)", row, default="")
        
        # モーション解析
        row = self._add_section_header(frame, "🎥 モーション・ICA", row, group_key="motion")
        row = self._add_checkbox(frame, "motion_analysis", "モーション解析を実行", row, default=True, group="motion")
        row = self._add_text_input(frame, "motion_roi", "ROI固定 (x,y,w,h) 空=手動", row, default="")
        row = self._add_number_input(frame, "motion_percentile", "ノイズ閾値パーセンタイル", row, default=75.0)
        row = self._add_number_input(frame, "motion_expand_sec", "マスク拡張 (秒)", row, default=0.1)
        
        # ICA設定
        row = self._add_checkbox(frame, "ica_enabled", "ICAアーティファクト除去", row, default=True, group="motion")
        row = self._add_number_input(frame, "ica_noise_ratio_threshold", "ICA除去閾値", row, default=1.5)
        row = self._add_number_input(frame, "ica_max_remove", "最大除去成分数", row, default=4, is_int=True)
    
    # ================================================================
    # タブ3: 解析
    # ================================================================
    def _build_tab_analysis(self, parent):
        frame = self._create_scrollable_frame(parent)
        row = 0
        
        # 周波数帯域
        row = self._add_section_header(frame, "📊 周波数帯域", row)
        self.band_editor = BandEditorFrame(frame)
        self.band_editor.grid(row=row, column=0, columnspan=3, sticky='ew', padx=5, pady=5)
        row += 1
        
        # FFT表示設定
        row = self._add_section_header(frame, "📉 FFT比較プロット", row)
        row = self._add_number_input(frame, "fft_freq_max", "FFT表示最大周波数 (Hz)", row, default=300.0)
        
        # パワースペクトル表示設定（独立）
        row = self._add_section_header(frame, "📈 パワースペクトル表示", row)
        row = self._add_number_input(frame, "power_freq_min", "表示最小周波数 (Hz)", row, default=0.5)
        row = self._add_number_input(frame, "power_freq_max", "表示最大周波数 (Hz)", row, default=100.0)
        
        # ウェーブレット解析
        row = self._add_section_header(frame, "🌊 ウェーブレット解析", row, group_key="wavelet")
        row = self._add_checkbox(frame, "wavelet_enabled", "ウェーブレット解析を実行", row, default=False, group="wavelet")
        row = self._add_number_input(frame, "wavelet_start", "開始時刻 (秒, 0=最初)", row, default=0.0)
        row = self._add_number_input(frame, "wavelet_end", "終了時刻 (秒, 0=最後)", row, default=100.0)
        row = self._add_number_input(frame, "wavelet_freq_min", "最小周波数 (Hz)", row, default=1.0)
        row = self._add_number_input(frame, "wavelet_freq_max", "最大周波数 (Hz)", row, default=100.0)
        row = self._add_number_input(frame, "wavelet_n_freqs", "周波数分割数", row, default=50, is_int=True)
        row = self._add_checkbox(frame, "wavelet_single", "単一チャンネル表示", row, default=True, group="wavelet")
        row = self._add_checkbox(frame, "wavelet_all", "全チャンネル表示", row, default=True, group="wavelet")
        row = self._add_number_input(frame, "wavelet_channel", "表示チャンネル", row, default=0, is_int=True)
    
    # ================================================================
    # タブ4: 出力
    # ================================================================
    def _build_tab_output(self, parent):
        frame = self._create_scrollable_frame(parent)
        row = 0
        
        # === 基本設定 ===
        row = self._add_section_header(frame, "🖼️ 表示・保存", row)
        row = self._add_checkbox(frame, "show_plots", "プロットを画面に表示", row, default=True)
        row = self._add_checkbox(frame, "save_plots", "プロットを画像保存", row, default=True)
        row = self._add_number_input(frame, "plot_t_start", "表示範囲 開始 (秒)", row, default=0.0)
        row = self._add_number_input(frame, "plot_t_end", "表示範囲 終了 (秒)", row, default=100.0)
        
        # === 出力するプロット（折りたたみ的に「詳細設定」にまとめる案）===
        row = self._add_section_header(frame, "📊 プロット選択", row, group_key="plots")
        row = self._add_checkbox(frame, "processing_overview", "処理概要", row, default=True, group="plots")
        row = self._add_checkbox(frame, "power_analysis", "パワー解析", row, default=True, group="plots")
        row = self._add_checkbox(frame, "channel_heatmap", "チャンネルヒートマップ", row, default=True, group="plots")
        row = self._add_checkbox(frame, "ica_components", "ICA成分", row, default=True, group="plots")
        row = self._add_checkbox(frame, "fft_comparison", "FFT比較", row, default=False, group="plots")  # デフォルトOFF
        row = self._add_checkbox(frame, "lfp_regions", "全チャンネル波形", row, default=False, group="plots")  # デフォルトOFF
        row = self._add_checkbox(frame, "edge_check", "端部効果確認", row, default=False, group="plots")
        
        # === データ保存 ===
        row = self._add_section_header(frame, "💾 データ保存", row, group_key="save_data")
        row = self._add_checkbox(frame, "save_summary_csv", "サマリーCSV", row, default=True, group="save_data")
        row = self._add_checkbox(frame, "save_channel_csv", "チャンネル別CSV", row, default=True, group="save_data")
        row = self._add_checkbox(frame, "save_results_npz", "解析結果NPZ", row, default=False, group="save_data")  # 大きいからデフォルトOFF
        row = self._add_checkbox(frame, "save_processed_npz", "処理済みLFP NPZ", row, default=False, group="save_data")
        
        # === 動画（あまり使わないから最後に）===
        row = self._add_section_header(frame, "🎬 同期動画", row)
        row = self._add_checkbox(frame, "create_sync_video", "同期動画作成", row, default=False)
        row = self._add_number_input(frame, "sync_video_start", "開始 (秒)", row, default=0.0)
        row = self._add_number_input(frame, "sync_video_end", "終了 (秒)", row, default=0.0)
        
        # === その他 ===
        row = self._add_section_header(frame, "🎬 進捗", row)
        row = self._add_checkbox(frame, "verbose", "進捗を表示", row, default=True)
        
    # ================================================================
    # ヘルパーメソッド
    # ================================================================
    def _create_scrollable_frame(self, parent):
        """スクロール可能なフレームを作成"""
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # マウスホイール
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        return scrollable_frame
    
    def _add_section_header(self, parent, text, row, group_key=None):
        """セクションヘッダー"""
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(15, 5))
        
        ttk.Label(frame, text=text, font=("", 10, "bold")).pack(side="left")
        
        if group_key:
            var = tk.BooleanVar(value=True)
            self.group_vars[group_key] = var
            cb = ttk.Checkbutton(frame, text="全選択", variable=var,
                                  command=lambda g=group_key: self._toggle_group(g))
            cb.pack(side="right", padx=10)
        
        ttk.Separator(parent, orient="horizontal").grid(row=row+1, column=0, columnspan=3, sticky="ew")
        return row + 2
    
    def _add_checkbox(self, parent, key, label, row, default=True, group=None):
        """チェックボックス"""
        var = tk.BooleanVar(value=default)
        self.vars[key] = {"var": var, "type": "bool", "default": default, "group": group}
        
        cb = ttk.Checkbutton(parent, text=label, variable=var)
        cb.grid(row=row, column=0, columnspan=2, sticky="w", padx=20, pady=2)
        return row + 1
    
    def _add_combo(self, parent, key, label, row, options, default=None):
        """コンボボックス（ドロップダウン）"""
        if default is None:
            default = options[0]
        var = tk.StringVar(value=default)
        self.vars[key] = {"var": var, "type": "str", "default": default}
        
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=20, pady=2)
        combo = ttk.Combobox(parent, textvariable=var, values=options, state='readonly', width=12)
        combo.grid(row=row, column=1, sticky="w", pady=2)
        return row + 1
    
    def _add_number_input(self, parent, key, label, row, default=0.0, is_int=False):
        """数値入力"""
        var = tk.StringVar(value=str(default))
        self.vars[key] = {"var": var, "type": "int" if is_int else "float", "default": default}
        
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=20, pady=2)
        entry = ttk.Entry(parent, textvariable=var, width=10)
        entry.grid(row=row, column=1, sticky="w", pady=2)
        return row + 1
    
    def _add_text_input(self, parent, key, label, row, default=""):
        """テキスト入力"""
        var = tk.StringVar(value=default)
        self.vars[key] = {"var": var, "type": "str", "default": default}
        
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=20, pady=2)
        entry = ttk.Entry(parent, textvariable=var, width=20)
        entry.grid(row=row, column=1, sticky="w", pady=2)
        return row + 1
    
    def _add_file_selector(self, parent, key, label, row, is_dir=False):
        """ファイル/フォルダ選択"""
        var = tk.StringVar(value="")
        self.vars[key] = {"var": var, "type": "path", "default": ""}
        
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=20, pady=2)
        
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=1, columnspan=2, sticky="ew", pady=2)
        
        entry = ttk.Entry(frame, textvariable=var, width=35)
        entry.pack(side="left", fill="x", expand=True)
        
        if is_dir:
            cmd = lambda: var.set(filedialog.askdirectory() or var.get())
        else:
            cmd = lambda: var.set(filedialog.askopenfilename(filetypes=[("PLX files", "*.plx")]) or var.get())
        
        ttk.Button(frame, text="参照", command=cmd, width=5).pack(side="left", padx=3)
        return row + 1
    
    def _toggle_group(self, group_key):
        """グループ全選択/全解除"""
        state = self.group_vars[group_key].get()
        for key, info in self.vars.items():
            if info.get("group") == group_key and info["type"] == "bool":
                info["var"].set(state)
    
    def _get_config(self) -> PipelineConfig:
        """GUIの値からPipelineConfigを生成"""
        kwargs = {}
        
        for key, info in self.vars.items():
            var = info["var"]
            vtype = info["type"]
            
            try:
                if vtype == "bool":
                    kwargs[key] = var.get()
                elif vtype == "int":
                    kwargs[key] = int(var.get())
                elif vtype == "float":
                    kwargs[key] = float(var.get())
                elif vtype == "str":
                    kwargs[key] = var.get()
                elif vtype == "path":
                    kwargs[key] = var.get()
            except ValueError:
                kwargs[key] = info["default"]
        
        # 帯域設定
        kwargs['bands'] = self.band_editor.get_bands()
        
        # 特殊処理
        if kwargs.get("manual_bad_channels"):
            try:
                kwargs["manual_bad_channels"] = [int(x.strip()) for x in kwargs["manual_bad_channels"].split(",") if x.strip()]
            except:
                kwargs["manual_bad_channels"] = []
        else:
            kwargs["manual_bad_channels"] = []
        
        if kwargs.get("motion_roi"):
            try:
                parts = [int(x.strip()) for x in kwargs["motion_roi"].split(",")]
                kwargs["motion_roi"] = tuple(parts) if len(parts) == 4 else None
            except:
                kwargs["motion_roi"] = None
        else:
            kwargs["motion_roi"] = None
        
        # 0 → None
        for key in ["plot_t_start", "plot_t_end", "sync_video_start", "sync_video_end", 
                    "wavelet_start", "wavelet_end"]:
            if kwargs.get(key, 0) == 0:
                kwargs[key] = None
        
        # FIRタップ数: 0 → None（自動計算）
        if kwargs.get("filter_fir_numtaps", 0) == 0:
            kwargs["filter_fir_numtaps"] = None
        
        return PipelineConfig(**kwargs)
    
    def _save_config(self):
        """設定をJSONに保存"""
        config_dict = {}
        for key, info in self.vars.items():
            var = info["var"]
            config_dict[key] = var.get()
        
        config_dict['bands'] = self.band_editor.get_bands_for_json()
        
        try:
            with open(CONFIG_SAVE_PATH, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            messagebox.showinfo("保存完了", f"設定を保存しました")
        except Exception as e:
            messagebox.showerror("エラー", f"保存に失敗しました:\n{e}")
    
    def _load_last_config(self):
        """前回の設定を読み込み"""
        if not os.path.exists(CONFIG_SAVE_PATH):
            return
        
        try:
            with open(CONFIG_SAVE_PATH, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            
            for key, value in config_dict.items():
                if key in self.vars:
                    self.vars[key]["var"].set(value)
            
            if 'bands' in config_dict and self.band_editor:
                self.band_editor.set_bands_from_json(config_dict['bands'])
            
            self._update_group_states()
            
        except Exception as e:
            print(f"設定読み込みエラー: {e}")
    
    def _reset_to_default(self):
        """デフォルト値に戻す"""
        for key, info in self.vars.items():
            info["var"].set(info["default"])
        
        if self.band_editor:
            self.band_editor.set_bands(BandEditorFrame.PRESETS['Standard'])
        
        self._update_group_states()
    
    def _update_group_states(self):
        """グループ状態を更新"""
        for group_key in self.group_vars:
            group_items = [info for info in self.vars.values() 
                          if info.get("group") == group_key and info["type"] == "bool"]
            if group_items:
                all_checked = all(info["var"].get() for info in group_items)
                self.group_vars[group_key].set(all_checked)
    
    def _run_pipeline(self):
        """パイプライン実行"""
        try:
            config = self._get_config()
            # self._save_config()
            
            bands_str = ", ".join([f"{k}({v[0]}-{v[1]})" for k, v in config.bands.items()])
            
            # フィルタ情報
            if config.filter_type.lower() == 'fir':
                taps_str = "自動" if config.filter_fir_numtaps is None else str(config.filter_fir_numtaps)
                filter_info = f"{config.filter_lowcut}-{config.filter_highcut}Hz [FIR taps={taps_str}]"
            else:
                filter_info = f"{config.filter_lowcut}-{config.filter_highcut}Hz [IIR order={config.filter_order}]"
            
            if config.notch_enabled:
                filter_info += f", ノッチ{config.notch_freq}Hz"
            
            msg = f"実行しますか?\n\n"
            msg += f"📁 {os.path.basename(config.plx_file) or '(未選択)'}\n"
            msg += f"🔧 フィルタ: {filter_info}\n"
            msg += f"🔇 高調波除去: {'ON' if config.harmonic_removal_enabled else 'OFF'}\n"
            msg += f"🧠 ICA: {'ON' if config.ica_enabled else 'OFF'}\n"
            msg += f"📊 帯域: {bands_str}\n"
            msg += f"📈 パワー表示: {config.power_freq_min}-{config.power_freq_max}Hz\n"
            
            if not messagebox.askyesno("確認", msg):
                return
            
            self.root.destroy()
            
            print("\n" + "="*60)
            print("パイプライン実行開始")
            print("="*60)
            
            results = run_pipeline(config)
            
            print("\n" + "="*60)
            print("完了!")
            print("="*60)
            
        except Exception as e:
            messagebox.showerror("エラー", f"実行エラー:\n{e}")
            import traceback
            traceback.print_exc()
    
    def run(self):
        """GUIを起動"""
        self.root.mainloop()


def launch_gui():
    app = ConfigGUI()
    app.run()


if __name__ == "__main__":
    launch_gui()