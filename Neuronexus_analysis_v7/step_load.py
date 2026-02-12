"""
step_load.py - Step 0: PLXファイル読込パネル

PLXファイル選択、出力ディレクトリ指定、ファイル情報表示、読込実行。
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading

from main_gui import StepPanel


class Panel(StepPanel):
    """PLX読込パネル"""

    def __init__(self, parent, app):
        super().__init__(parent, app)
        self._build_ui()

    def _build_ui(self):
        # 左: 設定、右: 情報表示
        pane = ttk.PanedWindow(self, orient='horizontal')
        pane.pack(fill='both', expand=True, padx=5, pady=5)

        left = ttk.Frame(pane)
        pane.add(left, weight=0)

        right = ttk.Frame(pane)
        pane.add(right, weight=1)

        # === 左: ファイル選択 ===
        ttk.Label(left, text="Step 0: Data Load",
                  style='Heading.TLabel').pack(padx=10, pady=(10, 5), anchor='w')

        # PLXファイル
        f_plx = ttk.LabelFrame(left, text="PLX File")
        f_plx.pack(fill='x', padx=10, pady=5)

        self._plx_var = tk.StringVar(master=self.winfo_toplevel(), value=self.state.plx_file)
        ttk.Entry(f_plx, textvariable=self._plx_var, width=50).pack(
            fill='x', padx=5, pady=(5, 0))
        ttk.Button(f_plx, text="Browse...",
                   command=self._browse_plx).pack(padx=5, pady=5)

        # 出力ディレクトリ
        f_out = ttk.LabelFrame(left, text="Output Directory")
        f_out.pack(fill='x', padx=10, pady=5)

        self._out_var = tk.StringVar(master=self.winfo_toplevel(), value=self.state.output_dir)
        ttk.Entry(f_out, textvariable=self._out_var, width=50).pack(
            fill='x', padx=5, pady=(5, 0))
        ttk.Button(f_out, text="Browse...",
                   command=self._browse_output).pack(padx=5, pady=5)

        # 読込ボタン
        ttk.Button(left, text="読み込み実行",
                   command=self._run_load,
                   style='Run.TButton').pack(padx=10, pady=15)

        # === 右: ファイル情報 ===
        ttk.Label(right, text="File Information",
                  style='Heading.TLabel').pack(padx=10, pady=(10, 5), anchor='w')

        self._info_text = tk.Text(right, wrap='word', state='disabled',
                                  font=('Consolas', 9))
        self._info_text.pack(fill='both', expand=True, padx=10, pady=5)

    def on_show(self):
        self._plx_var.set(self.state.plx_file)
        self._out_var.set(self.state.output_dir)
        if self.app.plx_data is not None:
            self._show_plx_info()

    def get_config(self) -> dict:
        return {
            'plx_file': self._plx_var.get(),
            'output_dir': self._out_var.get(),
        }

    # ===========================================================
    # ファイル選択
    # ===========================================================

    def _browse_plx(self):
        path = filedialog.askopenfilename(
            title="PLXファイルを選択",
            filetypes=[("Plexon files", "*.plx"), ("All files", "*.*")])
        if path:
            self._plx_var.set(path)
            if not self._out_var.get():
                self._out_var.set(os.path.dirname(path))

    def _browse_output(self):
        d = filedialog.askdirectory(title="出力ディレクトリを選択")
        if d:
            self._out_var.set(d)

    # ===========================================================
    # 読込実行
    # ===========================================================

    def _run_load(self):
        plx_path = self._plx_var.get().strip()
        out_dir = self._out_var.get().strip()

        if not plx_path or not os.path.exists(plx_path):
            messagebox.showwarning("Warning", "PLXファイルが見つかりません")
            return

        if not out_dir:
            out_dir = os.path.dirname(plx_path)

        # 下流ステップリセット確認
        if not self.app.reset_downstream("load"):
            return

        self.state.plx_file = plx_path
        self.state.output_dir = out_dir

        self.app.set_status("PLX読込中...")

        # スレッドで読込
        def _load():
            try:
                from data_loader import load_plx
                plx_data = load_plx(plx_path)
                plx_data.output_dir = out_dir

                self.app.plx_data = plx_data
                self.app.lfp_results = None
                self.app.spike_results = None
                self.app.phase_results = None

                self.after(0, self._on_load_done)
            except Exception as e:
                self.after(0, lambda: self._on_load_error(str(e)))

        t = threading.Thread(target=_load, daemon=True)
        t.start()

    def _on_load_done(self):
        self.app.complete_step("load")
        self._show_plx_info()
        self.app.set_status("PLX読込完了")

    def _on_load_error(self, msg: str):
        messagebox.showerror("Error", f"読込エラー:\n{msg}")
        self.app.set_status("読込エラー")

    # ===========================================================
    # 情報表示
    # ===========================================================

    def _show_plx_info(self):
        pd = self.app.plx_data
        if pd is None:
            return

        lines = []
        lines.append(f"File: {pd.filepath}")
        lines.append(f"Basename: {pd.basename}")
        lines.append(f"")
        lines.append(f"=== LFP ===")
        if pd.lfp_raw is not None:
            shape = pd.lfp_raw.shape
            dur = shape[0] / pd.lfp_fs if pd.lfp_fs else 0
            lines.append(f"  Channels: {shape[1] if len(shape) > 1 else 1}")
            lines.append(f"  Samples: {shape[0]}")
            lines.append(f"  Fs: {pd.lfp_fs} Hz")
            lines.append(f"  Duration: {dur:.1f} sec")
        else:
            lines.append("  (not available)")

        lines.append(f"")
        lines.append(f"=== Wideband ===")
        if pd.wideband_raw is not None:
            shape = pd.wideband_raw.shape
            dur = shape[0] / pd.wideband_fs if pd.wideband_fs else 0
            lines.append(f"  Channels: {shape[1] if len(shape) > 1 else 1}")
            lines.append(f"  Samples: {shape[0]}")
            lines.append(f"  Fs: {pd.wideband_fs} Hz")
            lines.append(f"  Duration: {dur:.1f} sec")
        else:
            lines.append("  (not available)")

        lines.append(f"")
        lines.append(f"=== Events ===")
        n_stim = len(pd.stim_times) if pd.stim_times is not None else 0
        n_frame = len(pd.frame_times) if pd.frame_times is not None else 0
        lines.append(f"  Stim events: {n_stim}")
        lines.append(f"  Frame events: {n_frame}")

        if hasattr(pd, 'trim_start') and pd.trim_start is not None:
            lines.append(f"  Trim range: {pd.trim_start:.2f} - {pd.trim_end:.2f} sec")

        lines.append(f"")
        lines.append(f"=== Channel Order ===")
        if pd.original_ch_numbers:
            lines.append(f"  {pd.original_ch_numbers}")

        if hasattr(pd, 'has_sorted_spikes'):
            lines.append(f"")
            lines.append(f"=== PLX Sorted Spikes ===")
            lines.append(f"  Available: {pd.has_sorted_spikes}")

        self._info_text.config(state='normal')
        self._info_text.delete('1.0', 'end')
        self._info_text.insert('1.0', '\n'.join(lines))
        self._info_text.config(state='disabled')
