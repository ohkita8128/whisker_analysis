"""
main_gui.py - Neuronexus Analysis v7 統合GUI

Suite2P風の統合インターフェース。
左サイドバーでステップ選択、メインエリアで設定・プレビュー・実行。
全設定はProjectStateで一元管理。
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys

from project_state import ProjectState, STEP_NAMES
from font_manager import apply_gui_font_size, apply_plot_font_size


# ステップ定義: (内部名, 表示名, モジュール名)
STEPS = [
    ("load",       "0: Load",       "step_load"),
    ("lfp",        "1: LFP Filter", "step_lfp"),
    ("spike_sort", "2: Spike Sort", "step_spike_sort"),
    ("phase",      "3: Phase Lock", "step_phase"),
]

# ステップ状態の色
STATUS_COLORS = {
    "pending":   "#999999",
    "completed": "#27ae60",
}


class MainGUI:
    """統合GUIメインウィンドウ"""

    def __init__(self):
        self.state = ProjectState()

        # メモリ上のデータ（ステップ間受け渡し用）
        self.plx_data = None       # Step 0 の出力
        self.lfp_results = None    # Step 1 の出力
        self.spike_results = None  # Step 2 の出力
        self.phase_results = None  # Step 3 の出力

        # ステップパネルのキャッシュ
        self._panels = {}
        self._current_step = None

        self._build_window()
        self._build_sidebar()
        self._build_main_area()
        self._build_statusbar()

        # 初期フォント適用
        apply_gui_font_size(self.root, self.state.font_size_gui)
        apply_plot_font_size(self.state.font_size_plot, self.state.plot_font_scale)

        # 初期ステップ表示
        self._switch_step("load")

    # ===========================================================
    # ウィンドウ構築
    # ===========================================================

    def _build_window(self):
        self.root = tk.Tk()
        self.root.title("Neuronexus Analysis v7")
        self.root.geometry("1400x900")
        self.root.minsize(1000, 600)

        # メインの水平分割
        self._outer_pane = ttk.PanedWindow(self.root, orient='horizontal')
        self._outer_pane.pack(fill='both', expand=True)

    def _build_sidebar(self):
        """左サイドバー: ステップボタン + プロジェクト情報 + 表示設定"""
        sidebar = ttk.Frame(self._outer_pane, width=200)
        self._outer_pane.add(sidebar, weight=0)

        # --- ステップボタン ---
        step_frame = ttk.LabelFrame(sidebar, text="Steps")
        step_frame.pack(fill='x', padx=5, pady=(5, 3))

        self._step_buttons = {}
        self._step_indicators = {}
        for step_name, display_name, _ in STEPS:
            f = ttk.Frame(step_frame)
            f.pack(fill='x', padx=3, pady=2)

            indicator = tk.Canvas(f, width=12, height=12,
                                  highlightthickness=0)
            indicator.pack(side='left', padx=(3, 5))
            self._step_indicators[step_name] = indicator

            btn = ttk.Button(f, text=display_name,
                             command=lambda s=step_name: self._switch_step(s),
                             style='SidebarStep.TButton')
            btn.pack(fill='x', expand=True)
            self._step_buttons[step_name] = btn

        # --- プロジェクト情報 ---
        proj_frame = ttk.LabelFrame(sidebar, text="Project")
        proj_frame.pack(fill='x', padx=5, pady=3)

        self._proj_file_label = ttk.Label(proj_frame, text="(未読込)",
                                           wraplength=180)
        self._proj_file_label.pack(fill='x', padx=5, pady=2)

        btn_frame = ttk.Frame(proj_frame)
        btn_frame.pack(fill='x', padx=5, pady=3)
        ttk.Button(btn_frame, text="新規",
                   command=self._new_project).pack(side='left', padx=2)
        ttk.Button(btn_frame, text="開く",
                   command=self._open_project).pack(side='left', padx=2)
        ttk.Button(btn_frame, text="保存",
                   command=self._save_project).pack(side='left', padx=2)

        # --- 表示設定 ---
        disp_frame = ttk.LabelFrame(sidebar, text="Display")
        disp_frame.pack(fill='x', padx=5, pady=3)

        # GUIフォントサイズ
        f1 = ttk.Frame(disp_frame)
        f1.pack(fill='x', padx=5, pady=2)
        ttk.Label(f1, text="GUI Font:").pack(side='left')
        self._gui_font_var = tk.IntVar(master=self.root, value=self.state.font_size_gui)
        spin_gui = ttk.Spinbox(f1, from_=6, to=16, width=4,
                               textvariable=self._gui_font_var,
                               command=self._on_gui_font_change)
        spin_gui.pack(side='right')

        # プロットフォントサイズ
        f2 = ttk.Frame(disp_frame)
        f2.pack(fill='x', padx=5, pady=2)
        ttk.Label(f2, text="Plot Font:").pack(side='left')
        self._plot_font_var = tk.IntVar(master=self.root, value=self.state.font_size_plot)
        spin_plot = ttk.Spinbox(f2, from_=4, to=16, width=4,
                                textvariable=self._plot_font_var,
                                command=self._on_plot_font_change)
        spin_plot.pack(side='right')

        # プロットスケール
        f3 = ttk.Frame(disp_frame)
        f3.pack(fill='x', padx=5, pady=2)
        ttk.Label(f3, text="Plot Scale:").pack(side='left')
        self._plot_scale_var = tk.DoubleVar(master=self.root, value=self.state.plot_font_scale)
        scale = ttk.Scale(f3, from_=0.5, to=2.0, orient='horizontal',
                          variable=self._plot_scale_var,
                          command=self._on_plot_scale_change)
        scale.pack(side='right', fill='x', expand=True, padx=(5, 0))

        self._sidebar = sidebar

    def _build_main_area(self):
        """メインエリア: ステップパネルの表示領域"""
        self._main_frame = ttk.Frame(self._outer_pane)
        self._outer_pane.add(self._main_frame, weight=1)

    def _build_statusbar(self):
        """ステータスバー"""
        self._status_var = tk.StringVar(master=self.root, value="Ready")
        bar = ttk.Label(self.root, textvariable=self._status_var,
                        relief='sunken', anchor='w')
        bar.pack(fill='x', side='bottom')

    # ===========================================================
    # ステップ切替
    # ===========================================================

    def _switch_step(self, step_name: str):
        """メインエリアに表示するステップを切り替える"""
        # 現パネルを隠す
        if self._current_step and self._current_step in self._panels:
            panel = self._panels[self._current_step]
            panel.on_hide()
            panel.pack_forget()

        # パネルを取得（初回はインポートして生成）
        if step_name not in self._panels:
            self._panels[step_name] = self._create_panel(step_name)

        panel = self._panels[step_name]
        panel.pack(fill='both', expand=True)
        panel.on_show()
        self._current_step = step_name

        self._update_sidebar()
        self._update_status(f"Step: {step_name}")

    def _create_panel(self, step_name: str):
        """ステップパネルをインポートして生成"""
        for name, _, module_name in STEPS:
            if name == step_name:
                mod = __import__(module_name)
                panel_class = mod.Panel
                return panel_class(self._main_frame, self)
        raise ValueError(f"Unknown step: {step_name}")

    # ===========================================================
    # サイドバー更新
    # ===========================================================

    def _update_sidebar(self):
        """ステップインジケーターの色を更新"""
        for step_name, _, _ in STEPS:
            indicator = self._step_indicators[step_name]
            status = self.state.step_status.get(step_name, "pending")
            color = STATUS_COLORS.get(status, "#999999")
            indicator.delete("all")
            indicator.create_oval(1, 1, 11, 11, fill=color, outline=color)

        # プロジェクト情報
        if self.state.plx_file:
            txt = f"PLX: {os.path.basename(self.state.plx_file)}\nOut: {self.state.output_dir}"
        else:
            txt = "(未読込)"
        self._proj_file_label.config(text=txt)

    # ===========================================================
    # ステップ完了 / リセット
    # ===========================================================

    def complete_step(self, step_name: str):
        """ステップを完了にしてサイドバー更新"""
        self.state.complete_step(step_name)
        self._update_sidebar()
        self._save_project_auto()

    def reset_downstream(self, step_name: str):
        """指定ステップ以降をリセット（確認ダイアログ付き）"""
        idx = STEP_NAMES.index(step_name)
        downstream = STEP_NAMES[idx:]
        completed_downstream = [s for s in downstream
                                if self.state.is_step_completed(s)]
        if completed_downstream:
            names = ", ".join(completed_downstream)
            ok = messagebox.askyesno(
                "確認",
                f"以下のステップの結果がリセットされます:\n{names}\n\n続行しますか？")
            if not ok:
                return False
        self.state.reset_from_step(step_name)
        self._update_sidebar()
        return True

    # ===========================================================
    # プロジェクト管理
    # ===========================================================

    def _new_project(self):
        path = filedialog.askopenfilename(
            title="PLXファイルを選択",
            filetypes=[("Plexon files", "*.plx"), ("All files", "*.*")])
        if not path:
            return
        out_dir = filedialog.askdirectory(
            title="出力ディレクトリを選択",
            initialdir=os.path.dirname(path))
        if not out_dir:
            out_dir = os.path.dirname(path)

        self.state = ProjectState()
        self.state.new_project(path, out_dir)
        self.plx_data = None
        self.lfp_results = None
        self.spike_results = None
        self.phase_results = None

        self._update_sidebar()
        self._switch_step("load")
        self._update_status(f"新規プロジェクト: {os.path.basename(path)}")

    def _open_project(self):
        path = filedialog.askopenfilename(
            title="プロジェクトファイルを選択",
            filetypes=[("Project JSON", "*_project.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.state = ProjectState.load(path)
            self._gui_font_var.set(self.state.font_size_gui)
            self._plot_font_var.set(self.state.font_size_plot)
            self._plot_scale_var.set(self.state.plot_font_scale)
            apply_gui_font_size(self.root, self.state.font_size_gui)
            apply_plot_font_size(self.state.font_size_plot,
                                self.state.plot_font_scale)

            # パネルをリセット（設定を再読込させる）
            for panel in self._panels.values():
                panel.pack_forget()
            self._panels.clear()
            self._current_step = None

            self._update_sidebar()
            self._switch_step("load")
            self._update_status(f"プロジェクト読込: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Error", f"読込エラー:\n{e}")

    def _save_project(self):
        if self.state.project_file_path:
            self._save_current_panel_config()
            self.state.save()
            self._update_status(
                f"保存: {os.path.basename(self.state.project_file_path)}")
        else:
            messagebox.showwarning("Warning", "プロジェクトが未設定です")

    def _save_project_auto(self):
        """ステップ完了時等の自動保存"""
        if self.state.project_file_path:
            self._save_current_panel_config()
            self.state.save()

    def _save_current_panel_config(self):
        """現在のパネルの設定をProjectStateに保存"""
        if self._current_step and self._current_step in self._panels:
            panel = self._panels[self._current_step]
            config = panel.get_config()
            if self._current_step == "lfp":
                self.state.lfp_config = config
            elif self._current_step == "spike_sort":
                self.state.spike_config = config
            elif self._current_step == "phase":
                self.state.phase_config = config

    # ===========================================================
    # フォントサイズ変更
    # ===========================================================

    def _on_gui_font_change(self):
        size = self._gui_font_var.get()
        self.state.font_size_gui = size
        apply_gui_font_size(self.root, size)

    def _on_plot_font_change(self):
        size = self._plot_font_var.get()
        self.state.font_size_plot = size
        apply_plot_font_size(size, self.state.plot_font_scale)

    def _on_plot_scale_change(self, _val=None):
        scale = self._plot_scale_var.get()
        self.state.plot_font_scale = round(scale, 2)
        apply_plot_font_size(self.state.font_size_plot, scale)

    # ===========================================================
    # ステータスバー
    # ===========================================================

    def _update_status(self, msg: str):
        self._status_var.set(msg)

    def set_status(self, msg: str):
        """外部（パネル）からステータス更新"""
        self._status_var.set(msg)
        self.root.update_idletasks()

    # ===========================================================
    # 起動
    # ===========================================================

    def run(self):
        self.root.mainloop()


# ============================================================
# ステップパネル基底クラス
# ============================================================

class StepPanel(ttk.Frame):
    """全ステップパネルの基底クラス"""

    def __init__(self, parent, app: MainGUI):
        super().__init__(parent)
        self.app = app

    @property
    def state(self) -> ProjectState:
        return self.app.state

    def on_show(self):
        """パネルが表示されたとき"""
        pass

    def on_hide(self):
        """パネルが非表示になるとき"""
        pass

    def get_config(self) -> dict:
        """現在のGUI設定値を辞書で返す"""
        return {}

    def set_config(self, config: dict):
        """設定値をGUIに反映"""
        pass


# ============================================================
# エントリポイント
# ============================================================

def main():
    app = MainGUI()
    app.run()


if __name__ == '__main__':
    main()
