"""
project_state.py - プロジェクト状態の一元管理

全設定・パス・ステップ完了状態を1つのJSONファイルで管理。
"""
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, Any
from datetime import datetime


STEP_NAMES = ["load", "lfp", "spike_sort", "phase"]


@dataclass
class ProjectState:
    """プロジェクト全体の状態"""

    # メタ情報
    version: str = "7.0"
    created: str = ""
    last_modified: str = ""

    # パス
    plx_file: str = ""
    output_dir: str = ""

    # ステップ完了状態: "pending" / "completed"
    step_status: Dict[str, str] = field(default_factory=lambda: {
        "load": "pending",
        "lfp": "pending",
        "spike_sort": "pending",
        "phase": "pending",
    })

    # 中間データファイルパス（output_dir からの相対パス）
    data_files: Dict[str, str] = field(default_factory=dict)

    # 各ステップの設定
    lfp_config: Dict[str, Any] = field(default_factory=dict)
    spike_config: Dict[str, Any] = field(default_factory=dict)
    phase_config: Dict[str, Any] = field(default_factory=dict)

    # 表示設定
    font_size_gui: int = 9
    font_size_plot: int = 8
    plot_font_scale: float = 1.0  # プロット文字サイズ倍率 (0.5 - 2.0)

    @property
    def basename(self) -> str:
        if self.plx_file:
            return os.path.splitext(os.path.basename(self.plx_file))[0]
        return ""

    @property
    def project_file_path(self) -> str:
        if self.output_dir and self.basename:
            return os.path.join(self.output_dir, f"{self.basename}_project.json")
        return ""

    # -------------------------------------------------------
    # ステップ管理
    # -------------------------------------------------------

    def complete_step(self, step_name: str):
        """ステップを完了にする"""
        self.step_status[step_name] = "completed"
        self.last_modified = datetime.now().isoformat()

    def reset_from_step(self, step_name: str):
        """指定ステップ以降をpendingにリセット（設定値は保持）"""
        idx = STEP_NAMES.index(step_name)
        for name in STEP_NAMES[idx:]:
            self.step_status[name] = "pending"

    def is_step_ready(self, step_name: str) -> bool:
        """指定ステップが実行可能か（前ステップがcompleted）"""
        idx = STEP_NAMES.index(step_name)
        if idx == 0:
            return True
        prev = STEP_NAMES[idx - 1]
        return self.step_status[prev] == "completed"

    def is_step_completed(self, step_name: str) -> bool:
        return self.step_status.get(step_name) == "completed"

    # -------------------------------------------------------
    # 保存 / 読込
    # -------------------------------------------------------

    def save(self, path: str = ""):
        """JSONに保存"""
        path = path or self.project_file_path
        if not path:
            return
        self.last_modified = datetime.now().isoformat()
        data = asdict(self)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> 'ProjectState':
        """JSONから読込"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        state = cls()
        for key, value in data.items():
            if hasattr(state, key):
                setattr(state, key, value)
        return state

    def new_project(self, plx_file: str, output_dir: str = ""):
        """新規プロジェクト初期化"""
        self.plx_file = plx_file
        self.output_dir = output_dir or os.path.dirname(plx_file)
        self.created = datetime.now().isoformat()
        self.step_status = {name: "pending" for name in STEP_NAMES}
        self.data_files = {}
