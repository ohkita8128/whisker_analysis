# Neuronexus Analysis v7 - 設計書

## 概要

v6の解析パイプライン（PLX読込 → LFPフィルタ → スパイクソーティング → 位相ロック解析）を
**1つの統合GUI**で操作できるようにする。Suite2P/MATLABツールボックス風の設計。

### v6からの変更方針
- **バックエンドは流用**: data_loader, lfp_pipeline, lfp_processing, spike_sorting,
  kilosort_wrapper, phase_locking, phase_plotting_v6, spike_processing 等はコピーしてそのまま使う
- **プロットは一切変更しない**: 既存の描画関数・レイアウト・軸設定をそのまま維持
- **GUIの「器」だけ作り直す**: 各ステップの設定UIをFrameとして切り出し、統合GUIに組み込む
- **ProjectStateで一元管理**: 設定・パス・ステップ状態を1ファイルに統合

---

## アーキテクチャ

### 全体構成図

```
┌─────────────────────────────────────────────────────────────┐
│  Neuronexus Analysis v7                              [─][□][×] │
├──────────┬──────────────────────────────────────────────────┤
│          │                                                  │
│ SIDEBAR  │  MAIN AREA                                       │
│          │  (選択中ステップのUIパネル)                       │
│ [0]Load  │                                                  │
│ [1]LFP   │  ┌─────────────┬────────────────────┐            │
│ [2]Sort  │  │ 設定パネル  │  プレビュー/結果   │            │
│ [3]Phase │  │ (左)        │  (右)              │            │
│          │  └─────────────┴────────────────────┘            │
│──────────│                                                  │
│ PROJECT  │                                                  │
│ File:... │                                                  │
│ Out:...  │                                                  │
│──────────│                                                  │
│ SETTINGS │                                                  │
│ Font:  8 │                                                  │
│          │                                                  │
├──────────┴──────────────────────────────────────────────────┤
│ [ステータスバー]                                            │
└─────────────────────────────────────────────────────────────┘
```

### データフロー

```
ProjectState (project_state.json)
  │
  ├→ Step 0: Load
  │    data_loader.load_plx(plx_path) → PlxData
  │    → plx_data保持（メモリ）
  │    → step_status["load"] = "completed"
  │
  ├→ Step 1: LFP Filter
  │    LfpConfig (ProjectState内) → lfp_pipeline.run_lfp_pipeline()
  │    → lfp_results保持（メモリ）+ .npz自動保存
  │    → step_status["lfp"] = "completed"
  │
  ├→ Step 2: Spike Sort
  │    SortingConfig (ProjectState内) → spike_sorting / kilosort
  │    → spike_results保持（メモリ）+ .pkl自動保存
  │    → step_status["spike_sort"] = "completed"
  │
  └→ Step 3: Phase Locking
       PhaseConfig (ProjectState内) → phase_locking解析
       → phase_results保持（メモリ）+ .pkl/.pdf保存
       → step_status["phase"] = "completed"
```

---

## コア設計

### 1. ProjectState (project_state.py)

全設定・全パス・全ステップ状態を1ファイルで管理するdataclass。

```python
@dataclass
class ProjectState:
    # メタ
    version: str = "7.0"
    created: str = ""
    last_modified: str = ""

    # パス（プロジェクト単位で統一）
    plx_file: str = ""
    output_dir: str = ""

    # ステップ完了状態
    step_status: Dict[str, str] = field(default_factory=lambda: {
        "load": "pending",
        "lfp": "pending",
        "spike_sort": "pending",
        "phase": "pending",
    })

    # 中間データファイルパス（output_dir相対）
    data_files: Dict[str, str] = field(default_factory=dict)

    # 各ステップの設定（現行の個別JSONの中身をここに統合）
    lfp_config: Dict = field(default_factory=dict)
    spike_config: Dict = field(default_factory=dict)
    phase_config: Dict = field(default_factory=dict)

    # 表示設定
    font_size_gui: int = 9       # GUIウィジェットの文字サイズ
    font_size_plot: int = 8      # プロット出力の文字サイズ

    def save(self, path: str): ...
    def load(cls, path: str) -> 'ProjectState': ...
```

保存先: `{output_dir}/{basename}_project.json`

### 2. MainGUI (main_gui.py)

統合GUIのメインウィンドウ。

**責務:**
- ウィンドウ管理（サイドバー + メインエリア + ステータスバー）
- ステップ切替（サイドバーのボタンクリック → メインエリアのFrame切替）
- ProjectStateの保持・保存・読込
- ステップ間のデータ受け渡し（メモリ上のplx_data, lfp_results等）
- フォントサイズ設定の一括適用

**サイドバー:**
- Step 0-3 のボタン（完了=緑、実行中=黄、未完了=グレー）
- プロジェクト情報（PLXファイル名、出力ディレクトリ）
- 表示設定（GUIフォントサイズ、プロットフォントサイズ）

**ステップ遷移ルール:**
- Step N は Step N-1 が completed でないと実行不可（ボタンは押せるが「実行」はブロック）
- 完了済みステップに戻って再実行可能 → 以降のステップは "pending" にリセット

### 3. ステップパネル (step_*.py)

各ステップのUI。`ttk.Frame` のサブクラスとして実装。

```python
class StepPanel(ttk.Frame):
    """全ステップパネルの基底クラス"""
    def __init__(self, parent, app: 'MainGUI'):
        self.app = app  # MainGUIへの参照（データ・設定アクセス用）

    def on_show(self):
        """パネルが表示されたとき（データの再読込等）"""

    def on_hide(self):
        """パネルが非表示になるとき（設定の一時保存等）"""

    def get_config(self) -> dict:
        """現在のGUI設定値を辞書で返す"""

    def set_config(self, config: dict):
        """設定値をGUIに反映"""
```

**Step 0: LoadPanel (step_load.py)**
- PLXファイル選択（ファイルダイアログ）
- 出力ディレクトリ選択
- ファイル情報プレビュー（ch数、fs、記録時間、イベント数）
- 「読み込み」ボタン

**Step 1: LfpPanel (step_lfp.py)**
- 現 lfp_filter_gui.py の設定パネルを移植
  - フィルタ設定（バンドパス、ノッチ、ハーモニック除去）
  - 帯域エディタ（プリセット付き）
  - ICA設定
  - bad channel検出設定
  - 条件設定（セッション数、刺激数、baseline/stim/post区間）
- プレビューエリア: FFT before/after、PSD、フィルタ応答
- 「実行」ボタン

**Step 2: SpikeSortPanel (step_spike_sort.py)**
- 現 spike_sort_gui.py の設定+キュレーションUIを移植
  - 手法選択（GMM / KiloSort4）
  - フィルタ・検出・クラスタリングパラメータ
  - チャンネルリスト + 波形表示
  - PCA散布図 + ISIヒストグラム
  - ユニット品質テーブル
  - 手動キュレーション（マージ/削除/MUA指定）
- 「全チャンネル実行」「選択ch実行」ボタン

**Step 3: PhasePanel (step_phase.py)**
- 現 phase_gui.py の設定+プレビューを移植
  - スパイクソース選択（PLX / sorted）
  - 品質フィルタ（Sortedのみ / 全ユニット / MUAのみ）
  - 帯域選択（LFP設定から自動取得）
  - 解析パラメータ（最小スパイク数、アーティファクト除外窓）
  - 条件別解析チェック
- プレビュー: PPCヒートマップ + 極座標
- グランドサマリー出力ボタン
- **将来: STAインタラクティブ表示（スパイクchドロップダウン）**

### 4. フォントサイズ管理 (font_manager.py)

**GUI フォントサイズ:**
- ttk.Style() でグローバルに設定
- サイドバーのスピンボックスで変更 → 全ウィジェットに即時反映
- ProjectStateに保存

**プロット フォントサイズ:**
- matplotlib.rcParams で一括制御
- `font.size`, `axes.titlesize`, `axes.labelsize`, `xtick.labelsize`, `ytick.labelsize`
- 既存プロット関数内のハードコード fontsize は rcParams のデフォルト値で動作
  → ユーザーが rcParams を変更すると、明示指定のないラベルに反映
  → 既存コード内で fontsize= を明示しているものは変わらない（既存プロット保護）
- 追加で「プロット文字サイズ倍率」(0.5x - 2.0x) スライダーを設け、
  各プロット関数呼び出し前に rcParams を一時的にスケーリング

---

## ファイル構成

```
Neuronexus_analysis_v7/
│
│  # --- v7 新規 ---
├── main_gui.py            # 統合GUIエントリポイント
├── project_state.py       # ProjectState dataclass + save/load
├── step_load.py           # Step 0: PLX読込パネル
├── step_lfp.py            # Step 1: LFPフィルタパネル
├── step_spike_sort.py     # Step 2: スパイクソーティングパネル
├── step_phase.py          # Step 3: 位相ロック解析パネル
├── font_manager.py        # フォントサイズ一括管理
│
│  # --- v6 からコピー（変更なし）---
├── data_loader.py
├── lfp_pipeline.py
├── lfp_processing.py
├── lfp_plotting.py
├── spike_sorting.py
├── kilosort_wrapper.py
├── spike_processing.py
├── spike_plotting.py
├── phase_locking.py
├── phase_plotting_v6.py
├── phase_plotting.py
├── saving.py
├── numpy_compat.py
├── waveform_browser.py
└── get_path.py
```

---

## 実装順序

### Phase 1: 骨格
1. `project_state.py` — ProjectState dataclass + JSON save/load
2. `font_manager.py` — フォントサイズ管理
3. `main_gui.py` — メインウィンドウ + サイドバー + ステップ切替の骨格
4. v6バックエンドファイルをコピー

### Phase 2: Step 0 (Load)
5. `step_load.py` — PLX読込パネル（ファイル選択、情報表示、読込実行）

### Phase 3: Step 1 (LFP)
6. `step_lfp.py` — lfp_filter_gui.py から設定UI移植 + lfp_pipeline実行

### Phase 4: Step 2 (Spike Sort)
7. `step_spike_sort.py` — spike_sort_gui.py から設定+キュレーションUI移植

### Phase 5: Step 3 (Phase)
8. `step_phase.py` — phase_gui.py からUI移植 + グランドサマリー出力

### Phase 6: 仕上げ
9. ステップ間データ受け渡しの統合テスト
10. フォントサイズ調整の動作確認
11. ProjectState保存/読込の途中再開テスト

---

## 設計メモ

### ステップ再実行時の挙動
- Step 1 を再実行 → Step 2, 3 は "pending" にリセット（依存データが変わるため）
- ただし設定値はリセットしない（ユーザーが調整済みの設定を保持）
- 確認ダイアログ: 「Step 1を再実行すると、Step 2以降の結果がリセットされます。続行しますか？」

### 中間データの保存形式
- LFP結果: `.npz` (numpy配列)
- スパイク結果: `.pkl` (pickle、ChannelSortResult含むため)
- 位相結果: `.pkl` + PDF/PNG

### 既存プロットの保護方針
- phase_plotting_v6.py 等のプロット関数は **一切変更しない**
- フォントサイズ調整は matplotlib.rcParams 経由で間接的に影響
- 各プロット関数内で fontsize= を明示指定しているラベルはそのまま維持される
- 「プロット文字サイズ倍率」機能は、rcParams を一時変更してプロット関数を呼び、
  呼び出し後に元に戻す context manager で実装

### スレッド/プログレス
- 長時間処理（LFP pipeline、スパイクソーティング）はスレッドで実行
- ステータスバーにプログレス表示
- 実行中はサイドバーのステップ切替をブロック
