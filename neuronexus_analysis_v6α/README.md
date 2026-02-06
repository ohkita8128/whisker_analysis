# Neuronexus Analysis Pipeline v6

NeuroNexus A1x16 プローブによる S1BF（バレル皮質）記録データの統合解析パイプライン。

## 全体フロー

```
PLX File
  │
  ├─→ [Step 1] データ読み込み (data_loader.py)
  │     └─→ RecordingSession (LFP + Wideband + Events)
  │
  ├─→ [Step 2] LFP処理 (processing.py)
  │     ├─→ バンドパスフィルタ (0.1-100Hz)
  │     ├─→ ノッチフィルタ (60Hz)
  │     ├─→ 高調波除去 (10Hz ピエゾ由来)
  │     ├─→ 悪チャンネル検出
  │     ├─→ モーションアーティファクト (ICA)
  │     └─→ lfp_cleaned
  │
  ├─→ [Step 3] スパイクソーティング (spike_sorting.py)
  │     ├─→ 300-3000Hz バンドパス
  │     ├─→ MADベース閾値検出
  │     ├─→ PCA + GMM クラスタリング
  │     ├─→ GUI手動キュレーション
  │     └─→ sorting_results {ch: ChannelSortResult}
  │
  ├─→ [Step 4] 刺激応答解析 (stimulus.py)
  │     ├─→ PSTH (Peri-Stimulus Time Histogram)
  │     ├─→ Trial別ラスタープロット
  │     ├─→ 適応解析 (Adaptation)
  │     ├─→ STA (Spike Triggered Average)
  │     └─→ 条件マスク (Baseline/Stim/Post)
  │
  ├─→ [Step 5] 位相ロック解析 (spike_lfp_analysis.py)
  │     ├─→ MRL / PPC / Rayleigh検定
  │     ├─→ 周波数帯域別 × LFPチャンネル別
  │     ├─→ 条件別比較 (Baseline/Stim/Post)
  │     └─→ 集団サマリー
  │
  └─→ [Step 6] 保存 & 可視化
        ├─→ CSV (位相ロック結果、ユニットサマリー)
        ├─→ NPZ (ソーティング結果)
        └─→ PNG (9パネルサマリー、集団プロット)

  └─→ [Step 7] 全チャンネル統合解析 (comprehensive_analysis.py)
        ├─→ スパイクソーティング全チャンネル概要
        │     ├─→ 波形一覧、品質散布図、発火率深度プロファイル
        │     └─→ 条件別発火率深度プロファイル
        ├─→ LFP深度解析
        │     ├─→ CSD (Current Source Density)
        │     ├─→ パワースペクトル深度ヒートマップ
        │     ├─→ 帯域別パワー深度プロファイル
        │     └─→ チャンネル間コヒーレンス行列
        ├─→ スパイク-LFP統合深度解析
        │     ├─→ 位相ロック深度プロファイル
        │     └─→ STA深度プロファイル
        └─→ グランドサマリー (12パネル) + レポート
```

## ファイル構成

### 新規モジュール (v6)
| ファイル | 説明 |
|---------|------|
| `data_loader.py` | PLX読み込み、RecordingSessionデータクラス |
| `stimulus.py` | 刺激プロトコル解析（PSTH、適応、STA） |
| `spike_lfp_analysis.py` | スパイク-LFP統合解析（位相ロック） |
| `comprehensive_analysis.py` | 全チャンネル統合解析（CSD、深度プロファイル、グランドサマリー） |
| `run_analysis.py` | 統合ワークフロー（Step 1-7） |

### 既存モジュール
| ファイル | 説明 |
|---------|------|
| `processing.py` | LFP前処理（フィルタ、ICA、ノイズ除去） |
| `spike_sorting.py` | スパイクソーティングバックエンド |
| `spike_sorting_gui.py` | ソーティングGUI（Tkinter） |
| `waveform_browser.py` | 波形ブラウザ |
| `phase_locking.py` | 位相ロック計算の中核関数 |
| `phase_plotting.py` | 位相ロック可視化 |
| `pipeline.py` | LFP解析パイプライン（従来版） |
| `config_gui.py` | LFPパイプライン設定GUI |
| `plotting.py` | LFP可視化 |
| `saving.py` | 結果保存 |

## 使い方

### A. Jupyter Notebook（推奨）

```python
# === Step 1: データ読み込み ===
from data_loader import load_plx_session

session = load_plx_session(
    "data.plx",
    channel_order=[8, 7, 9, 6, 12, 3, 11, 4, 14, 1, 15, 0, 13, 2, 10, 5]
)
print(session)

# === Step 2: LFP処理 ===
from run_analysis import step2_process_lfp

lfp_result = step2_process_lfp(session)
# lfp_result['lfp_cleaned']  : 処理済みLFP
# lfp_result['lfp_times']    : 時間軸
# lfp_result['noise_mask']   : ノイズマスク

# === Step 3a: スパイクソーティング（新規） ===
from run_analysis import step3_spike_sorting

sorting_results = step3_spike_sorting(
    session,
    channels=[0],        # Ch0のみ
    use_gui=True         # GUIで手動キュレーション
)

# === Step 3b: ソーティング結果の読み込み（保存済み） ===
from run_analysis import step3_load_sorting

sorting_results = step3_load_sorting("sorting.npz")

# === Step 4: 刺激応答解析 ===
from run_analysis import step4_stimulus_analysis

protocol, stim_results = step4_stimulus_analysis(
    session, sorting_results, lfp_result
)

# PSTH確認
protocol.plot_psth(sorting_results[0].units[0].spike_times)

# サマリープロット
protocol.plot_summary(
    sorting_results[0].units[0].spike_times,
    lfp_data=lfp_result['lfp_cleaned'],
    lfp_times=lfp_result['lfp_times'],
    fs=lfp_result['fs']
)

# === Step 5: 位相ロック解析 ===
from run_analysis import step5_phase_locking

analyzer = step5_phase_locking(
    sorting_results, lfp_result, protocol,
    freq_bands={
        'theta': (4, 8),
        'gamma': (30, 80),
    }
)

# ユニットサマリー
analyzer.plot_unit_summary(channel=0, unit_id=1)

# 集団サマリー
analyzer.plot_population_summary()

# === Step 6: 保存 ===
from run_analysis import step6_save_and_plot

step6_save_and_plot(analyzer, sorting_results, protocol, "output/")
```

### B. コマンドライン

```bash
# 新規ソーティング + GUI
python run_analysis.py --plx data.plx --gui --channels 0 1 2

# 保存済みソーティングを使用
python run_analysis.py --plx data.plx --sorting sorting.npz

# 出力先指定
python run_analysis.py --plx data.plx --sorting sorting.npz --output results/
```

### C. LFPパイプラインのみ（従来版）

```python
from pipeline import PipelineConfig, run_pipeline
from config_gui import run_config_gui

# GUIで設定
config = run_config_gui()
results = run_pipeline(config)
```

## データ仕様

### PLXファイル構造
| Signal | Stream ID | fs | Shape | 内容 |
|--------|-----------|-----|-------|------|
| Signal 0 | AI | 1000 Hz | (N, 2) | アナログ入力 |
| Signal 1 | FP | 1000 Hz | (N, 16) | LFP |
| Signal 2 | SPKC | 40000 Hz | (N, 16) | 広帯域（スパイク用） |

### イベント
| Event | 内容 | 数 |
|-------|------|-----|
| EVT01 | 個別刺激タイミング | 90 (= 10 × 9) |
| EVT02 | Trial開始タイミング | 9 |
| EVT03 | カメラフレーム同期 | ~1242 |

### チャンネル順序（NeuroNexus A1x16-5mm-50-703）
```
Depth (µm)    Physical Ch    Logical Index
    0              8              0
   50              7              1
  100              9              2
  150              6              3
  200             12              4
  250              3              5
  300             11              6
  350              4              7
  400             14              8
  450              1              9
  500             15             10
  550              0             11
  600             13             12
  650              2             13
  700             10             14
  750              5             15
```

## 品質基準

### スパイク品質
| 品質 | ISI違反率 | SNR | 判定 |
|------|----------|-----|------|
| 優秀 | < 0.5% | > 6 | Single Unit |
| 良好 | < 2% | > 3 | Single Unit |
| MUA | 2-10% | 2-3 | Multi-Unit |
| ノイズ | > 10% | < 2 | 除外 |

### 位相ロック
| 指標 | 意味 | 基準 |
|------|------|------|
| MRL | Mean Resultant Length | > 0.1 で弱い位相ロック |
| PPC | Pairwise Phase Consistency | スパイク数に依存しない |
| p-value | Rayleigh検定 | < 0.05 で有意 |

## 依存パッケージ

```
numpy, scipy, matplotlib, scikit-learn, neo, pandas
# オプション:
opencv-python  (モーション解析)
pywt           (ウェーブレット)
```

## バージョン履歴

- **v6**: 統合パイプライン（data_loader, stimulus, spike_lfp_analysis, run_analysis）
- **v5**: スパイクソーティングGUI改善（autocorrelogram、waveform browser）
- **v4**: 位相ロックモジュール追加
- **v3**: ICA、環境ノイズ除去、高調波除去
- **v2**: ウェーブレット解析
- **v1**: 基本LFP解析パイプライン
