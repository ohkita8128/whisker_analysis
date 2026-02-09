# Neuronexus Analysis v6

## 概要

PLXファイルからLFP・スパイク・位相ロック解析を統合的に行うGUI駆動パイプライン。

## v5 → v6 主な変更点

### 🆕 3段階GUI駆動フロー
```
PLX読み込み → LFP Filter GUI → Spike Sorting GUI → Phase Locking GUI
```

各ステップが独立したGUIで操作可能。設定はJSON保存/読み込みに対応。

### 🆕 LFP Filter GUI (`lfp_filter_gui.py`)
- フィルタの**周波数応答プレビュー**（振幅・位相特性）
- PSDプレビュー（データ読み込み後）
- バンドパス（IIR/FIR）、ノッチ、高調波除去の設定
- チャンネル処理、モーション、ICA の設定
- 帯域パワー設定（プリセット付き）

### 🆕 改良 Spike Sorting GUI (`spike_sort_gui.py`)
- **設定パネル**でフィルタ/検出/PCA/クラスタの全パラメータをGUI上で変更
- 全チャンネル一括ソーティング
- グランドサマリーボタン
- 設定のJSON保存/読み込み

### 🆕 Phase Locking GUI (`phase_gui.py`)
- スパイクソース選択（PLX内スパイク or ソーティング結果）
- 解析帯域の選択（theta/gamma/beta）
- 条件別解析（baseline/stim/post）
- MRLヒートマップのプレビュー
- **グランドサマリー**出力

### 🆕 プロット整理
- **パワー解析サマリー**: PSD + バンドパワー + ヒートマップ を1枚に統合
- **スパイクグランドサマリー**: 全ch×全ユニットの波形+PCAを1枚に
- **位相ロックグランドサマリー**: 全ユニット×全帯域のMRL + 極座標 + 条件比較
- チャンネル詳細は必要な時のみ出力（枚数削減）

## ファイル構成

```
Neuronexus_analysis_v6/
├── main.ipynb              # メインノートブック（推奨）
├── main.py                 # メインオーケストレータ（CLI版）
├── numpy_compat.py         # NumPy 2.0+ 互換パッチ
├── data_loader.py          # PLXデータ読み込み
│
├── lfp_filter_gui.py       # LFPフィルタリングGUI
├── lfp_pipeline.py         # LFP処理パイプライン
├── lfp_processing.py       # LFP処理関数
├── lfp_plotting.py         # LFPプロット（統合版）
│
├── spike_sorting.py        # スパイクソーティング バックエンド
├── spike_sort_gui.py       # スパイクソーティングGUI（改良版）
├── spike_plotting.py       # スパイクプロット（統合版）
├── spike_processing.py     # スパイクデータ前処理
├── waveform_browser.py     # 波形ブラウザ
│
├── phase_locking.py        # 位相ロック解析 バックエンド
├── phase_gui.py            # 位相ロック解析GUI
├── phase_plotting.py       # 位相ロックプロット（v5互換）
├── phase_plotting_v6.py    # 位相ロックプロット（グランドサマリー）
│
├── saving.py               # データ保存（CSV, NPZ）
├── get_path.py             # ファイルパスユーティリティ
│
├── lfp_config.json         # (自動生成) LFP設定
├── spike_config.json       # (自動生成) スパイク設定
├── phase_config.json       # (自動生成) 位相ロック設定
└── README.md               # このファイル
```

## 使い方

### Jupyter Notebook（推奨）
```bash
jupyter notebook main.ipynb
```
ノートブック冒頭の設定セルでPLXファイルパスやステップのON/OFFを指定し、上から順にセルを実行してください。

### コマンドライン実行
```bash
python main.py                                # GUI でファイル選択
python main.py --plx /path/to/file.plx        # ファイル指定
python main.py --plx file.plx --skip-spike    # スパイクソーティングをスキップ
python main.py --plx file.plx --skip-phase    # 位相ロック解析をスキップ
python main.py --plx file.plx --no-wideband   # Widebandデータを読み込まない
```

### 個別GUI起動
```python
# LFPフィルタリングのみ
from lfp_filter_gui import launch_lfp_gui
launch_lfp_gui()

# スパイクソーティングのみ（PLXファイルから）
from spike_sort_gui import launch_spike_gui
import neo, numpy as np
plx = neo.io.PlexonIO(filename="file.plx")
seg = plx.read()[0].segments[0]
wideband = np.array(seg.analogsignals[0])
fs = float(seg.analogsignals[0].sampling_rate)
launch_spike_gui(wideband_data=wideband, fs=fs)
```

## 出力プロット一覧

| プロット | ファイル名 | 内容 |
|---------|-----------|------|
| 処理概要 | `*_processing_overview.png` | Raw→Filtered→Cleaned + モーション |
| FFT比較 | `*_fft_comparison.png` | フィルタ前後のPSD比較 |
| パワーサマリー | `*_power_summary.png` | **PSD + バンドパワー + ヒートマップ (1枚)** |
| ICA成分 | `*_ica_components.png` | 上位ICA成分の時系列 |
| 全チャンネル | `*_all_channels.png` | 全ch波形 + 領域マーク |
| スパイクサマリー | `*_spike_grand_summary.png` | **全ch×全ユニット (1枚)** |
| 品質テーブル | `*_spike_quality.png` | ユニット品質指標テーブル |
| チャンネル詳細 | `*_chN_spike_detail.png` | 波形+PCA+ISI+ACG (オプション) |
| 位相グランドサマリー | `*_phase_grand_summary.png` | **MRL + 極座標 + 条件比較 (1枚)** |

## 処理フロー

```
PLXファイル
  │
  ├── LFPデータ ──→ [LFP Filter GUI]
  │     │               ├── バンドパス (IIR/FIR)
  │     │               ├── ノッチ
  │     │               ├── 高調波除去
  │     │               ├── チャンネル処理
  │     │               ├── モーション解析
  │     │               └── ICA
  │     │
  │     └──→ LFPプロット
  │               ├── 処理概要, FFT比較
  │               ├── パワーサマリー (統合1枚)
  │               └── 全チャンネル波形
  │
  ├── Widebandデータ ──→ [Spike Sorting GUI]
  │     │                    ├── フィルタ (300-3000Hz)
  │     │                    ├── スパイク検出
  │     │                    ├── PCA + GMM
  │     │                    └── 手動修正
  │     │
  │     └──→ スパイクプロット
  │               ├── グランドサマリー (1枚)
  │               └── 品質テーブル
  │
  └── LFP + スパイク ──→ [Phase Locking GUI]
          │                   ├── 位相抽出 (Hilbert)
          │                   ├── MRL / PPC / Rayleigh検定
          │                   └── 条件別解析
          │
          └──→ 位相ロックプロット
                    ├── グランドサマリー (1枚)
                    └── CSV出力
```

## 依存パッケージ
```
numpy, scipy, matplotlib, scikit-learn, neo, pywt, opencv-python
```

### NumPy 2.0+ の場合
NumPy 2.0 で `ndarray.ptp` が削除されたため、`quantities` / `neo` パッケージでエラーが発生します。
本プロジェクトでは `numpy_compat.py` で自動パッチを適用します。

もしパッチで解決しない場合は以下のいずれかを実行してください:
```bash
# 方法1: quantities と neo を最新にアップデート
pip install --upgrade quantities neo

# 方法2: NumPy をダウングレード
pip install "numpy<2.0"
```
