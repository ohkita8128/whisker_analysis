# Neuronexus Whisker Stimulation Analysis v5

## 概要
PLXファイルからLFPデータを読み込み、ノイズ処理・解析を行うパイプライン。

## v5 変更点

### 🆕 FIR/IIRフィルター選択
- バンドパスフィルター: IIR (Butterworth) または FIR (窓関数法) を選択可能
- ノッチフィルター: IIR または FIR を選択可能
- FIRタップ数の設定が可能（奇数推奨）

### 🆕 表示範囲の独立設定
- **フィルター設定**: `filter_lowcut` / `filter_highcut` 
- **パワースペクトル表示**: `power_freq_min` / `power_freq_max` （独立）
- **FFT比較表示**: `fft_freq_max` （独立）

### 改善
- GUIのフィルター設定をセクション分け（バンドパス/ノッチ）
- Q値の説明を追加（大きい=狭帯域）
- 実行確認ダイアログにフィルター種類を表示

## ファイル構成

```
neuronexus_analysis/
├── config_gui.py      # GUI設定ランチャー
├── pipeline.py        # パイプライン（フロー制御）
├── processing.py      # 前処理関数（フィルタ、ICAなど）
├── plotting.py        # プロット関数
├── saving.py          # 保存関数（CSV、NPZ）
└── last_config.json   # (自動生成) 前回の設定
```

## 使い方

### 🖱️ GUIで実行

```bash
python config_gui.py
```

### 💻 コードで実行

```python
from pipeline import PipelineConfig, run_pipeline

# デフォルト設定
config = PipelineConfig()
results = run_pipeline(config)

# カスタム設定（FIRフィルター使用例）
config = PipelineConfig(
    filter_type='fir',           # FIRフィルター
    filter_fir_numtaps=201,      # タップ数
    notch_type='fir',            # ノッチもFIR
    power_freq_min=1.0,          # パワー表示範囲
    power_freq_max=80.0,
)
results = run_pipeline(config)
```

### バッチ処理

```python
files = ["file1.plx", "file2.plx"]
for f in files:
    config = PipelineConfig(
        plx_file=f,
        motion_roi=(100, 100, 200, 200),
        show_plots=False,
    )
    run_pipeline(config)
```

## 設定オプション

### バンドパスフィルタ
| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `filter_enabled` | True | フィルタON/OFF |
| `filter_type` | 'iir' | 'iir' (Butterworth) or 'fir' |
| `filter_lowcut` | 0.1 | ハイパス (Hz) |
| `filter_highcut` | 100.0 | ローパス (Hz) |
| `filter_order` | 4 | IIRフィルタ次数 |
| `filter_fir_numtaps` | None | FIRタップ数（None/0=自動計算）|

### ノッチフィルタ（IIR固定）
| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `notch_enabled` | True | ノッチON/OFF |
| `notch_freq` | 60.0 | ノッチ周波数 (Hz) |
| `notch_Q` | 60.0 | Q値（大=狭帯域）|

※ノッチフィルタはIIR固定です（FIRは非効率なため）

### 表示範囲設定
| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `power_freq_min` | 0.5 | パワースペクトル表示最小 (Hz) |
| `power_freq_max` | 100.0 | パワースペクトル表示最大 (Hz) |
| `fft_freq_max` | 300.0 | FFT比較プロット最大 (Hz) |

### モーション・ICA
| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `motion_analysis` | True | モーション解析ON/OFF |
| `motion_roi` | None | ROI (None=手動選択) |
| `ica_enabled` | True | ICA ON/OFF |
| `ica_noise_ratio_threshold` | 1.5 | 除去閾値 |
| `ica_max_remove` | 4 | 最大除去数 |

### 保存
| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `save_processing_overview` | True | 処理概要プロット |
| `save_ica_components` | True | ICA成分プロット |
| `save_power_analysis` | True | パワー解析プロット |
| `save_channel_heatmap` | True | ヒートマップ |
| `save_summary_csv` | True | サマリーCSV |
| `save_channel_csv` | True | チャンネル別CSV |

### ウェーブレット
| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `wavelet_enabled` | False | ウェーブレット解析ON/OFF |
| `wavelet_freq_min` | 1.0 | 最小周波数 (Hz) |
| `wavelet_freq_max` | 100.0 | 最大周波数 (Hz) |
| `wavelet_n_freqs` | 50 | 周波数分割数 |

## FIR vs IIR の選択ガイド

| 特性 | IIR (Butterworth) | FIR |
|------|-------------------|-----|
| 位相特性 | 非線形（filtfilt使用で対称化）| 線形位相 |
| 計算コスト | 低い | 高い（タップ数に依存）|
| 急峻なカットオフ | 低次数で可能 | 多いタップ数が必要 |
| 安定性 | 数値誤差の影響を受けやすい | 常に安定 |
| 推奨用途 | 一般的な用途 | 位相が重要な解析 |

### FIRタップ数の自動計算

`filter_fir_numtaps = None` または `0` の場合、以下の式で自動計算されます：

```
遷移帯域幅 = max(lowcut × 0.5, 1.0) Hz
タップ数 = 3.3 × fs ÷ 遷移帯域幅（奇数に切り上げ）
```

例：
- fs=1000Hz, lowcut=1Hz → 遷移帯域=1Hz → タップ数≈3301
- fs=1000Hz, lowcut=0.1Hz → 遷移帯域=1Hz → タップ数≈3301
- fs=1000Hz, lowcut=10Hz → 遷移帯域=5Hz → タップ数≈661

### ノッチフィルタがIIR固定の理由

ノッチフィルタは非常に狭い帯域（Q=60で約1Hz幅）を除去するため、FIRでは数千タップ必要になり非効率。IIRなら少ない計算量で実現できます。

## 処理フロー

1. **読み込み** - PLX、チャンネル並び替え、同期
2. **フィルタ** - バンドパス + ノッチ（IIR/FIR選択可）
3. **Trim** - 動画同期範囲で切り出し
4. **チャンネル** - 悪いチャンネル除外
5. **モーション** - ノイズマスク作成
6. **ICA** - アーティファクト除去
7. **解析** - PSD、バンドパワー
8. **ウェーブレット** - CWTスペクトログラム（オプション）
