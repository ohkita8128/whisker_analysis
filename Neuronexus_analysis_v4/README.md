# Neuronexus Whisker Stimulation Analysis

## 概要
PLXファイルからLFPデータを読み込み、ノイズ処理・解析を行うパイプライン。

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

# カスタム設定
config = PipelineConfig(
    filter_lowcut=1.0,
    ica_enabled=False,
    save_channel_heatmap=False,
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

### フィルタ
| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `filter_enabled` | True | フィルタON/OFF |
| `filter_lowcut` | 0.1 | ハイパス (Hz) |
| `filter_highcut` | 100.0 | ローパス (Hz) |
| `notch_enabled` | True | ノッチON/OFF |
| `notch_freq` | 60.0 | ノッチ周波数 |

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

### 追加プロット
| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `save_lfp_regions` | True | 全チャンネル+領域プロット |
| `lfp_regions_duration` | 60.0 | 表示時間 (秒) |
| `lfp_regions_t_start` | None | 開始時刻 (None=最初から) |
| `save_fft_comparison` | True | FFT比較プロット |
| `fft_freq_max` | 300.0 | FFT最大周波数 (Hz) |

### 同期動画
| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `create_sync_video` | False | 同期動画作成 |
| `sync_video_duration` | 30.0 | 動画の長さ (None=PLX全長) |
| `sync_video_t_start` | None | 開始時刻 (None=最初から) |

### ウェーブレット
| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `wavelet_enabled` | False | ウェーブレット解析ON/OFF |
| `wavelet_freq_min` | 1.0 | 最小周波数 (Hz) |
| `wavelet_freq_max` | 100.0 | 最大周波数 (Hz) |
| `wavelet_n_freqs` | 50 | 周波数分割数 |
| `wavelet_duration` | 30.0 | 表示時間 (秒) |
| `save_wavelet_single` | True | 単一チャンネル保存 |
| `save_wavelet_all` | True | 全チャンネル保存 |

## 処理フロー

1. **読み込み** - PLX、チャンネル並び替え、同期
2. **フィルタ** - バンドパス + ノッチ
3. **Trim** - 動画同期範囲で切り出し
4. **チャンネル** - 悪いチャンネル除外
5. **モーション** - ノイズマスク作成
6. **ICA** - アーティファクト除去
7. **解析** - PSD、バンドパワー
8. **ウェーブレット** - CWTスペクトログラム（オプション）
