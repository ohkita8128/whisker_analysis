# 位相ロック解析モジュール 設計書

## 1. 概要

### 1.1 目的
既存のNeuronexus LFP解析パイプラインに、スパイクデータの読み込みと位相ロック解析機能を追加する。

### 1.2 背景
- Sigurdsson et al. (2010) Nature の手法を参考に、海馬-前頭前野間の機能的結合を評価
- 22q11.2欠失マウスモデルにおける神経同期障害の検出が可能に
- S1BF（バレル皮質）でのウィスカー刺激応答解析への応用

---

## 2. システム構成

### 2.1 現在のファイル構成
```
Neuronexus_analysis_v5/
├── config_gui.py      # GUI設定ランチャー
├── pipeline.py        # パイプライン（フロー制御）
├── processing.py      # 前処理関数（フィルタ、ICAなど）
├── plotting.py        # プロット関数
├── saving.py          # 保存関数
└── last_config.json   # 設定ファイル
```

### 2.2 追加するファイル
```
Neuronexus_analysis_v5/
├── spike_processing.py    # 🆕 スパイクデータ処理
├── phase_locking.py       # 🆕 位相ロック解析
├── phase_plotting.py      # 🆕 位相ロック可視化
└── (既存ファイルの更新)
```

---

## 3. データフロー

```
PLXファイル
    │
    ├──────────────────┬──────────────────┐
    ↓                  ↓                  ↓
[LFP信号]         [スパイク時刻]      [イベント]
    │                  │                  │
    ↓                  │                  │
バンドパスフィルタ     │                  │
(theta: 4-12Hz)       │                  │
    │                  │                  │
    ↓                  │                  │
ヒルベルト変換         │                  │
(瞬時位相抽出)         │                  │
    │                  │                  │
    └────────┬─────────┘                  │
             ↓                            │
    スパイク時点での位相取得              │
             │                            │
             ↓                            │
    位相ロック指標計算                    │
    (MRL, PPC, Rayleigh検定)             │
             │                            │
             └────────────────────────────┘
                         ↓
                 刺激条件別の解析
                 (baseline/stim/post)
```

---

## 4. モジュール詳細設計

### 4.1 spike_processing.py

```python
"""
spike_processing.py - スパイクデータの読み込みと前処理
"""

def load_spike_data(segment, unit_filter=None, verbose=True):
    """
    PLXファイルからスパイクデータを読み込む
    
    Parameters
    ----------
    segment : neo.Segment
        Neoライブラリで読み込んだセグメント
    unit_filter : dict or None
        ユニットのフィルタリング条件
        例: {'channel': [1,2,3], 'unit_id': [1,2]}
    verbose : bool
        詳細出力
    
    Returns
    -------
    spike_data : dict
        {
            'spike_trains': list of SpikeTrain objects,
            'unit_info': list of dict (channel, unit_id, n_spikes),
            'all_spike_times': dict {unit_key: np.array}
        }
    """
    pass


def get_unit_key(channel, unit_id):
    """ユニット識別子を生成: 'ch{channel}_unit{unit_id}'"""
    return f"ch{channel}_unit{unit_id}"


def filter_spikes_by_time(spike_times, t_start, t_end):
    """
    時間範囲でスパイクをフィルタリング
    
    Parameters
    ----------
    spike_times : np.ndarray
        スパイクタイムスタンプ（秒）
    t_start, t_end : float
        時間範囲
    
    Returns
    -------
    filtered_spikes : np.ndarray
    """
    pass


def get_spike_counts_per_condition(spike_times, condition_masks, lfp_times):
    """
    条件別のスパイク数をカウント
    
    Parameters
    ----------
    spike_times : np.ndarray
    condition_masks : dict
        {'baseline': mask, 'stim': mask, 'post': mask}
    lfp_times : np.ndarray
    
    Returns
    -------
    counts : dict
        {'baseline': n, 'stim': n, 'post': n}
    """
    pass


def compute_firing_rate(spike_times, t_start, t_end):
    """発火率を計算 (spikes/sec)"""
    pass
```

### 4.2 phase_locking.py

```python
"""
phase_locking.py - 位相ロック解析の中核関数
"""
import numpy as np
from scipy import signal
from scipy.signal import hilbert


# ============================================================
# LFP位相抽出
# ============================================================

def extract_instantaneous_phase(lfp_data, fs, freq_band, filter_order=4):
    """
    LFPから瞬時位相を抽出
    
    Parameters
    ----------
    lfp_data : np.ndarray (n_samples,) or (n_samples, n_channels)
        LFPデータ
    fs : int
        サンプリング周波数
    freq_band : tuple (low, high)
        周波数帯域 (Hz)
    filter_order : int
        バンドパスフィルタの次数
    
    Returns
    -------
    phase : np.ndarray
        瞬時位相 (-π to π)
    amplitude : np.ndarray
        瞬時振幅
    filtered_lfp : np.ndarray
        フィルタ後のLFP
    
    Notes
    -----
    - filtfiltを使用してゼロ位相フィルタリング
    - ヒルベルト変換で解析信号を生成
    """
    pass


def get_spike_phases(spike_times, lfp_phase, lfp_times):
    """
    各スパイク時点でのLFP位相を取得
    
    Parameters
    ----------
    spike_times : np.ndarray
        スパイクタイムスタンプ（秒）
    lfp_phase : np.ndarray
        LFPの瞬時位相
    lfp_times : np.ndarray
        LFPのタイムスタンプ
    
    Returns
    -------
    spike_phases : np.ndarray
        各スパイク時点での位相
    valid_mask : np.ndarray (bool)
        有効なスパイクのマスク（時間範囲内）
    """
    pass


# ============================================================
# 位相ロック指標
# ============================================================

def compute_mean_resultant_length(phases):
    """
    Mean Resultant Length (MRL) を計算
    
    Parameters
    ----------
    phases : np.ndarray
        位相データ（ラジアン）
    
    Returns
    -------
    mrl : float
        MRL値 (0-1)
    preferred_phase : float
        平均位相（ラジアン）
    
    Formula
    -------
    MRL = |1/n × Σ e^(iφₖ)|
    """
    pass


def compute_pairwise_phase_consistency(phases):
    """
    Pairwise Phase Consistency (PPC) を計算
    
    バイアスの少ない位相ロック指標
    
    Parameters
    ----------
    phases : np.ndarray
    
    Returns
    -------
    ppc : float
        PPC値
    
    Formula
    -------
    PPC = (Σᵢ Σⱼ cos(φᵢ - φⱼ)) / (n(n-1)/2)
    """
    pass


def rayleigh_test(phases):
    """
    Rayleigh検定 - 位相分布の一様性を検定
    
    H0: 位相は一様分布（位相ロックなし）
    
    Parameters
    ----------
    phases : np.ndarray
    
    Returns
    -------
    mrl : float
    z_stat : float
        Rayleigh統計量 z = n × MRL²
    p_value : float
    """
    pass


def compute_phase_locking_value(lfp_phase1, lfp_phase2):
    """
    Phase Locking Value (PLV) - LFP間の位相同期
    
    Parameters
    ----------
    lfp_phase1, lfp_phase2 : np.ndarray
        2つのLFPチャンネルの瞬時位相
    
    Returns
    -------
    plv : float
        PLV値 (0-1)
    """
    pass


# ============================================================
# 解析ワークフロー
# ============================================================

def analyze_spike_lfp_coupling(
    spike_times,
    lfp_data,
    lfp_times,
    fs,
    freq_bands=None,
    min_spikes=50,
    verbose=True
):
    """
    スパイク-LFP位相ロック解析のメイン関数
    
    Parameters
    ----------
    spike_times : np.ndarray
        スパイクタイムスタンプ
    lfp_data : np.ndarray (n_samples, n_channels)
        LFPデータ
    lfp_times : np.ndarray
        LFPタイムスタンプ
    fs : int
        サンプリング周波数
    freq_bands : dict or None
        解析する周波数帯域
        デフォルト: {'delta': (1,4), 'theta': (4,8), ...}
    min_spikes : int
        解析に必要な最小スパイク数
    
    Returns
    -------
    results : dict
        {
            'band_name': {
                'channel_idx': {
                    'mrl': float,
                    'ppc': float,
                    'p_value': float,
                    'preferred_phase': float,
                    'n_spikes': int,
                    'spike_phases': np.ndarray
                }
            }
        }
    """
    pass


def analyze_phase_locking_by_condition(
    spike_times,
    lfp_data,
    lfp_times,
    fs,
    condition_masks,
    freq_band=(4, 12),
    lfp_channel=0,
    min_spikes=30
):
    """
    条件別（baseline/stim/post）の位相ロック解析
    
    Parameters
    ----------
    condition_masks : dict
        {'baseline': bool_mask, 'stim': bool_mask, 'post': bool_mask}
    
    Returns
    -------
    results : dict
        各条件での位相ロック結果
    """
    pass
```

### 4.3 phase_plotting.py

```python
"""
phase_plotting.py - 位相ロック解析の可視化
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_phase_histogram(
    spike_phases,
    title="Phase Distribution",
    n_bins=36,
    color='steelblue',
    ax=None
):
    """
    位相分布の極座標ヒストグラム
    
    Parameters
    ----------
    spike_phases : np.ndarray
        スパイク位相（ラジアン）
    title : str
    n_bins : int
    color : str
    ax : matplotlib.axes.Axes or None
    
    Returns
    -------
    fig, ax
    """
    pass


def plot_phase_locking_summary(
    results,
    band_names,
    channel_labels,
    output_dir,
    basename,
    show=True,
    save=True
):
    """
    位相ロック結果のサマリープロット
    
    - バンド×チャンネルのMRLヒートマップ
    - 有意性マーカー
    - 平均位相ベクトル
    """
    pass


def plot_condition_comparison(
    condition_results,
    freq_band_name,
    output_dir,
    basename,
    show=True,
    save=True
):
    """
    条件間（baseline/stim/post）の位相ロック比較
    
    - 各条件の極座標ヒストグラム
    - MRL/PPCの棒グラフ比較
    """
    pass


def plot_spike_lfp_relationship(
    lfp_data,
    lfp_times,
    spike_times,
    lfp_phase,
    t_window=(0, 5),
    fs=1000,
    output_dir=None,
    basename=None,
    show=True,
    save=True
):
    """
    スパイク-LFP関係の時系列プロット
    
    - 上段: 生LFP + フィルタ後LFP
    - 中段: 瞬時位相
    - 下段: スパイクラスタープロット（位相で色分け）
    """
    pass
```

### 4.4 pipeline.py への追加

```python
# PipelineConfig への追加パラメータ

@dataclass
class PipelineConfig:
    # ... 既存パラメータ ...
    
    # === 🆕 スパイク・位相ロック解析 ===
    spike_analysis_enabled: bool = False
    spike_unit_filter: Optional[Dict] = None  # {'channel': [1,2], 'unit_id': [1]}
    
    phase_locking_enabled: bool = False
    phase_locking_bands: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            'theta': (4, 12),
            'gamma': (30, 80)
        }
    )
    phase_locking_lfp_channel: int = 0  # 位相抽出に使うLFPチャンネル
    phase_locking_min_spikes: int = 50  # 解析に必要な最小スパイク数
    
    # 保存オプション
    save_phase_locking_plots: bool = True
    save_phase_locking_csv: bool = True
```

---

## 5. PLXファイルからのスパイクデータ読み込み

### 5.1 Neoライブラリでの読み込み方法

```python
import neo

# PLXファイルを開く
plx = neo.io.PlexonIO(filename=plx_file)
data = plx.read()
segment = data[0].segments[0]

# LFP (AnalogSignal)
lfp_signals = segment.analogsignals  # list of AnalogSignal

# スパイク (SpikeTrain)
spike_trains = segment.spiketrains  # list of SpikeTrain

# 各SpikeTrainの属性
for st in spike_trains:
    print(f"Unit: {st.annotations}")
    print(f"Times: {st.times}")  # スパイク時刻
    print(f"Waveforms: {st.waveforms}")  # 波形（あれば）
```

### 5.2 ユニット情報の取得

```python
def get_unit_info(spike_trains):
    """
    SpikeTrainからユニット情報を抽出
    """
    unit_info = []
    for st in spike_trains:
        info = {
            'channel': st.annotations.get('channel_id', None),
            'unit_id': st.annotations.get('unit_id', None),
            'n_spikes': len(st.times),
            't_start': float(st.t_start),
            't_stop': float(st.t_stop),
        }
        unit_info.append(info)
    return unit_info
```

---

## 6. 設計の妥当性検討

### 6.1 🧠 神経科学者の視点

#### ✅ 妥当な点

1. **周波数帯域の選択**
   - シータ帯域 (4-12 Hz) は海馬-皮質間の長距離同期に関与
   - Sigurdsson論文と同様の設定で比較可能

2. **位相抽出手法**
   - ヒルベルト変換は標準的手法
   - `filtfilt`によるゼロ位相フィルタリングで位相遅延を回避

3. **複数指標の実装**
   - MRL: 直感的で広く使用される
   - PPC: スパイク数に依存しない、より安定した指標
   - Rayleigh検定: 統計的有意性の評価

4. **条件別解析**
   - baseline/stim/post の比較は刺激応答研究の標準

5. **最小スパイク数の閾値設定**
   - 50スパイクは位相ロック解析の標準的な閾値

#### ⚠️ 考慮が必要な点

1. **マルチユニット vs シングルユニット**
   - PLXファイルのスパイクソーティング品質に依存
   - ソート済みユニットのみを使用すべき

2. **LFPチャンネル選択**
   - S1BFの特定層（Layer 4/5）からの記録が重要
   - 複数チャンネルでの解析オプションが必要

3. **呼吸アーティファクト**
   - バレル皮質のデルタ/シータ波は呼吸と位相ロック
   - Nature Comms (2014) の知見を考慮

4. **刺激アーティファクト**
   - ウィスカー刺激直後のスパイクは除外すべき可能性
   - `stim_margin_sec` パラメータで対応

#### 📝 推奨する追加機能

```python
# 1. スパイクソーティング品質フィルタ
spike_quality_filter: str = 'sorted_only'  # 'all', 'sorted_only', 'mua'

# 2. 刺激アーティファクト除外
stim_artifact_window: float = 0.005  # 刺激後5ms除外

# 3. 呼吸位相との関係（オプション）
respiration_correction: bool = False
```

---

### 6.2 💻 プログラマーの視点

#### ✅ 妥当な点

1. **モジュール分離**
   - 機能別にファイルを分割（spike_processing, phase_locking, phase_plotting）
   - 既存コードへの影響を最小化

2. **データ構造**
   - 辞書ベースの結果格納は柔軟性が高い
   - NumPy配列の一貫した使用

3. **エラーハンドリング**
   - min_spikes閾値でスパイク不足を検出
   - 時間範囲外のスパイクを除外

4. **既存パイプラインとの統合**
   - PipelineConfigへの自然な拡張
   - 同じデータロード機構を再利用

#### ⚠️ 考慮が必要な点

1. **メモリ効率**
   ```python
   # 問題: 全チャンネル×全バンドの位相を一度に計算
   # 解決: ジェネレータパターンで逐次処理
   
   def iter_phase_analysis(lfp_data, fs, freq_bands):
       for band_name, (low, high) in freq_bands.items():
           phase = extract_instantaneous_phase(lfp_data, fs, (low, high))
           yield band_name, phase
   ```

2. **並列化の可能性**
   ```python
   # チャンネル別・バンド別の処理は独立
   # joblib等で並列化可能
   
   from joblib import Parallel, delayed
   
   results = Parallel(n_jobs=-1)(
       delayed(analyze_single_channel)(ch, lfp_data[:, ch], ...)
       for ch in range(n_channels)
   )
   ```

3. **型ヒントの完備**
   ```python
   from typing import Dict, Tuple, Optional, List
   import numpy as np
   from numpy.typing import NDArray
   
   def compute_mrl(phases: NDArray[np.float64]) -> Tuple[float, float]:
       ...
   ```

4. **テスト容易性**
   ```python
   # 合成データでの単体テスト
   def test_mrl_perfect_locking():
       phases = np.zeros(100)  # 全て同位相
       mrl, _ = compute_mrl(phases)
       assert mrl > 0.99
   
   def test_mrl_no_locking():
       phases = np.random.uniform(-np.pi, np.pi, 100)
       mrl, _ = compute_mrl(phases)
       assert mrl < 0.2
   ```

#### 📝 推奨する追加実装

```python
# 1. 進捗表示
from tqdm import tqdm

for band in tqdm(freq_bands, desc="Phase-locking analysis"):
    ...

# 2. キャッシュ機構
import hashlib
import pickle

def cache_results(func):
    """解析結果をキャッシュするデコレータ"""
    def wrapper(*args, **kwargs):
        cache_key = hashlib.md5(str(args).encode()).hexdigest()
        cache_file = f".cache/{cache_key}.pkl"
        if os.path.exists(cache_file):
            return pickle.load(open(cache_file, 'rb'))
        result = func(*args, **kwargs)
        pickle.dump(result, open(cache_file, 'wb'))
        return result
    return wrapper

# 3. 設定のバリデーション
def validate_config(config: PipelineConfig) -> List[str]:
    """設定の妥当性チェック"""
    warnings = []
    if config.phase_locking_enabled and not config.spike_analysis_enabled:
        warnings.append("phase_locking requires spike_analysis")
    if config.phase_locking_min_spikes < 20:
        warnings.append("min_spikes < 20 may produce unreliable results")
    return warnings
```

---

## 7. 実装優先順位

### Phase 1: 基本機能（必須）
1. ✅ `spike_processing.py` - スパイク読み込み
2. ✅ `phase_locking.py` - MRL, PPC, Rayleigh検定
3. ✅ 極座標ヒストグラム

### Phase 2: 統合（必須）
4. ✅ `pipeline.py` への統合
5. ✅ 条件別解析（baseline/stim/post）
6. ✅ 結果CSV出力

### Phase 3: 拡張（推奨）
7. ⬜ config_gui.py への追加
8. ⬜ PLV（LFP-LFP間）解析
9. ⬜ 時間窓解析（sliding window）

### Phase 4: 高度な機能（オプション）
10. ⬜ 周波数依存性の詳細解析
11. ⬜ 並列処理対応
12. ⬜ 呼吸補正

---

## 8. 出力ファイル形式

### 8.1 CSV出力例

```csv
# phase_locking_summary.csv
unit_id,channel,band,condition,n_spikes,mrl,ppc,p_value,preferred_phase_deg,significant
ch1_unit1,1,theta,baseline,156,0.234,0.198,0.0012,45.3,True
ch1_unit1,1,theta,stim,203,0.312,0.287,0.0001,52.1,True
ch1_unit1,1,theta,post,178,0.189,0.156,0.0234,48.7,True
```

### 8.2 NPZ出力例

```python
np.savez(
    'phase_locking_results.npz',
    spike_phases=spike_phases,      # dict of arrays
    mrl_values=mrl_values,          # (n_units, n_bands, n_conditions)
    ppc_values=ppc_values,
    p_values=p_values,
    preferred_phases=preferred_phases,
    unit_info=unit_info,            # list of dicts
    freq_bands=freq_bands,
    conditions=['baseline', 'stim', 'post']
)
```

---

## 9. まとめ

### 神経科学的観点での結論
- 設計は Sigurdsson et al. (2010) の手法と整合性がある
- S1BFウィスカー刺激実験への適用に適している
- 22q11.2欠失マウスでの同期障害検出に利用可能
- 呼吸アーティファクトへの注意が必要

### プログラミング観点での結論
- モジュール設計は既存コードベースと整合的
- メモリ効率と並列化の余地あり
- テスト容易な構造
- 段階的実装が可能

### 推奨する次のステップ
1. `spike_processing.py` の実装とPLXファイルでの動作確認
2. 合成データでの位相ロック関数の単体テスト
3. 既存データでの統合テスト
