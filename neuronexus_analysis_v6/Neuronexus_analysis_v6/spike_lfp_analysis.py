"""
spike_lfp_analysis.py - スパイク-LFP統合解析

スパイクソーティング結果（spike_sorting.py）とLFPデータを統合し、
位相ロック解析を実行するメインモジュール。

機能:
  - 位相ロック解析（MRL, PPC, Rayleigh検定）
  - 周波数帯域別 × チャンネル別 解析
  - 条件別（Baseline/Stim/Post）比較
  - Spike Triggered Average (STA)
  - 結果の可視化・保存

使い方:
    from spike_lfp_analysis import SpikeLFPAnalyzer
    
    analyzer = SpikeLFPAnalyzer(
        lfp_data=lfp_filtered,
        lfp_times=lfp_times,
        fs_lfp=1000,
        sorting_results=sorting_results,  # spike_sorting.pyの出力
        protocol=protocol                  # stimulus.StimulusProtocol
    )
    
    # 全ユニット解析
    results = analyzer.analyze_all()
    
    # 可視化
    analyzer.plot_unit_summary(channel=0, unit_id=1)
    
    # CSV保存
    analyzer.save_results("output_dir")
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
import os
import csv
import warnings


# ============================================================
# 結果データクラス
# ============================================================

@dataclass
class UnitPhaseLocking:
    """1ユニットの位相ロック結果"""
    channel: int
    unit_id: int
    n_spikes: int
    is_mua: bool
    snr: float
    isi_violation_rate: float
    
    # 周波数帯域別結果 {band_name: {lfp_ch: PhaseLockingResult}}
    band_results: Dict = field(default_factory=dict)
    
    # 条件別結果 {condition: PhaseLockingResult}
    condition_results: Dict = field(default_factory=dict)
    
    # STA
    sta_time: np.ndarray = None
    sta: np.ndarray = None
    
    @property
    def unit_key(self) -> str:
        return f"ch{self.channel}_unit{self.unit_id}"
    
    def get_best_phase_locking(self, band_name: str) -> Optional[Any]:
        """指定帯域で最もMRLが高いチャンネルの結果を返す"""
        if band_name not in self.band_results:
            return None
        results = self.band_results[band_name]
        best = None
        best_mrl = 0
        for ch, result in results.items():
            if result is not None and result.mrl > best_mrl:
                best = result
                best_mrl = result.mrl
        return best


# ============================================================
# メイン解析クラス
# ============================================================

class SpikeLFPAnalyzer:
    """
    スパイク-LFP統合解析
    
    Parameters
    ----------
    lfp_data : ndarray (n_samples, n_channels)
        処理済みLFPデータ
    lfp_times : ndarray
        LFP時間軸（秒）
    fs_lfp : int
        LFPサンプリングレート
    sorting_results : dict
        {channel: ChannelSortResult} - spike_sorting.pyの出力
    protocol : StimulusProtocol or None
        刺激プロトコル（条件別解析に使用）
    noise_mask : ndarray or None
        ノイズマスク（モーションアーティファクト区間を除外）
    freq_bands : dict or None
        解析する周波数帯域。Noneの場合はデフォルト。
    """
    
    # デフォルト周波数帯域
    DEFAULT_BANDS = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 14),
        'beta': (14, 30),
        'low_gamma': (30, 50),
        'high_gamma': (50, 100),
    }
    
    def __init__(self, lfp_data, lfp_times, fs_lfp,
                 sorting_results=None, protocol=None, 
                 noise_mask=None, freq_bands=None):
        
        self.lfp_data = np.atleast_2d(lfp_data) if lfp_data.ndim == 1 else lfp_data
        self.lfp_times = lfp_times
        self.fs_lfp = fs_lfp
        self.sorting_results = sorting_results or {}
        self.protocol = protocol
        self.noise_mask = noise_mask
        self.freq_bands = freq_bands or self.DEFAULT_BANDS
        
        # 結果格納
        self.unit_results: Dict[str, UnitPhaseLocking] = {}
        
        # LFPチャンネル数
        self.n_lfp_channels = self.lfp_data.shape[1]
    
    # ============================================================
    # ユニット列挙
    # ============================================================
    
    def get_all_units(self, include_noise=False, include_mua=True):
        """
        全ソーティング結果からユニット一覧を取得
        
        Returns
        -------
        units : list of (channel, unit_id, SpikeUnit)
        """
        units = []
        for ch, result in self.sorting_results.items():
            for unit in result.units:
                if unit.is_noise and not include_noise:
                    continue
                if unit.is_mua and not include_mua:
                    continue
                units.append((ch, unit.unit_id, unit))
        return units
    
    def get_spike_times(self, channel, unit_id):
        """指定ユニットのスパイク時刻を取得"""
        if channel in self.sorting_results:
            for unit in self.sorting_results[channel].units:
                if unit.unit_id == unit_id:
                    return unit.spike_times
        return np.array([])
    
    # ============================================================
    # 位相ロック解析
    # ============================================================
    
    def analyze_unit(self, channel, unit_id, 
                     lfp_channels=None,
                     min_spikes=50,
                     verbose=True) -> Optional[UnitPhaseLocking]:
        """
        1ユニットの位相ロック解析
        
        Parameters
        ----------
        channel : int
            スパイクチャンネル
        unit_id : int
            ユニットID
        lfp_channels : list or None
            解析するLFPチャンネル。Noneの場合は全チャンネル。
        min_spikes : int
            最小スパイク数
        verbose : bool
        
        Returns
        -------
        result : UnitPhaseLocking or None
        """
        from phase_locking import (
            extract_instantaneous_phase,
            analyze_spike_phase_locking,
        )
        
        log = print if verbose else lambda *a, **kw: None
        
        # ユニット情報取得
        unit = None
        if channel in self.sorting_results:
            for u in self.sorting_results[channel].units:
                if u.unit_id == unit_id:
                    unit = u
                    break
        
        if unit is None:
            log(f"  Unit ch{channel}_unit{unit_id} not found")
            return None
        
        spike_times = unit.spike_times
        
        if len(spike_times) < min_spikes:
            log(f"  Unit ch{channel}_unit{unit_id}: insufficient spikes ({len(spike_times)} < {min_spikes})")
            return None
        
        log(f"\n  Analyzing ch{channel}_unit{unit_id} "
            f"(n={len(spike_times)}, SNR={unit.snr:.1f}, "
            f"ISI={unit.isi_violation_rate:.1f}%)")
        
        # 結果オブジェクト
        pl_result = UnitPhaseLocking(
            channel=channel,
            unit_id=unit_id,
            n_spikes=len(spike_times),
            is_mua=unit.is_mua,
            snr=unit.snr,
            isi_violation_rate=unit.isi_violation_rate,
        )
        
        # LFPチャンネル
        if lfp_channels is None:
            lfp_channels = list(range(self.n_lfp_channels))
        
        # ----- 周波数帯域別解析 -----
        for band_name, freq_band in self.freq_bands.items():
            log(f"    {band_name} ({freq_band[0]}-{freq_band[1]} Hz):")
            
            # LFP位相抽出
            phase, amplitude, filtered = extract_instantaneous_phase(
                self.lfp_data, self.fs_lfp, freq_band
            )
            
            band_ch_results = {}
            
            for lfp_ch in lfp_channels:
                ch_phase = phase[:, lfp_ch]
                
                result = analyze_spike_phase_locking(
                    spike_times, ch_phase, self.lfp_times, min_spikes
                )
                
                band_ch_results[lfp_ch] = result
                
                if verbose and result is not None:
                    sig = "*" if result.significant else ""
                    log(f"      LFP-Ch{lfp_ch}: MRL={result.mrl:.3f}, "
                        f"PPC={result.ppc:.3f}, p={result.p_value:.4f}{sig}")
            
            pl_result.band_results[band_name] = band_ch_results
        
        # ----- 条件別解析 -----
        if self.protocol is not None:
            log(f"    Condition analysis:")
            condition_masks = self.protocol.create_condition_masks(
                self.lfp_times, self.fs_lfp
            )
            
            # ノイズマスクを適用
            if self.noise_mask is not None:
                for cond in condition_masks:
                    condition_masks[cond] = condition_masks[cond] & ~self.noise_mask
            
            from phase_locking import analyze_phase_locking_by_condition
            
            # デフォルトでtheta帯域を使用
            theta_band = self.freq_bands.get('theta', (4, 8))
            
            cond_results = analyze_phase_locking_by_condition(
                spike_times,
                self.lfp_data,
                self.lfp_times,
                self.fs_lfp,
                condition_masks,
                freq_band=theta_band,
                lfp_channel=0,  # 最初のLFPチャンネル
                min_spikes=min_spikes // 2,
                verbose=verbose
            )
            
            pl_result.condition_results = cond_results
        
        # ----- STA -----
        if self.protocol is not None:
            sta_time, sta = self.protocol.compute_sta(
                spike_times, self.lfp_data, self.lfp_times, self.fs_lfp
            )
            pl_result.sta_time = sta_time
            pl_result.sta = sta
        
        # 保存
        self.unit_results[pl_result.unit_key] = pl_result
        
        return pl_result
    
    def analyze_all(self, lfp_channels=None, min_spikes=50, 
                    verbose=True) -> Dict[str, UnitPhaseLocking]:
        """
        全ユニットの位相ロック解析
        
        Returns
        -------
        results : dict {unit_key: UnitPhaseLocking}
        """
        log = print if verbose else lambda *a, **kw: None
        
        units = self.get_all_units()
        log(f"\n=== Phase-Locking Analysis: {len(units)} units ===")
        
        for ch, uid, unit in units:
            self.analyze_unit(ch, uid, lfp_channels, min_spikes, verbose)
        
        log(f"\n=== Complete: {len(self.unit_results)} units analyzed ===")
        return self.unit_results
    
    # ============================================================
    # 可視化
    # ============================================================
    
    def plot_unit_summary(self, channel, unit_id, 
                          lfp_channel=0, figsize=(18, 14),
                          save_path=None):
        """
        1ユニットの統合サマリープロット（9パネル）
        
        Layout:
        [Waveforms    ] [PCA           ] [ISI histogram  ]
        [PSTH         ] [Raster/trial  ] [Adaptation     ]
        [Phase polar  ] [Phase heatmap ] [STA            ]
        """
        from phase_plotting import plot_phase_histogram
        from spike_sorting import compute_isi_histogram, compute_autocorrelogram
        
        unit_key = f"ch{channel}_unit{unit_id}"
        
        # ユニット取得
        unit = None
        result = self.sorting_results.get(channel)
        if result:
            for u in result.units:
                if u.unit_id == unit_id:
                    unit = u
                    break
        
        if unit is None:
            print(f"Unit {unit_key} not found")
            return None
        
        pl_result = self.unit_results.get(unit_key)
        
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        
        # ========= Row 1: Spike Sorting Quality =========
        
        # 1-1: Waveforms
        ax = axes[0, 0]
        if unit.waveforms is not None and len(unit.waveforms) > 0:
            n_show = min(100, len(unit.waveforms))
            indices = np.random.choice(len(unit.waveforms), n_show, replace=False)
            time_ms = result.waveform_time_ms if result.waveform_time_ms is not None else \
                      np.arange(unit.waveforms.shape[1]) / result.fs * 1000
            
            for i in indices:
                ax.plot(time_ms, unit.waveforms[i], color=unit.color, alpha=0.1, linewidth=0.5)
            ax.plot(time_ms, np.mean(unit.waveforms, axis=0), 'k-', linewidth=2)
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Waveforms (n={unit.n_spikes})')
        
        # 1-2: PCA
        ax = axes[0, 1]
        if unit.pca_features is not None and result.all_pca_features is not None:
            # 他のユニットも表示
            for other_unit in result.units:
                if other_unit.unit_id != unit_id and not other_unit.is_noise:
                    ax.scatter(other_unit.pca_features[:, 0], other_unit.pca_features[:, 1],
                             s=3, alpha=0.2, color='gray')
            ax.scatter(unit.pca_features[:, 0], unit.pca_features[:, 1],
                      s=3, alpha=0.5, color=unit.color)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(f'PCA (SNR={unit.snr:.1f}, ISI={unit.isi_violation_rate:.1f}%)')
        
        # 1-3: ISI + Autocorrelogram
        ax = axes[0, 2]
        if len(unit.spike_times) > 1:
            bins_isi, hist_isi = compute_isi_histogram(unit.spike_times)
            if len(bins_isi) > 0:
                ax.bar(bins_isi, hist_isi, width=1.0, color=unit.color, alpha=0.7)
                ax.axvline(2.0, color='red', linestyle='--', linewidth=1, label='2ms refractory')
                ax.set_xlabel('ISI (ms)')
                ax.set_ylabel('Count')
                ax.legend(fontsize=8)
        ax.set_title('ISI Histogram')
        
        # ========= Row 2: Stimulus Response =========
        
        if self.protocol is not None:
            spike_times = unit.spike_times
            
            # 2-1: PSTH
            self.protocol.plot_psth(spike_times, ax=axes[1, 0])
            
            # 2-2: Raster by trial
            self.protocol.plot_raster_by_trial(spike_times, ax=axes[1, 1])
            
            # 2-3: Adaptation
            self.protocol.plot_adaptation(spike_times, ax=axes[1, 2])
        else:
            # Protocolがない場合はAutocorrelogramなど
            ax = axes[1, 0]
            if len(unit.spike_times) > 1:
                bins_ac, ac = compute_autocorrelogram(unit.spike_times)
                ax.bar(bins_ac, ac, width=1.0, color=unit.color, alpha=0.7)
                ax.set_xlabel('Lag (ms)')
                ax.set_ylabel('Count')
            ax.set_title('Autocorrelogram')
            
            axes[1, 1].text(0.5, 0.5, 'No stimulus protocol', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 2].text(0.5, 0.5, 'No stimulus protocol', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
        
        # ========= Row 3: Phase-Locking =========
        
        if pl_result is not None:
            # 3-1: Phase histogram (best band)
            ax_polar = fig.add_subplot(3, 3, 7, projection='polar')
            axes[2, 0].remove()
            
            # 最も有意な帯域を見つける
            best_band = None
            best_result_pl = None
            best_mrl = 0
            
            for band_name, ch_results in pl_result.band_results.items():
                for lfp_ch, res in ch_results.items():
                    if res is not None and res.significant and res.mrl > best_mrl:
                        best_mrl = res.mrl
                        best_result_pl = res
                        best_band = f"{band_name} (LFP-Ch{lfp_ch})"
            
            if best_result_pl is not None:
                plot_phase_histogram(
                    best_result_pl.spike_phases,
                    mrl=best_result_pl.mrl,
                    preferred_phase=best_result_pl.preferred_phase,
                    title=f"Phase: {best_band}",
                    ax=ax_polar
                )
            else:
                ax_polar.set_title("No significant\nphase locking")
            
            # 3-2: MRL heatmap (bands × channels)
            ax = axes[2, 1]
            band_names = list(pl_result.band_results.keys())
            n_bands = len(band_names)
            n_lfp = self.n_lfp_channels
            
            mrl_matrix = np.zeros((n_bands, n_lfp))
            sig_matrix = np.zeros((n_bands, n_lfp), dtype=bool)
            
            for i, band in enumerate(band_names):
                if band in pl_result.band_results:
                    for j in range(n_lfp):
                        res = pl_result.band_results[band].get(j)
                        if res is not None:
                            mrl_matrix[i, j] = res.mrl
                            sig_matrix[i, j] = res.significant
            
            im = ax.imshow(mrl_matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.5)
            plt.colorbar(im, ax=ax, label='MRL', shrink=0.8)
            
            # 有意性マーカー
            for i in range(n_bands):
                for j in range(n_lfp):
                    if sig_matrix[i, j]:
                        ax.text(j, i, '*', ha='center', va='center',
                               fontsize=14, color='white', fontweight='bold')
            
            ax.set_yticks(range(n_bands))
            ax.set_yticklabels(band_names, fontsize=8)
            ax.set_xlabel('LFP Channel')
            ax.set_title('Phase Locking (MRL)')
            
            # 3-3: STA or condition comparison
            ax = axes[2, 2]
            if pl_result.sta is not None:
                sta = pl_result.sta[:, lfp_channel] if pl_result.sta.ndim > 1 else pl_result.sta
                ax.plot(pl_result.sta_time, sta, 'k-', linewidth=1.5)
                ax.axvline(0, color='red', linewidth=1, linestyle='--', label='Spike')
                ax.set_xlabel('Time from spike (ms)')
                ax.set_ylabel('LFP')
                ax.set_title(f'STA (LFP Ch{lfp_channel})')
                ax.legend(fontsize=8)
            elif pl_result.condition_results:
                # Condition comparison bar plot
                conditions = ['baseline', 'stim', 'post']
                colors = {'baseline': 'gray', 'stim': 'red', 'post': 'blue'}
                mrl_vals = []
                labels = []
                for cond in conditions:
                    res = pl_result.condition_results.get(cond)
                    if res is not None:
                        mrl_vals.append(res.mrl)
                        labels.append(cond)
                    else:
                        mrl_vals.append(0)
                        labels.append(cond)
                
                bars = ax.bar(labels, mrl_vals, 
                             color=[colors.get(l, 'gray') for l in labels],
                             alpha=0.7, edgecolor='black')
                ax.set_ylabel('MRL')
                ax.set_title('Phase locking by condition')
        else:
            for ax in axes[2, :]:
                ax.text(0.5, 0.5, 'Run analyze_unit() first',
                       ha='center', va='center', transform=ax.transAxes)
        
        # タイトル
        status = "SU" if not unit.is_mua else "MUA"
        fig.suptitle(f'{unit_key} ({status}, SNR={unit.snr:.1f}, n={unit.n_spikes})',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    # ============================================================
    # CSV保存
    # ============================================================
    
    def save_results_csv(self, output_dir, basename="analysis"):
        """
        全結果をCSV保存
        
        出力ファイル:
          - {basename}_phase_locking.csv: 周波数帯域別MRL/PPC
          - {basename}_conditions.csv: 条件別比較
          - {basename}_unit_summary.csv: ユニットサマリー
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # --- Phase Locking CSV ---
        pl_path = os.path.join(output_dir, f'{basename}_phase_locking.csv')
        with open(pl_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'unit_key', 'channel', 'unit_id', 'is_mua', 'n_spikes',
                'snr', 'isi_violation',
                'band', 'lfp_channel', 'mrl', 'ppc', 'p_value',
                'preferred_phase_deg', 'significant'
            ])
            
            for unit_key, pl_result in self.unit_results.items():
                for band_name, ch_results in pl_result.band_results.items():
                    for lfp_ch, res in ch_results.items():
                        if res is not None:
                            writer.writerow([
                                unit_key, pl_result.channel, pl_result.unit_id,
                                pl_result.is_mua, pl_result.n_spikes,
                                f'{pl_result.snr:.2f}', 
                                f'{pl_result.isi_violation_rate:.2f}',
                                band_name, lfp_ch,
                                f'{res.mrl:.4f}', f'{res.ppc:.4f}',
                                f'{res.p_value:.6f}',
                                f'{np.degrees(res.preferred_phase):.1f}',
                                res.significant
                            ])
        
        print(f"Phase locking CSV: {pl_path}")
        
        # --- Conditions CSV ---
        if any(r.condition_results for r in self.unit_results.values()):
            cond_path = os.path.join(output_dir, f'{basename}_conditions.csv')
            with open(cond_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'unit_key', 'channel', 'unit_id',
                    'condition', 'n_spikes', 'mrl', 'ppc', 'p_value',
                    'preferred_phase_deg', 'significant'
                ])
                
                for unit_key, pl_result in self.unit_results.items():
                    for cond, res in pl_result.condition_results.items():
                        if res is not None:
                            writer.writerow([
                                unit_key, pl_result.channel, pl_result.unit_id,
                                cond, res.n_spikes,
                                f'{res.mrl:.4f}', f'{res.ppc:.4f}',
                                f'{res.p_value:.6f}',
                                f'{np.degrees(res.preferred_phase):.1f}',
                                res.significant
                            ])
            
            print(f"Conditions CSV: {cond_path}")
        
        # --- Unit Summary CSV ---
        summary_path = os.path.join(output_dir, f'{basename}_unit_summary.csv')
        with open(summary_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'unit_key', 'channel', 'unit_id', 'is_mua',
                'n_spikes', 'snr', 'isi_violation',
                'best_band', 'best_mrl', 'best_phase_deg', 'best_significant'
            ])
            
            for unit_key, pl_result in self.unit_results.items():
                # 最良の帯域を見つける
                best_band = ''
                best_mrl = 0
                best_phase = 0
                best_sig = False
                
                for band_name, ch_results in pl_result.band_results.items():
                    for lfp_ch, res in ch_results.items():
                        if res is not None and res.mrl > best_mrl:
                            best_mrl = res.mrl
                            best_band = band_name
                            best_phase = np.degrees(res.preferred_phase)
                            best_sig = res.significant
                
                writer.writerow([
                    unit_key, pl_result.channel, pl_result.unit_id,
                    pl_result.is_mua, pl_result.n_spikes,
                    f'{pl_result.snr:.2f}', f'{pl_result.isi_violation_rate:.2f}',
                    best_band, f'{best_mrl:.4f}', f'{best_phase:.1f}', best_sig
                ])
        
        print(f"Unit summary CSV: {summary_path}")
    
    # ============================================================
    # バッチ可視化
    # ============================================================
    
    def plot_all_summaries(self, output_dir, lfp_channel=0):
        """全ユニットのサマリープロットを保存"""
        os.makedirs(output_dir, exist_ok=True)
        
        for unit_key, pl_result in self.unit_results.items():
            save_path = os.path.join(output_dir, f'{unit_key}_summary.png')
            self.plot_unit_summary(
                pl_result.channel, pl_result.unit_id,
                lfp_channel=lfp_channel,
                save_path=save_path
            )
            plt.close()
        
        print(f"Saved {len(self.unit_results)} summary plots to {output_dir}")
    
    def plot_population_summary(self, save_path=None, figsize=(14, 10)):
        """
        全ユニットの位相ロックサマリー
        
        Layout:
        [MRL distribution] [Phase distribution] [Band comparison ]
        [MRL vs SNR      ] [MRL vs n_spikes   ] [Condition comp  ]
        """
        if not self.unit_results:
            print("No results. Run analyze_all() first.")
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # データ収集
        all_mrl = {}  # band → list of MRL
        all_phase = {}  # band → list of phase
        unit_snr = []
        unit_mrl = []
        unit_n = []
        
        for unit_key, pl_result in self.unit_results.items():
            best_mrl_overall = 0
            for band_name, ch_results in pl_result.band_results.items():
                if band_name not in all_mrl:
                    all_mrl[band_name] = []
                    all_phase[band_name] = []
                
                for lfp_ch, res in ch_results.items():
                    if res is not None:
                        all_mrl[band_name].append(res.mrl)
                        if res.significant:
                            all_phase[band_name].append(res.preferred_phase)
                        if res.mrl > best_mrl_overall:
                            best_mrl_overall = res.mrl
            
            unit_snr.append(pl_result.snr)
            unit_mrl.append(best_mrl_overall)
            unit_n.append(pl_result.n_spikes)
        
        # 1-1: MRL distribution by band
        ax = axes[0, 0]
        band_names = list(all_mrl.keys())
        data = [all_mrl[b] for b in band_names]
        if data and any(len(d) > 0 for d in data):
            bp = ax.boxplot(data, labels=band_names, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('steelblue')
                patch.set_alpha(0.6)
        ax.set_ylabel('MRL')
        ax.set_title('MRL by frequency band')
        ax.tick_params(axis='x', rotation=45)
        
        # 1-2: Preferred phase (significant only)
        ax = axes[0, 1]
        colors = plt.cm.Set2(np.linspace(0, 1, len(band_names)))
        for i, band in enumerate(band_names):
            phases = all_phase.get(band, [])
            if len(phases) > 0:
                ax.scatter(np.degrees(phases), [i] * len(phases), 
                          color=colors[i], s=30, alpha=0.7, label=band)
        ax.set_xlabel('Preferred phase (degrees)')
        ax.set_yticks(range(len(band_names)))
        ax.set_yticklabels(band_names)
        ax.set_xlim(-180, 180)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title('Preferred phase (significant)')
        
        # 1-3: Band comparison (mean MRL)
        ax = axes[0, 2]
        mean_mrls = [np.mean(all_mrl[b]) if all_mrl[b] else 0 for b in band_names]
        sem_mrls = [np.std(all_mrl[b])/np.sqrt(len(all_mrl[b])) if len(all_mrl[b]) > 1 else 0 
                    for b in band_names]
        ax.bar(band_names, mean_mrls, yerr=sem_mrls, color='steelblue',
               alpha=0.7, edgecolor='black', capsize=3)
        ax.set_ylabel('Mean MRL')
        ax.set_title('Phase locking by band')
        ax.tick_params(axis='x', rotation=45)
        
        # 2-1: MRL vs SNR
        ax = axes[1, 0]
        ax.scatter(unit_snr, unit_mrl, s=30, alpha=0.7, color='steelblue')
        ax.set_xlabel('SNR')
        ax.set_ylabel('Best MRL')
        ax.set_title('MRL vs SNR')
        
        # 2-2: MRL vs n_spikes
        ax = axes[1, 1]
        ax.scatter(unit_n, unit_mrl, s=30, alpha=0.7, color='steelblue')
        ax.set_xlabel('Number of spikes')
        ax.set_ylabel('Best MRL')
        ax.set_title('MRL vs spike count')
        
        # 2-3: Condition comparison (if available)
        ax = axes[1, 2]
        cond_mrl = {'baseline': [], 'stim': [], 'post': []}
        for pl_result in self.unit_results.values():
            for cond, res in pl_result.condition_results.items():
                if res is not None and cond in cond_mrl:
                    cond_mrl[cond].append(res.mrl)
        
        if any(len(v) > 0 for v in cond_mrl.values()):
            conditions = ['baseline', 'stim', 'post']
            cond_colors = {'baseline': 'gray', 'stim': 'red', 'post': 'blue'}
            means = [np.mean(cond_mrl[c]) if cond_mrl[c] else 0 for c in conditions]
            sems = [np.std(cond_mrl[c])/np.sqrt(len(cond_mrl[c])) if len(cond_mrl[c]) > 1 else 0 
                    for c in conditions]
            ax.bar(conditions, means, yerr=sems,
                   color=[cond_colors[c] for c in conditions],
                   alpha=0.7, edgecolor='black', capsize=3)
            ax.set_ylabel('Mean MRL')
            ax.set_title('Phase locking by condition')
        else:
            ax.text(0.5, 0.5, 'No condition data', 
                   ha='center', va='center', transform=ax.transAxes)
        
        n_units = len(self.unit_results)
        n_sig = sum(1 for r in self.unit_results.values() 
                    for b, cr in r.band_results.items() 
                    for c, res in cr.items() 
                    if res is not None and res.significant)
        
        fig.suptitle(f'Population Summary ({n_units} units, {n_sig} significant pairs)',
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
