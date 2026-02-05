"""
stimulus.py - 刺激プロトコル解析

刺激応答の定量解析:
  - PSTH (Peri-Stimulus Time Histogram)
  - Trial別ラスタープロット
  - 刺激位置別応答率（適応解析）
  - Baseline/Stim/Post 条件マスク作成
  - Spike Triggered Average (STA)

使い方:
    from stimulus import StimulusProtocol
    
    protocol = StimulusProtocol(session.stim_times, session.trial_starts,
                                 n_stim_per_trial=10, stim_freq=10.0)
    
    # PSTH
    psth = protocol.compute_psth(spike_times, bin_ms=1.0)
    protocol.plot_psth(spike_times)
    
    # 適応
    adaptation = protocol.compute_adaptation(spike_times)
    
    # 条件マスク
    masks = protocol.create_condition_masks(lfp_times, fs)
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any


# ============================================================
# PSTH結果データクラス
# ============================================================

@dataclass
class PSTHResult:
    """PSTH解析結果"""
    bin_centers: np.ndarray      # ビン中心 (ms)
    counts: np.ndarray           # スパイク数
    rate: np.ndarray             # 発火率 (Hz)
    n_trials: int                # trial数
    bin_width_ms: float          # ビン幅 (ms)
    window: Tuple[float, float]  # 時間窓 (ms)
    
    @property
    def peak_latency_ms(self) -> float:
        """ピーク潜時 (ms)"""
        if len(self.rate) == 0:
            return 0.0
        # 刺激後のみで探す
        post_mask = self.bin_centers > 0
        if not np.any(post_mask):
            return 0.0
        post_rate = self.rate.copy()
        post_rate[~post_mask] = 0
        return float(self.bin_centers[np.argmax(post_rate)])
    
    @property
    def peak_rate(self) -> float:
        """ピーク発火率 (Hz)"""
        post_mask = self.bin_centers > 0
        if not np.any(post_mask):
            return 0.0
        return float(np.max(self.rate[post_mask]))
    
    @property
    def baseline_rate(self) -> float:
        """ベースライン発火率 (Hz, 刺激前)"""
        pre_mask = self.bin_centers < 0
        if not np.any(pre_mask):
            return 0.0
        return float(np.mean(self.rate[pre_mask]))


@dataclass
class AdaptationResult:
    """刺激位置別応答（適応）解析結果"""
    stim_positions: np.ndarray     # 刺激番号 (1, 2, ..., n)
    mean_spikes: np.ndarray        # 各位置の平均スパイク数
    std_spikes: np.ndarray         # 標準偏差
    response_prob: np.ndarray      # 応答確率 (0-1)
    n_trials: int                  # trial数
    response_window: Tuple[float, float]  # 応答検出窓 (ms)
    
    @property
    def adaptation_ratio(self) -> float:
        """適応率 = (1st応答 - 最小応答) / 1st応答"""
        if len(self.mean_spikes) == 0 or self.mean_spikes[0] == 0:
            return 0.0
        return float(1.0 - np.min(self.mean_spikes) / self.mean_spikes[0])


# ============================================================
# StimulusProtocol クラス
# ============================================================

class StimulusProtocol:
    """
    刺激プロトコルの管理と解析
    
    Parameters
    ----------
    stim_times : ndarray
        全刺激のタイムスタンプ（秒）
    trial_starts : ndarray
        各trialの開始タイムスタンプ（秒）
    n_stim_per_trial : int
        trial当たりの刺激数
    stim_freq : float
        刺激周波数 (Hz)
    iti : float
        Inter-trial interval (秒)。0の場合は自動推定。
    """
    
    def __init__(self, stim_times, trial_starts=None, 
                 n_stim_per_trial=10, stim_freq=10.0, iti=0.0):
        self.stim_times = np.asarray(stim_times, dtype=float)
        self.n_stim_per_trial = n_stim_per_trial
        self.stim_freq = stim_freq
        self.stim_period_ms = 1000.0 / stim_freq  # 100ms for 10Hz
        
        # Trial starts
        if trial_starts is not None and len(trial_starts) > 0:
            self.trial_starts = np.asarray(trial_starts, dtype=float)
        else:
            self.trial_starts = self._estimate_trial_starts()
        
        self.n_trials = len(self.trial_starts)
        
        # ITI
        if iti > 0:
            self.iti = iti
        elif self.n_trials > 1:
            stim_isi = np.diff(self.stim_times) * 1000
            gap_threshold = self.stim_period_ms * 3
            gaps = stim_isi[stim_isi > gap_threshold]
            self.iti = float(np.mean(gaps)) / 1000.0 if len(gaps) > 0 else 0.0
        else:
            self.iti = 0.0
        
        # Trial構造行列
        self.stim_matrix = self._build_stim_matrix()
    
    def _estimate_trial_starts(self) -> np.ndarray:
        """刺激タイミングからtrial開始を推定"""
        if len(self.stim_times) <= self.n_stim_per_trial:
            return np.array([self.stim_times[0]])
        
        isi = np.diff(self.stim_times) * 1000
        gap_threshold = self.stim_period_ms * 3
        gaps = np.where(isi > gap_threshold)[0]
        
        starts = [self.stim_times[0]]
        for g in gaps:
            starts.append(self.stim_times[g + 1])
        
        return np.array(starts)
    
    def _build_stim_matrix(self) -> np.ndarray:
        """刺激を (n_trials, n_stim_per_trial) の行列に整形"""
        try:
            return self.stim_times.reshape(self.n_trials, self.n_stim_per_trial)
        except ValueError:
            # 均等に分割できない場合
            matrix = []
            idx = 0
            for trial_start in self.trial_starts:
                trial_stims = self.stim_times[
                    (self.stim_times >= trial_start) & 
                    (self.stim_times < trial_start + self.n_stim_per_trial / self.stim_freq + 0.5)
                ][:self.n_stim_per_trial]
                if len(trial_stims) == self.n_stim_per_trial:
                    matrix.append(trial_stims)
            return np.array(matrix) if matrix else np.array([])
    
    def __repr__(self):
        return (f"StimulusProtocol({self.n_trials} trials × "
                f"{self.n_stim_per_trial} stim @ {self.stim_freq:.0f}Hz, "
                f"ITI={self.iti:.1f}s)")
    
    # ============================================================
    # PSTH
    # ============================================================
    
    def compute_psth(self, spike_times, bin_ms=1.0, 
                     window=(-50, 150)) -> PSTHResult:
        """
        Peri-Stimulus Time Histogram を計算
        
        Parameters
        ----------
        spike_times : ndarray
            スパイクタイムスタンプ（秒）
        bin_ms : float
            ビン幅 (ms)
        window : tuple (pre_ms, post_ms)
            刺激前後の時間窓 (ms)
        
        Returns
        -------
        result : PSTHResult
        """
        bins = np.arange(window[0], window[1] + bin_ms, bin_ms)
        counts = np.zeros(len(bins) - 1)
        
        for stim_t in self.stim_times:
            relative = (spike_times - stim_t) * 1000  # ms
            hist, _ = np.histogram(relative, bins=bins)
            counts += hist
        
        bin_centers = (bins[:-1] + bins[1:]) / 2
        n_stim = len(self.stim_times)
        
        # 発火率 = counts / (n_stim × bin_width_sec)
        rate = counts / (n_stim * bin_ms / 1000.0)
        
        return PSTHResult(
            bin_centers=bin_centers,
            counts=counts,
            rate=rate,
            n_trials=n_stim,
            bin_width_ms=bin_ms,
            window=window
        )
    
    # ============================================================
    # 適応（刺激位置別応答）
    # ============================================================
    
    def compute_adaptation(self, spike_times, 
                           response_window=(5, 50)) -> AdaptationResult:
        """
        刺激位置別の応答率を計算（適応解析）
        
        Parameters
        ----------
        spike_times : ndarray
            スパイクタイムスタンプ（秒）
        response_window : tuple (start_ms, end_ms)
            刺激後の応答検出窓 (ms)
        
        Returns
        -------
        result : AdaptationResult
        """
        win_start = response_window[0] / 1000.0
        win_end = response_window[1] / 1000.0
        
        n_pos = self.n_stim_per_trial
        trial_responses = np.zeros((self.n_trials, n_pos))
        
        for trial_i, trial_start in enumerate(self.trial_starts):
            # このtrialの刺激を取得
            trial_stims = self.stim_times[
                (self.stim_times >= trial_start) & 
                (self.stim_times < trial_start + n_pos / self.stim_freq + 0.5)
            ][:n_pos]
            
            for stim_i, stim_t in enumerate(trial_stims):
                # 応答窓内のスパイク数
                n_resp = np.sum(
                    (spike_times > stim_t + win_start) & 
                    (spike_times < stim_t + win_end)
                )
                trial_responses[trial_i, stim_i] = n_resp
        
        mean_spikes = np.mean(trial_responses, axis=0)
        std_spikes = np.std(trial_responses, axis=0)
        response_prob = np.mean(trial_responses > 0, axis=0)
        
        return AdaptationResult(
            stim_positions=np.arange(1, n_pos + 1),
            mean_spikes=mean_spikes,
            std_spikes=std_spikes,
            response_prob=response_prob,
            n_trials=self.n_trials,
            response_window=response_window
        )
    
    # ============================================================
    # 条件マスク
    # ============================================================
    
    def create_condition_masks(self, lfp_times, fs=None,
                               baseline_pre_sec=3.0,
                               post_duration_sec=3.0,
                               stim_margin_sec=0.0) -> Dict[str, np.ndarray]:
        """
        Baseline / Stim / Post の条件マスクを作成
        
        Parameters
        ----------
        lfp_times : ndarray
            LFPの時間軸（秒）
        fs : int or None
            LFPサンプリングレート（lfp_timesが等間隔でない場合に使用）
        baseline_pre_sec : float
            刺激開始前のベースライン期間（秒）
        post_duration_sec : float
            刺激終了後のポスト期間（秒）
        stim_margin_sec : float
            刺激マスクのマージン（秒）
        
        Returns
        -------
        masks : dict
            {'baseline': ndarray, 'stim': ndarray, 'post': ndarray}
        """
        n_samples = len(lfp_times)
        stim_mask = np.zeros(n_samples, dtype=bool)
        baseline_mask = np.zeros(n_samples, dtype=bool)
        post_mask = np.zeros(n_samples, dtype=bool)
        
        trial_ranges = self.trial_ranges if hasattr(self, 'trial_ranges_cached') else []
        
        # Trial構造から範囲を計算
        if self.stim_matrix.ndim == 2 and len(self.stim_matrix) > 0:
            for trial_stims in self.stim_matrix:
                t_start = float(trial_stims[0])
                t_end = float(trial_stims[-1])
                
                # Stim mask
                stim_mask |= (
                    (lfp_times >= t_start - stim_margin_sec) & 
                    (lfp_times <= t_end + stim_margin_sec)
                )
                
                # Baseline mask
                baseline_mask |= (
                    (lfp_times >= t_start - baseline_pre_sec) & 
                    (lfp_times < t_start)
                )
                
                # Post mask
                post_mask |= (
                    (lfp_times > t_end) & 
                    (lfp_times <= t_end + post_duration_sec)
                )
        
        return {
            'baseline': baseline_mask,
            'stim': stim_mask,
            'post': post_mask
        }
    
    # ============================================================
    # STA (Spike Triggered Average)
    # ============================================================
    
    def compute_sta(self, spike_times, lfp_data, lfp_times, fs,
                    window_ms=(-100, 100)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Spike-Triggered Average LFP を計算
        
        Parameters
        ----------
        spike_times : ndarray
            スパイクタイムスタンプ（秒）
        lfp_data : ndarray (n_samples,) or (n_samples, n_channels)
            LFPデータ
        lfp_times : ndarray
            LFP時間軸（秒）
        fs : int
            LFPサンプリングレート
        window_ms : tuple
            スパイク周辺の時間窓 (ms)
        
        Returns
        -------
        sta_time : ndarray
            STA時間軸 (ms)
        sta : ndarray
            STA波形 (n_samples,) or (n_samples, n_channels)
        """
        pre_samples = int(abs(window_ms[0]) / 1000 * fs)
        post_samples = int(window_ms[1] / 1000 * fs)
        total_samples = pre_samples + post_samples
        
        is_1d = lfp_data.ndim == 1
        if is_1d:
            lfp_data = lfp_data[:, np.newaxis]
        
        n_channels = lfp_data.shape[1]
        sta_sum = np.zeros((total_samples, n_channels))
        n_valid = 0
        
        spike_indices = np.searchsorted(lfp_times, spike_times)
        
        for idx in spike_indices:
            start = idx - pre_samples
            end = idx + post_samples
            if start >= 0 and end < len(lfp_data):
                sta_sum += lfp_data[start:end, :]
                n_valid += 1
        
        if n_valid > 0:
            sta = sta_sum / n_valid
        else:
            sta = np.zeros((total_samples, n_channels))
        
        sta_time = np.linspace(window_ms[0], window_ms[1], total_samples)
        
        if is_1d:
            sta = sta.flatten()
        
        return sta_time, sta
    
    # ============================================================
    # 可視化
    # ============================================================
    
    def plot_psth(self, spike_times, bin_ms=1.0, window=(-50, 150),
                  title=None, ax=None, color='steelblue'):
        """PSTHプロット"""
        psth = self.compute_psth(spike_times, bin_ms, window)
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        
        ax.bar(psth.bin_centers, psth.counts, width=bin_ms, 
               color=color, alpha=0.7, edgecolor='none')
        ax.axvline(0, color='red', linewidth=2, label='Stimulus')
        ax.set_xlabel('Time from stimulus (ms)')
        ax.set_ylabel('Spike count')
        
        title = title or f'PSTH (n={len(self.stim_times)} stim, peak={psth.peak_latency_ms:.0f}ms)'
        ax.set_title(title)
        ax.legend()
        
        return ax, psth
    
    def plot_raster_by_trial(self, spike_times, window_ms=1200, 
                              title=None, ax=None):
        """Trial別ラスタープロット"""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        
        for trial_i, trial_start in enumerate(self.trial_starts):
            trial_end = trial_start + window_ms / 1000.0
            
            # スパイク
            trial_spikes = spike_times[
                (spike_times >= trial_start) & (spike_times < trial_end)
            ]
            relative = (trial_spikes - trial_start) * 1000  # ms
            ax.scatter(relative, np.ones_like(relative) * trial_i,
                      s=3, c='black', marker='|', linewidths=0.5)
            
            # 刺激タイミング
            trial_stims = self.stim_times[
                (self.stim_times >= trial_start) & (self.stim_times < trial_end)
            ]
            for st in trial_stims:
                ax.axvline((st - trial_start) * 1000, color='red', 
                          alpha=0.3, linewidth=0.5)
        
        ax.set_xlabel('Time from trial start (ms)')
        ax.set_ylabel('Trial #')
        ax.set_xlim(-50, window_ms)
        ax.set_title(title or f'Raster by trial (n={self.n_trials})')
        
        return ax
    
    def plot_adaptation(self, spike_times, response_window=(5, 50),
                        title=None, ax=None, color='steelblue'):
        """刺激位置別応答率プロット"""
        adapt = self.compute_adaptation(spike_times, response_window)
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        
        ax.bar(adapt.stim_positions, adapt.mean_spikes,
               yerr=adapt.std_spikes / np.sqrt(adapt.n_trials),
               color=color, edgecolor='black', alpha=0.7, capsize=3)
        ax.set_xlabel('Stimulus # in trial')
        ax.set_ylabel('Mean spikes per stimulus')
        ax.set_xticks(adapt.stim_positions)
        ax.set_title(title or f'Adaptation (ratio={adapt.adaptation_ratio:.2f})')
        
        return ax, adapt
    
    def plot_summary(self, spike_times, lfp_data=None, lfp_times=None, 
                     fs=None, figsize=(16, 10), save_path=None):
        """
        刺激応答サマリー（6パネル）
        
        Panels:
        1. PSTH
        2. Raster by trial
        3. Adaptation
        4. STA (if LFP provided)
        5. Response probability
        6. Text summary
        """
        has_lfp = lfp_data is not None and lfp_times is not None and fs is not None
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # 1. PSTH
        self.plot_psth(spike_times, ax=axes[0, 0])
        
        # 2. Raster by trial
        self.plot_raster_by_trial(spike_times, ax=axes[0, 1])
        
        # 3. Adaptation
        _, adapt = self.plot_adaptation(spike_times, ax=axes[0, 2])
        
        # 4. STA or zoomed PSTH
        if has_lfp:
            sta_time, sta = self.compute_sta(spike_times, lfp_data, lfp_times, fs)
            ax = axes[1, 0]
            if sta.ndim == 1:
                ax.plot(sta_time, sta, 'k-', linewidth=1)
            else:
                ax.plot(sta_time, sta[:, 0], 'k-', linewidth=1)
            ax.axvline(0, color='red', linewidth=1, linestyle='--')
            ax.set_xlabel('Time from spike (ms)')
            ax.set_ylabel('LFP amplitude')
            ax.set_title('Spike-Triggered Average LFP')
        else:
            # Fine PSTH
            self.plot_psth(spike_times, bin_ms=0.5, window=(-10, 30),
                          ax=axes[1, 0], color='darkorange')
            axes[1, 0].set_title('PSTH (zoomed, 0.5ms bins)')
        
        # 5. Response probability
        ax = axes[1, 1]
        ax.bar(adapt.stim_positions, adapt.response_prob * 100,
               color='coral', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Stimulus # in trial')
        ax.set_ylabel('Response probability (%)')
        ax.set_xticks(adapt.stim_positions)
        ax.set_title('Response probability')
        ax.set_ylim(0, 100)
        
        # 6. Summary text
        ax = axes[1, 2]
        ax.axis('off')
        
        psth = self.compute_psth(spike_times)
        
        summary_text = (
            f"=== Stimulus Response Summary ===\n\n"
            f"Protocol:\n"
            f"  {self.n_trials} trials × {self.n_stim_per_trial} stim\n"
            f"  @ {self.stim_freq:.0f} Hz, ITI = {self.iti:.1f}s\n\n"
            f"Response:\n"
            f"  Peak latency: {psth.peak_latency_ms:.1f} ms\n"
            f"  Peak rate: {psth.peak_rate:.1f} Hz\n"
            f"  Baseline rate: {psth.baseline_rate:.1f} Hz\n\n"
            f"Adaptation:\n"
            f"  Ratio: {adapt.adaptation_ratio:.2f}\n"
            f"  1st stim: {adapt.mean_spikes[0]:.2f} spikes\n"
            f"  Min: {np.min(adapt.mean_spikes):.2f} @ stim #{np.argmin(adapt.mean_spikes)+1}\n"
        )
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.suptitle(f'Stimulus Response Analysis', fontsize=13)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
