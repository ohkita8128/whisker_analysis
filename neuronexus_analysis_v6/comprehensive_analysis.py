"""
comprehensive_analysis.py - 全チャンネル統合解析モジュール

NeuroNexus A1x16 ラミナープローブの16チャンネルを横断的に解析し、
深度プロファイル・CSD・クロスチャンネル比較を行う。

=== 3つの解析レイヤー ===

  1. スパイクソーティング概要
     - 全チャンネル波形一覧
     - 品質メトリクステーブル
     - 発火率の深度プロファイル
     - 条件別（Baseline/Stim/Post）発火率

  2. LFP詳細解析
     - CSD (Current Source Density)
     - Trial平均CSD
     - パワースペクトル深度プロファイル
     - チャンネル間コヒーレンス行列

  3. スパイク-LFP統合解析
     - 位相ロック深度プロファイル
     - STA深度プロファイル
     - グランドサマリー
     - テキストレポート生成

使い方:
    from comprehensive_analysis import ComprehensiveAnalyzer

    ca = ComprehensiveAnalyzer(
        session, lfp_result, sorting_results,
        protocol=protocol,
        spike_lfp_analyzer=analyzer,
    )

    # 個別プロット
    ca.plot_spike_overview()
    ca.plot_csd_summary()
    ca.plot_depth_profiles()

    # 全保存
    ca.save_all("output/comprehensive/")
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal
from typing import Optional, List, Tuple, Dict, Any
import os
import csv
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 定数
# ============================================================

# NeuroNexus A1x16-5mm-50-703 の深度（µm）
ELECTRODE_SPACING_UM = 50
N_ELECTRODES = 16
DEPTHS_UM = np.arange(N_ELECTRODES) * ELECTRODE_SPACING_UM  # 0, 50, ..., 750

# マウスS1BF 皮質層の概算境界（µm）
LAYER_BOUNDARIES = {
    'L1':    (0, 100),
    'L2/3':  (100, 350),
    'L4':    (350, 500),
    'L5':    (500, 650),
    'L6':    (650, 800),
}

LAYER_COLORS = {
    'L1':   '#E8E8E8',
    'L2/3': '#B3D9FF',
    'L4':   '#FFD9B3',
    'L5':   '#B3FFB3',
    'L6':   '#FFB3B3',
}


def get_layer_label(depth_um: float) -> str:
    """深度から皮質層名を返す"""
    for name, (lo, hi) in LAYER_BOUNDARIES.items():
        if lo <= depth_um < hi:
            return name
    return 'L6'  # 750µm以上


# ============================================================
# メインクラス
# ============================================================

class ComprehensiveAnalyzer:
    """
    全チャンネル統合解析

    Parameters
    ----------
    session : RecordingSession
        data_loader.load_plx_session() の出力
    lfp_result : dict
        run_analysis.step2_process_lfp() の出力
    sorting_results : dict
        {channel: ChannelSortResult}
    protocol : StimulusProtocol or None
        stimulus.StimulusProtocol
    spike_lfp_analyzer : SpikeLFPAnalyzer or None
        spike_lfp_analysis.SpikeLFPAnalyzer (step5後)
    """

    def __init__(self, session, lfp_result, sorting_results,
                 protocol=None, spike_lfp_analyzer=None):

        self.session = session
        self.lfp_result = lfp_result
        self.sorting = sorting_results
        self.protocol = protocol
        self.sla = spike_lfp_analyzer  # SpikeLFPAnalyzer

        # LFP関連のショートカット
        self.lfp = lfp_result['lfp_cleaned']
        self.lfp_times = lfp_result['lfp_times']
        self.fs = lfp_result['fs']
        self.good_channels = lfp_result.get('good_channels', list(range(self.lfp.shape[1])))
        self.original_ch = lfp_result.get('original_ch_numbers', self.good_channels)
        self.noise_mask = lfp_result.get('noise_mask')

        # 深度情報
        self.n_lfp_ch = self.lfp.shape[1]
        self.depths = DEPTHS_UM[self.good_channels] if len(self.good_channels) == self.n_lfp_ch \
            else DEPTHS_UM[:self.n_lfp_ch]

        # CSD キャッシュ
        self._csd = None
        self._csd_depths = None
        self._trial_avg_csd = None
        self._trial_avg_csd_times = None

    # ============================================================
    # ユーティリティ
    # ============================================================

    def _draw_layer_background(self, ax, orientation='horizontal'):
        """層境界の背景色を描画"""
        max_depth = self.depths[-1] + ELECTRODE_SPACING_UM / 2
        for name, (lo, hi) in LAYER_BOUNDARIES.items():
            hi = min(hi, max_depth + 50)
            if lo > max_depth + 50:
                continue
            color = LAYER_COLORS[name]
            if orientation == 'horizontal':
                ax.axhspan(lo - 25, hi - 25, color=color, alpha=0.25, zorder=0)
                ax.text(ax.get_xlim()[1], (lo + hi) / 2 - 25, f' {name}',
                        fontsize=8, va='center', ha='left', color='gray',
                        clip_on=False)
            else:
                ax.axvspan(lo - 25, hi - 25, color=color, alpha=0.25, zorder=0)

    def _get_clean_mask(self) -> np.ndarray:
        """ノイズ区間を除いたマスク"""
        if self.noise_mask is not None and len(self.noise_mask) == len(self.lfp_times):
            return ~self.noise_mask
        return np.ones(len(self.lfp_times), dtype=bool)

    # ================================================================
    #  1. スパイクソーティング概要
    # ================================================================

    def spike_sorting_table(self) -> List[Dict]:
        """
        全チャンネル・全ユニットの品質テーブルを生成

        Returns
        -------
        rows : list of dict
            各行は channel, depth, unit_id, n_spikes, snr,
            isi_violation, is_mua, is_noise, quality, firing_rate
        """
        rows = []
        for ch, result in sorted(self.sorting.items()):
            depth = DEPTHS_UM[ch] if ch < N_ELECTRODES else ch * ELECTRODE_SPACING_UM
            for unit in result.units:
                duration = (unit.spike_times[-1] - unit.spike_times[0]) if len(unit.spike_times) > 1 else 1.0
                fr = unit.n_spikes / duration if duration > 0 else 0

                # 品質分類
                if unit.is_noise:
                    quality = 'Noise'
                elif unit.is_mua:
                    quality = 'MUA'
                elif unit.isi_violation_rate < 0.5 and unit.snr > 6:
                    quality = 'Excellent SU'
                elif unit.isi_violation_rate < 2 and unit.snr > 3:
                    quality = 'Good SU'
                else:
                    quality = 'SU'

                rows.append({
                    'channel': ch,
                    'depth_um': depth,
                    'layer': get_layer_label(depth),
                    'unit_id': unit.unit_id,
                    'n_spikes': unit.n_spikes,
                    'snr': round(unit.snr, 2),
                    'isi_violation': round(unit.isi_violation_rate, 2),
                    'mean_amplitude': round(unit.mean_amplitude, 5),
                    'is_mua': unit.is_mua,
                    'is_noise': unit.is_noise,
                    'quality': quality,
                    'firing_rate_hz': round(fr, 2),
                })
        return rows

    def plot_spike_overview(self, save_path=None, figsize=(20, 16)):
        """
        全チャンネルのスパイクソーティング概要

        左: 各チャンネルの波形一覧（深度順）
        右上: 品質メトリクス散布図
        右下: 発火率深度プロファイル
        """
        sorted_channels = sorted(self.sorting.keys())
        n_ch = len(sorted_channels)
        if n_ch == 0:
            print("No sorting results available.")
            return None

        fig = plt.figure(figsize=figsize)
        gs = GridSpec(4, 6, figure=fig, hspace=0.35, wspace=0.4)

        # --- 左: 波形一覧 (4行 × 4列) ---
        n_cols_wf = 4
        n_rows_wf = 4
        for idx, ch in enumerate(sorted_channels[:n_rows_wf * n_cols_wf]):
            row = idx // n_cols_wf
            col = idx % n_cols_wf
            ax = fig.add_subplot(gs[row, col])

            result = self.sorting[ch]
            depth = DEPTHS_UM[ch] if ch < N_ELECTRODES else 0

            if len(result.units) == 0:
                ax.text(0.5, 0.5, 'No units', ha='center', va='center',
                        transform=ax.transAxes, color='gray')
            else:
                time_ms = result.waveform_time_ms
                if time_ms is None:
                    time_ms = np.arange(result.units[0].waveforms.shape[1]) / result.fs * 1000

                for unit in result.units:
                    if unit.is_noise:
                        continue
                    n_show = min(30, len(unit.waveforms))
                    indices = np.random.choice(len(unit.waveforms), n_show, replace=False)
                    for i in indices:
                        ax.plot(time_ms, unit.waveforms[i],
                                color=unit.color, alpha=0.08, linewidth=0.5)
                    ax.plot(time_ms, np.mean(unit.waveforms, axis=0),
                            color=unit.color, linewidth=2,
                            label=f'U{unit.unit_id}(n={unit.n_spikes})')

                ax.axhline(0, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
                ax.legend(fontsize=5, loc='lower right', framealpha=0.5)

            ax.set_title(f'Ch{ch} ({depth}µm, {get_layer_label(depth)})', fontsize=9)
            if row == n_rows_wf - 1:
                ax.set_xlabel('ms', fontsize=7)
            if col == 0:
                ax.set_ylabel('Amp', fontsize=7)
            ax.tick_params(labelsize=6)

        # --- 右上: SNR vs ISI violation ---
        ax_quality = fig.add_subplot(gs[0:2, 4:6])
        table_data = self.spike_sorting_table()
        valid = [r for r in table_data if not r['is_noise']]

        if valid:
            snrs = [r['snr'] for r in valid]
            isis = [r['isi_violation'] for r in valid]
            depths_pts = [r['depth_um'] for r in valid]
            is_mua = [r['is_mua'] for r in valid]

            sc = ax_quality.scatter(
                snrs, isis, c=depths_pts, cmap='viridis',
                s=50, edgecolors='black', linewidths=0.5, alpha=0.8,
                marker='o', zorder=5)

            # MUA を別マーカーで重ねる
            for i, m in enumerate(is_mua):
                if m:
                    ax_quality.scatter(snrs[i], isis[i], s=80,
                                       facecolors='none', edgecolors='red',
                                       linewidths=1.5, zorder=6)

            plt.colorbar(sc, ax=ax_quality, label='Depth (µm)', shrink=0.8)

            # 品質ゾーン
            ax_quality.axhline(2, color='orange', linestyle='--', alpha=0.5, label='ISI 2%')
            ax_quality.axhline(0.5, color='green', linestyle='--', alpha=0.5, label='ISI 0.5%')
            ax_quality.axvline(3, color='blue', linestyle='--', alpha=0.5, label='SNR 3')
            ax_quality.axvline(6, color='green', linestyle='--', alpha=0.5, label='SNR 6')

        ax_quality.set_xlabel('SNR')
        ax_quality.set_ylabel('ISI violation (%)')
        ax_quality.set_title('Unit Quality (red ring=MUA)')
        ax_quality.legend(fontsize=7)

        # --- 右下: 発火率深度プロファイル ---
        ax_fr = fig.add_subplot(gs[2:4, 4:6])
        if valid:
            for r in valid:
                marker = 's' if r['is_mua'] else 'o'
                color = 'gray' if r['is_mua'] else 'steelblue'
                ax_fr.scatter(r['firing_rate_hz'], r['depth_um'],
                              s=r['n_spikes'] / 10 + 20,
                              marker=marker, color=color, alpha=0.7,
                              edgecolors='black', linewidths=0.5)

            self._draw_layer_background(ax_fr, 'horizontal')
            ax_fr.invert_yaxis()
            ax_fr.set_xlabel('Firing rate (Hz)')
            ax_fr.set_ylabel('Depth (µm)')
            ax_fr.set_title('Firing Rate Depth Profile\n(size ∝ spike count)')

        n_su = sum(1 for r in table_data if r['quality'] not in ('Noise', 'MUA'))
        n_mua = sum(1 for r in table_data if r['quality'] == 'MUA')
        n_noise = sum(1 for r in table_data if r['quality'] == 'Noise')
        fig.suptitle(
            f'Spike Sorting Overview — {len(sorted_channels)} channels, '
            f'{n_su} SU + {n_mua} MUA + {n_noise} Noise',
            fontsize=14, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        return fig

    def plot_firing_rate_by_condition(self, save_path=None, figsize=(10, 8), fig=None):
        """
        条件別（Baseline / Stim / Post）発火率の深度プロファイル
        """
        if self.protocol is None:
            print("StimulusProtocol required for condition analysis.")
            return None

        masks = self.protocol.create_condition_masks(self.lfp_times, self.fs)

        if fig is None:
            fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
        else:
            axes = fig.subplots(1, 3, sharey=True)
        conditions = [('baseline', 'gray'), ('stim', 'red'), ('post', 'blue')]

        for ax, (cond, color) in zip(axes, conditions):
            mask = masks[cond]
            if self.noise_mask is not None:
                mask = mask & ~self.noise_mask
            duration = np.sum(mask) / self.fs

            for ch, result in sorted(self.sorting.items()):
                depth = DEPTHS_UM[ch] if ch < N_ELECTRODES else 0
                for unit in result.units:
                    if unit.is_noise:
                        continue
                    # mask 内のスパイク数
                    spike_indices = np.searchsorted(self.lfp_times, unit.spike_times)
                    spike_indices = np.clip(spike_indices, 0, len(mask) - 1)
                    n_in = np.sum(mask[spike_indices])
                    fr = n_in / duration if duration > 0 else 0

                    marker = 's' if unit.is_mua else 'o'
                    ax.scatter(fr, depth, s=40, marker=marker, color=color,
                               alpha=0.7, edgecolors='black', linewidths=0.5)

            self._draw_layer_background(ax, 'horizontal')
            ax.invert_yaxis()
            ax.set_xlabel('Firing rate (Hz)')
            ax.set_title(f'{cond.capitalize()}\n({duration:.1f}s)')

        axes[0].set_ylabel('Depth (µm)')
        fig.suptitle('Firing Rate by Condition × Depth', fontsize=13, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    # ================================================================
    #  2. LFP 詳細解析
    # ================================================================

    def compute_csd(self, sigma=1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Current Source Density（CSD）を計算

        二次空間微分法:
            CSD[i] = -σ * (Φ[i-1] - 2Φ[i] + Φ[i+1]) / Δz²

        Parameters
        ----------
        sigma : float
            導電率（相対値、デフォルト=1）

        Returns
        -------
        csd : ndarray (n_samples, n_channels-2)
        csd_depths : ndarray (n_channels-2,)
        """
        if self._csd is not None:
            return self._csd, self._csd_depths

        dz = ELECTRODE_SPACING_UM * 1e-6  # メートル変換
        n_samples, n_ch = self.lfp.shape

        if n_ch < 3:
            raise ValueError("CSD computation requires at least 3 channels")

        csd = np.zeros((n_samples, n_ch - 2))
        for i in range(1, n_ch - 1):
            csd[:, i - 1] = -sigma * (
                self.lfp[:, i - 1] - 2 * self.lfp[:, i] + self.lfp[:, i + 1]
            ) / dz ** 2

        csd_depths = self.depths[1:-1]

        self._csd = csd
        self._csd_depths = csd_depths
        return csd, csd_depths

    def compute_trial_averaged_csd(self, window_ms=(-50, 200),
                                    sigma=1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Trial平均CSD（刺激タイミング周辺）

        Parameters
        ----------
        window_ms : tuple
            刺激前後の時間窓 (ms)
        sigma : float
            CSD導電率パラメータ

        Returns
        -------
        avg_csd : ndarray (n_time, n_ch-2)
        csd_times_ms : ndarray
        csd_depths : ndarray
        """
        if self._trial_avg_csd is not None:
            return self._trial_avg_csd, self._trial_avg_csd_times, self._csd_depths

        csd, csd_depths = self.compute_csd(sigma)

        if self.protocol is None or len(self.protocol.stim_times) == 0:
            raise ValueError("StimulusProtocol required for trial-averaged CSD")

        pre_samples = int(abs(window_ms[0]) / 1000 * self.fs)
        post_samples = int(window_ms[1] / 1000 * self.fs)
        total = pre_samples + post_samples
        n_csd_ch = csd.shape[1]

        csd_sum = np.zeros((total, n_csd_ch))
        n_valid = 0

        for stim_t in self.protocol.stim_times:
            idx = int(stim_t * self.fs)
            start = idx - pre_samples
            end = idx + post_samples
            if start >= 0 and end < len(csd):
                csd_sum += csd[start:end, :]
                n_valid += 1

        avg_csd = csd_sum / max(n_valid, 1)
        csd_times_ms = np.linspace(window_ms[0], window_ms[1], total)

        self._trial_avg_csd = avg_csd
        self._trial_avg_csd_times = csd_times_ms
        return avg_csd, csd_times_ms, csd_depths

    def plot_csd_summary(self, window_ms=(-50, 200), save_path=None, figsize=(16, 10)):
        """
        CSD サマリー（3パネル）

        左: Trial平均CSD ヒートマップ
        中: Trial平均LFP（深度別重ね書き）
        右: 刺激応答時のCSD断面（ピーク時刻）
        """
        avg_csd, csd_times, csd_depths = self.compute_trial_averaged_csd(window_ms)

        # Trial平均LFP も計算
        pre_samples = int(abs(window_ms[0]) / 1000 * self.fs)
        post_samples = int(window_ms[1] / 1000 * self.fs)
        total = pre_samples + post_samples
        lfp_sum = np.zeros((total, self.n_lfp_ch))
        n_valid = 0
        for stim_t in self.protocol.stim_times:
            idx = int(stim_t * self.fs)
            start = idx - pre_samples
            end = idx + post_samples
            if start >= 0 and end < len(self.lfp):
                lfp_sum += self.lfp[start:end, :]
                n_valid += 1
        avg_lfp = lfp_sum / max(n_valid, 1)
        lfp_times_ms = np.linspace(window_ms[0], window_ms[1], total)

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # --- 左: CSD ヒートマップ ---
        ax = axes[0]
        vmax = np.percentile(np.abs(avg_csd), 95)
        im = ax.pcolormesh(csd_times, csd_depths, avg_csd.T,
                           cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
        ax.axvline(0, color='black', linewidth=2, linestyle='--')
        ax.set_xlabel('Time from stimulus (ms)')
        ax.set_ylabel('Depth (µm)')
        ax.set_title('Trial-Averaged CSD')
        ax.invert_yaxis()
        plt.colorbar(im, ax=ax, label='CSD (a.u.)', shrink=0.8)

        # --- 中: Trial平均LFP ---
        ax = axes[1]
        spacing = np.max(np.abs(avg_lfp)) * 0.8
        for i in range(self.n_lfp_ch):
            offset = -self.depths[i] / ELECTRODE_SPACING_UM * spacing
            ax.plot(lfp_times_ms, avg_lfp[:, i] + offset,
                    'k-', linewidth=0.8)
            ax.text(window_ms[1] + 2, offset,
                    f'D{self.good_channels[i]}({self.depths[i]}µm)',
                    fontsize=6, va='center')
        ax.axvline(0, color='red', linewidth=1.5, linestyle='--')
        ax.set_xlabel('Time from stimulus (ms)')
        ax.set_ylabel('Depth (stacked)')
        ax.set_title('Trial-Averaged LFP')
        ax.set_yticks([])

        # --- 右: CSD 深度断面（ピーク時刻） ---
        ax = axes[2]
        # 刺激後のピーク時刻を探す
        post_mask = csd_times > 3  # 刺激後3ms以降
        if np.any(post_mask):
            csd_post = avg_csd[post_mask, :]
            peak_idx = np.unravel_index(np.argmax(np.abs(csd_post)), csd_post.shape)
            peak_time = csd_times[post_mask][peak_idx[0]]
            peak_frame = np.argmin(np.abs(csd_times - peak_time))

            csd_slice = avg_csd[peak_frame, :]
            ax.barh(csd_depths, csd_slice, height=ELECTRODE_SPACING_UM * 0.8,
                    color=['red' if v > 0 else 'blue' for v in csd_slice],
                    alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.axvline(0, color='gray', linestyle='--')
            ax.invert_yaxis()
            ax.set_xlabel('CSD amplitude')
            ax.set_ylabel('Depth (µm)')
            ax.set_title(f'CSD Profile @ {peak_time:.1f}ms\n(red=source, blue=sink)')
            self._draw_layer_background(ax, 'horizontal')

        fig.suptitle(f'Current Source Density — {n_valid} stimuli averaged',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    def compute_power_by_depth(self, nperseg=1024) -> Tuple[np.ndarray, np.ndarray]:
        """
        チャンネル別パワースペクトル密度

        Returns
        -------
        freqs : ndarray
        psd : ndarray (n_freqs, n_channels)
        """
        clean = self._get_clean_mask()
        data = self.lfp[clean, :]
        freqs, psd = signal.welch(data, fs=self.fs, nperseg=nperseg, axis=0)
        return freqs, psd

    def plot_lfp_depth_analysis(self, save_path=None, figsize=(18, 12)):
        """
        LFP深度解析（4パネル）

        [パワースペクトル深度ヒートマップ] [帯域別パワー深度プロファイル]
        [コヒーレンス行列（theta）     ] [コヒーレンス行列（gamma）    ]
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # --- パワースペクトル深度ヒートマップ ---
        freqs, psd = self.compute_power_by_depth()
        ax = axes[0, 0]
        freq_mask = freqs <= 100
        psd_db = 10 * np.log10(psd[freq_mask, :] + 1e-10)
        im = ax.pcolormesh(freqs[freq_mask], self.depths, psd_db.T,
                           cmap='YlOrRd', shading='auto')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Depth (µm)')
        ax.set_title('Power Spectral Density by Depth')
        ax.invert_yaxis()
        plt.colorbar(im, ax=ax, label='Power (dB)', shrink=0.8)

        # --- 帯域別パワー深度プロファイル ---
        ax = axes[0, 1]
        bands = {'theta': (4, 8), 'alpha': (8, 14), 'beta': (14, 30),
                 'low_γ': (30, 50), 'high_γ': (50, 100)}
        colors_band = ['blue', 'green', 'orange', 'red', 'purple']

        for (bname, (lo, hi)), bc in zip(bands.items(), colors_band):
            bmask = (freqs >= lo) & (freqs < hi)
            band_power = np.mean(psd[bmask, :], axis=0)
            # 正規化（最大値=1）
            bp_norm = band_power / band_power.max() if band_power.max() > 0 else band_power
            ax.plot(bp_norm, self.depths, 'o-', color=bc, label=bname, linewidth=1.5, markersize=4)

        ax.invert_yaxis()
        self._draw_layer_background(ax, 'horizontal')
        ax.set_xlabel('Normalized power')
        ax.set_ylabel('Depth (µm)')
        ax.set_title('Band Power Depth Profile')
        ax.legend(fontsize=8)

        # --- コヒーレンス行列 ---
        for idx, (band_name, freq_range) in enumerate([('theta (4-8Hz)', (4, 8)),
                                                         ('gamma (30-80Hz)', (30, 80))]):
            ax = axes[1, idx]
            coh_matrix = self._compute_coherence_matrix(freq_range)
            im = ax.imshow(coh_matrix, cmap='viridis', vmin=0, vmax=1,
                           origin='upper', aspect='equal')
            ax.set_xticks(range(self.n_lfp_ch))
            ax.set_yticks(range(self.n_lfp_ch))
            xlabels = [f'{d}' for d in self.depths]
            ax.set_xticklabels(xlabels, fontsize=6, rotation=45)
            ax.set_yticklabels(xlabels, fontsize=6)
            ax.set_xlabel('Depth (µm)')
            ax.set_ylabel('Depth (µm)')
            ax.set_title(f'Coherence: {band_name}')
            plt.colorbar(im, ax=ax, label='Coherence', shrink=0.8)

        fig.suptitle('LFP Depth Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    def _compute_coherence_matrix(self, freq_range, nperseg=1024) -> np.ndarray:
        """チャンネル間コヒーレンスの行列を計算"""
        clean = self._get_clean_mask()
        data = self.lfp[clean, :]
        n_ch = data.shape[1]
        coh_matrix = np.eye(n_ch)

        for i in range(n_ch):
            for j in range(i + 1, n_ch):
                freqs, coh = signal.coherence(data[:, i], data[:, j],
                                               fs=self.fs, nperseg=nperseg)
                fmask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
                mean_coh = np.mean(coh[fmask])
                coh_matrix[i, j] = mean_coh
                coh_matrix[j, i] = mean_coh

        return coh_matrix

    def plot_lfp_overview(self, t_range=None, save_path=None, figsize=(16, 10)):
        """
        全チャンネルLFP波形の概要プロット

        Parameters
        ----------
        t_range : tuple (t_start, t_end) or None
            表示する時間範囲（秒）。None=最初の5秒
        """
        if t_range is None:
            t_range = (self.lfp_times[0], min(self.lfp_times[0] + 5, self.lfp_times[-1]))

        tmask = (self.lfp_times >= t_range[0]) & (self.lfp_times <= t_range[1])
        times = self.lfp_times[tmask]

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        spacing = np.percentile(np.abs(self.lfp), 98) * 1.5

        for i in range(self.n_lfp_ch):
            offset = -i * spacing
            ax.plot(times, self.lfp[tmask, i] + offset,
                    linewidth=0.5, color='black')
            ax.text(t_range[0] - 0.05, offset,
                    f'{self.depths[i]}µm\n{get_layer_label(self.depths[i])}',
                    fontsize=7, ha='right', va='center', color='steelblue')

        # 刺激タイミング
        if self.protocol is not None:
            for st in self.protocol.stim_times:
                if t_range[0] <= st <= t_range[1]:
                    ax.axvline(st, color='red', alpha=0.3, linewidth=0.5)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Depth (stacked channels)')
        ax.set_yticks([])
        ax.set_title(f'LFP Overview ({t_range[0]:.1f}–{t_range[1]:.1f}s)')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    # ================================================================
    #  3. スパイク-LFP 統合解析
    # ================================================================

    def plot_phase_locking_depth(self, save_path=None, figsize=(16, 8)):
        """
        位相ロック深度プロファイル

        左: MRL ヒートマップ（帯域 × ユニット深度）
        右: 各帯域のMRL深度プロファイル
        """
        if self.sla is None or not self.sla.unit_results:
            print("SpikeLFPAnalyzer with results required. Run step5 first.")
            return None

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # データ収集: ユニットごとの最良MRL（LFPチャンネル横断）
        unit_depths = []
        unit_labels = []
        band_names = list(self.sla.freq_bands.keys())
        mrl_data = {b: [] for b in band_names}

        for unit_key, pl_result in self.sla.unit_results.items():
            ch = pl_result.channel
            depth = DEPTHS_UM[ch] if ch < N_ELECTRODES else 0
            unit_depths.append(depth)
            unit_labels.append(unit_key)

            for band in band_names:
                best_mrl = 0
                if band in pl_result.band_results:
                    for lfp_ch, res in pl_result.band_results[band].items():
                        if res is not None and res.mrl > best_mrl:
                            best_mrl = res.mrl
                mrl_data[band].append(best_mrl)

        if len(unit_depths) == 0:
            print("No phase locking results.")
            return None

        unit_depths = np.array(unit_depths)
        n_units = len(unit_depths)
        n_bands = len(band_names)

        # --- 左: MRL ヒートマップ ---
        ax = axes[0]
        mrl_matrix = np.array([mrl_data[b] for b in band_names])  # (n_bands, n_units)

        # ユニットを深度でソート
        sort_idx = np.argsort(unit_depths)
        mrl_sorted = mrl_matrix[:, sort_idx]
        labels_sorted = [unit_labels[i] for i in sort_idx]
        depths_sorted = unit_depths[sort_idx]

        im = ax.imshow(mrl_sorted, aspect='auto', cmap='YlOrRd', vmin=0,
                       vmax=max(0.3, np.percentile(mrl_sorted, 95)))
        ax.set_yticks(range(n_bands))
        ax.set_yticklabels(band_names, fontsize=9)
        ax.set_xticks(range(n_units))
        ax.set_xticklabels([f'{labels_sorted[i]}\n{depths_sorted[i]}µm'
                            for i in range(n_units)],
                           fontsize=6, rotation=45, ha='right')
        ax.set_title('Phase Locking (best MRL per unit)')
        plt.colorbar(im, ax=ax, label='MRL', shrink=0.8)

        # 有意性マーク
        for bi, band in enumerate(band_names):
            for ui_new, ui_old in enumerate(sort_idx):
                uk = unit_labels[ui_old]
                pr = self.sla.unit_results[uk]
                if band in pr.band_results:
                    for _, res in pr.band_results[band].items():
                        if res is not None and res.significant:
                            ax.text(ui_new, bi, '*', ha='center', va='center',
                                    fontsize=10, color='white', fontweight='bold')
                            break

        # --- 右: MRL深度プロファイル（帯域ごと） ---
        ax = axes[1]
        colors_band = plt.cm.Set1(np.linspace(0, 0.8, n_bands))

        for bi, band in enumerate(band_names):
            mrls = np.array(mrl_data[band])
            ax.scatter(mrls, unit_depths, s=40, color=colors_band[bi],
                       alpha=0.7, label=band, edgecolors='black', linewidths=0.5)

        self._draw_layer_background(ax, 'horizontal')
        ax.invert_yaxis()
        ax.set_xlabel('MRL')
        ax.set_ylabel('Depth (µm)')
        ax.set_title('Phase Locking Depth Profile')
        ax.legend(fontsize=8, loc='lower right')
        ax.axvline(0.1, color='gray', linestyle=':', alpha=0.5, label='MRL=0.1')

        fig.suptitle('Phase Locking × Depth', fontsize=13, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    def plot_sta_depth(self, window_ms=(-100, 100), save_path=None, figsize=(12, 10)):
        """
        STA（Spike-Triggered Average LFP）を深度ごとに並べて表示
        """
        if self.sla is None or not self.sla.unit_results:
            print("SpikeLFPAnalyzer required.")
            return None

        units_with_sta = [(k, v) for k, v in self.sla.unit_results.items()
                          if v.sta is not None]

        if not units_with_sta:
            print("No STA data available.")
            return None

        n_units = len(units_with_sta)
        fig, axes = plt.subplots(n_units, 1, figsize=figsize, sharex=True)
        if n_units == 1:
            axes = [axes]

        # 深度順にソート
        units_with_sta.sort(key=lambda x: x[1].channel)

        for ax, (unit_key, pl_result) in zip(axes, units_with_sta):
            sta = pl_result.sta
            sta_time = pl_result.sta_time
            depth = DEPTHS_UM[pl_result.channel] if pl_result.channel < N_ELECTRODES else 0

            if sta.ndim == 1:
                ax.plot(sta_time, sta, 'k-', linewidth=1.5)
            else:
                # 複数LFPチャンネル: 同一スパイクチャンネルのLFPを強調
                for i in range(sta.shape[1]):
                    alpha = 1.0 if i == min(pl_result.channel, sta.shape[1] - 1) else 0.2
                    lw = 2.0 if alpha == 1.0 else 0.5
                    ax.plot(sta_time, sta[:, i], linewidth=lw, alpha=alpha, color='black')

            ax.axvline(0, color='red', linewidth=1, linestyle='--')
            ax.set_ylabel(f'{unit_key}\n{depth}µm', fontsize=8)
            ax.tick_params(labelsize=7)

        axes[-1].set_xlabel('Time from spike (ms)')
        fig.suptitle('Spike-Triggered Average LFP by Depth', fontsize=13, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    # ================================================================
    #  4. グランドサマリー
    # ================================================================

    def plot_grand_summary(self, save_path=None, figsize=(24, 18), fig=None):
        """
        全解析のグランドサマリー（12パネル）

        Row 1: [波形一覧(4ch代表)] [品質散布図] [発火率深度]
        Row 2: [CSD ヒートマップ ] [パワー深度] [コヒーレンス]
        Row 3: [位相ロック深度   ] [STA代表   ] [条件別MRL  ]
        Row 4: [PSTH代表         ] [適応代表  ] [テキストサマリー]
        """
        if fig is None:
            fig = plt.figure(figsize=figsize)
        gs = GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

        # ========= Row 1: スパイクソーティング =========

        # 1-1: 代表波形（4チャンネル）
        ax = fig.add_subplot(gs[0, 0])
        sorted_chs = sorted(self.sorting.keys())
        # 深さの異なる4チャンネルを選択
        n_show = min(4, len(sorted_chs))
        step = max(1, len(sorted_chs) // n_show)
        representative_chs = sorted_chs[::step][:n_show]

        for i, ch in enumerate(representative_chs):
            result = self.sorting[ch]
            depth = DEPTHS_UM[ch] if ch < N_ELECTRODES else 0
            for unit in result.units:
                if unit.is_noise or len(unit.waveforms) == 0:
                    continue
                time_ms = result.waveform_time_ms if result.waveform_time_ms is not None \
                    else np.arange(unit.waveforms.shape[1]) / result.fs * 1000
                mean_wf = np.mean(unit.waveforms, axis=0)
                offset = -i * np.max(np.abs(mean_wf)) * 2.5
                ax.plot(time_ms, mean_wf + offset, color=unit.color, linewidth=1.5)
                ax.text(time_ms[-1] + 0.05, offset,
                        f'Ch{ch} U{unit.unit_id}\n{depth}µm', fontsize=6, va='center')
        ax.set_xlabel('ms', fontsize=8)
        ax.set_title('Representative Waveforms', fontsize=10)
        ax.set_yticks([])

        # 1-2: 品質散布図
        ax = fig.add_subplot(gs[0, 1])
        table = self.spike_sorting_table()
        valid = [r for r in table if not r['is_noise']]
        if valid:
            snrs = [r['snr'] for r in valid]
            isis = [r['isi_violation'] for r in valid]
            ds = [r['depth_um'] for r in valid]
            sc = ax.scatter(snrs, isis, c=ds, cmap='viridis', s=30,
                            edgecolors='black', linewidths=0.5, alpha=0.8)
            plt.colorbar(sc, ax=ax, label='Depth', shrink=0.7)
            ax.axhline(2, color='orange', linestyle='--', alpha=0.5)
            ax.axvline(3, color='blue', linestyle='--', alpha=0.5)
        ax.set_xlabel('SNR', fontsize=8)
        ax.set_ylabel('ISI viol.(%)', fontsize=8)
        ax.set_title('Unit Quality', fontsize=10)
        ax.tick_params(labelsize=7)

        # 1-3: 発火率深度
        ax = fig.add_subplot(gs[0, 2])
        if valid:
            for r in valid:
                marker = 's' if r['is_mua'] else 'o'
                c = 'gray' if r['is_mua'] else 'steelblue'
                ax.scatter(r['firing_rate_hz'], r['depth_um'], s=30,
                           marker=marker, color=c, edgecolors='black', linewidths=0.5)
            ax.invert_yaxis()
            self._draw_layer_background(ax, 'horizontal')
        ax.set_xlabel('FR (Hz)', fontsize=8)
        ax.set_ylabel('Depth (µm)', fontsize=8)
        ax.set_title('Firing Rate Profile', fontsize=10)
        ax.tick_params(labelsize=7)

        # ========= Row 2: LFP =========

        # 2-1: CSD
        ax = fig.add_subplot(gs[1, 0])
        try:
            avg_csd, csd_times, csd_depths = self.compute_trial_averaged_csd()
            vmax = np.percentile(np.abs(avg_csd), 95)
            im = ax.pcolormesh(csd_times, csd_depths, avg_csd.T,
                               cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
            ax.axvline(0, color='black', linewidth=1.5, linestyle='--')
            ax.invert_yaxis()
            ax.set_xlabel('ms from stim', fontsize=8)
            ax.set_ylabel('Depth (µm)', fontsize=8)
            plt.colorbar(im, ax=ax, shrink=0.7)
        except Exception as e:
            ax.text(0.5, 0.5, f'CSD: {e}', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8)
        ax.set_title('Trial-Averaged CSD', fontsize=10)
        ax.tick_params(labelsize=7)

        # 2-2: パワー深度
        ax = fig.add_subplot(gs[1, 1])
        freqs, psd = self.compute_power_by_depth()
        bands = {'θ': (4, 8), 'γ': (30, 80)}
        for bname, (lo, hi) in bands.items():
            bmask = (freqs >= lo) & (freqs < hi)
            bp = np.mean(psd[bmask, :], axis=0)
            bp_norm = bp / bp.max() if bp.max() > 0 else bp
            ax.plot(bp_norm, self.depths, 'o-', linewidth=1.5, markersize=3, label=bname)
        ax.invert_yaxis()
        ax.set_xlabel('Norm. power', fontsize=8)
        ax.set_ylabel('Depth (µm)', fontsize=8)
        ax.set_title('Band Power Depth', fontsize=10)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

        # 2-3: コヒーレンス
        ax = fig.add_subplot(gs[1, 2])
        coh = self._compute_coherence_matrix((4, 8))
        im = ax.imshow(coh, cmap='viridis', vmin=0, vmax=1, origin='upper')
        ax.set_title('Theta Coherence', fontsize=10)
        ax.set_xlabel('Channel', fontsize=8)
        ax.set_ylabel('Channel', fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.7)
        ax.tick_params(labelsize=6)

        # ========= Row 3: 統合解析 =========

        # 3-1: 位相ロック深度
        ax = fig.add_subplot(gs[2, 0])
        if self.sla and self.sla.unit_results:
            band_names = list(self.sla.freq_bands.keys())
            colors_pl = plt.cm.Set1(np.linspace(0, 0.8, len(band_names)))
            for bi, band in enumerate(band_names):
                for uk, pr in self.sla.unit_results.items():
                    depth = DEPTHS_UM[pr.channel] if pr.channel < N_ELECTRODES else 0
                    best_mrl = 0
                    if band in pr.band_results:
                        for _, res in pr.band_results[band].items():
                            if res is not None and res.mrl > best_mrl:
                                best_mrl = res.mrl
                    if best_mrl > 0:
                        ax.scatter(best_mrl, depth, s=30, color=colors_pl[bi],
                                   alpha=0.7, edgecolors='black', linewidths=0.3)
            ax.invert_yaxis()
            ax.legend(band_names, fontsize=5, loc='lower right')
        ax.set_xlabel('MRL', fontsize=8)
        ax.set_ylabel('Depth (µm)', fontsize=8)
        ax.set_title('Phase Locking Depth', fontsize=10)
        ax.tick_params(labelsize=7)

        # 3-2: STA代表（最初のユニット）
        ax = fig.add_subplot(gs[2, 1])
        if self.sla and self.sla.unit_results:
            first_key = list(self.sla.unit_results.keys())[0]
            pr = self.sla.unit_results[first_key]
            if pr.sta is not None:
                sta = pr.sta[:, 0] if pr.sta.ndim > 1 else pr.sta
                ax.plot(pr.sta_time, sta, 'k-', linewidth=1.5)
                ax.axvline(0, color='red', linestyle='--', linewidth=1)
                ax.set_title(f'STA: {first_key}', fontsize=10)
            else:
                ax.text(0.5, 0.5, 'No STA', ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel('ms from spike', fontsize=8)
        ax.tick_params(labelsize=7)

        # 3-3: 条件別MRL
        ax = fig.add_subplot(gs[2, 2])
        if self.sla and self.sla.unit_results:
            cond_mrl = {'baseline': [], 'stim': [], 'post': []}
            for pr in self.sla.unit_results.values():
                for cond, res in pr.condition_results.items():
                    if res is not None and cond in cond_mrl:
                        cond_mrl[cond].append(res.mrl)
            conds = ['baseline', 'stim', 'post']
            ccolors = ['gray', 'red', 'blue']
            means = [np.mean(cond_mrl[c]) if cond_mrl[c] else 0 for c in conds]
            sems = [np.std(cond_mrl[c]) / np.sqrt(len(cond_mrl[c]))
                    if len(cond_mrl[c]) > 1 else 0 for c in conds]
            ax.bar(conds, means, yerr=sems, color=ccolors, alpha=0.7,
                   edgecolor='black', capsize=3)
            ax.set_ylabel('Mean MRL', fontsize=8)
        ax.set_title('MRL by Condition', fontsize=10)
        ax.tick_params(labelsize=7)

        # ========= Row 4: 刺激応答 + テキスト =========

        # 4-1: PSTH 代表
        ax = fig.add_subplot(gs[3, 0])
        if self.protocol and self.sorting:
            first_ch = sorted(self.sorting.keys())[0]
            first_unit = None
            for u in self.sorting[first_ch].units:
                if not u.is_noise:
                    first_unit = u
                    break
            if first_unit:
                self.protocol.plot_psth(first_unit.spike_times, ax=ax)
                ax.set_title(f'PSTH: ch{first_ch}_U{first_unit.unit_id}', fontsize=10)
        ax.tick_params(labelsize=7)

        # 4-2: 適応 代表
        ax = fig.add_subplot(gs[3, 1])
        if self.protocol and first_unit:
            self.protocol.plot_adaptation(first_unit.spike_times, ax=ax)
        ax.tick_params(labelsize=7)

        # 4-3: テキストサマリー
        ax = fig.add_subplot(gs[3, 2])
        ax.axis('off')
        summary = self._generate_summary_text()
        ax.text(0.05, 0.95, summary, transform=ax.transAxes,
                fontsize=8, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        fig.suptitle(f'Grand Summary — {self.session.basename}',
                     fontsize=15, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    # ================================================================
    #  レポート生成
    # ================================================================

    def _generate_summary_text(self) -> str:
        """テキストサマリーを生成"""
        table = self.spike_sorting_table()
        n_su = sum(1 for r in table if r['quality'] not in ('Noise', 'MUA'))
        n_mua = sum(1 for r in table if r['quality'] == 'MUA')
        n_noise = sum(1 for r in table if r['quality'] == 'Noise')
        total_spikes = sum(r['n_spikes'] for r in table if not r['is_noise'])

        lines = [
            f"=== Analysis Summary ===",
            f"",
            f"Recording: {self.session.basename}",
            f"Duration:  {self.session.duration:.1f}s",
            f"Channels:  {len(self.sorting)} sorted",
            f"",
            f"Spike Sorting:",
            f"  Single Units: {n_su}",
            f"  Multi-Units:  {n_mua}",
            f"  Noise:        {n_noise}",
            f"  Total spikes: {total_spikes}",
        ]

        if self.protocol:
            lines += [
                f"",
                f"Stimulus:",
                f"  {self.protocol.n_trials} trials",
                f"  × {self.protocol.n_stim_per_trial} stim",
                f"  @ {self.protocol.stim_freq:.0f} Hz",
            ]

        if self.sla and self.sla.unit_results:
            n_sig = sum(
                1 for pr in self.sla.unit_results.values()
                for b, cr in pr.band_results.items()
                for c, res in cr.items()
                if res is not None and res.significant
            )
            lines += [
                f"",
                f"Phase Locking:",
                f"  Units analyzed: {len(self.sla.unit_results)}",
                f"  Significant:    {n_sig} pairs",
            ]

        return '\n'.join(lines)

    def generate_report(self, output_dir: str):
        """テキストレポートをファイルに保存"""
        os.makedirs(output_dir, exist_ok=True)

        report_path = os.path.join(output_dir, 'analysis_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(self._generate_summary_text())
            f.write('\n\n')

            # ユニットテーブル
            f.write('=== Unit Quality Table ===\n')
            f.write(f'{"Ch":>4} {"Depth":>6} {"Layer":>5} {"Unit":>5} '
                    f'{"nSpk":>6} {"SNR":>6} {"ISI%":>6} {"FR":>7} {"Quality":>12}\n')
            f.write('-' * 70 + '\n')
            for r in self.spike_sorting_table():
                f.write(f'{r["channel"]:>4} {r["depth_um"]:>6} {r["layer"]:>5} '
                        f'{r["unit_id"]:>5} {r["n_spikes"]:>6} {r["snr"]:>6.2f} '
                        f'{r["isi_violation"]:>6.2f} {r["firing_rate_hz"]:>7.2f} '
                        f'{r["quality"]:>12}\n')

        print(f"Report saved: {report_path}")
        return report_path

    def save_spike_table_csv(self, output_dir: str):
        """スパイクソーティングテーブルをCSV保存"""
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, 'unit_quality_table.csv')

        table = self.spike_sorting_table()
        if not table:
            return

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=table[0].keys())
            writer.writeheader()
            writer.writerows(table)

        print(f"CSV saved: {csv_path}")

    # ================================================================
    #  全保存
    # ================================================================

    def save_all(self, output_dir: str, verbose=True):
        """
        全プロット・テーブル・レポートを保存

        出力ファイル:
          output_dir/
          ├── spike_overview.png
          ├── firing_rate_condition.png
          ├── csd_summary.png
          ├── lfp_depth_analysis.png
          ├── lfp_overview.png
          ├── phase_locking_depth.png
          ├── sta_depth.png
          ├── grand_summary.png
          ├── unit_quality_table.csv
          └── analysis_report.txt
        """
        os.makedirs(output_dir, exist_ok=True)
        log = print if verbose else lambda *a, **kw: None

        log(f"\n=== Comprehensive Analysis: Saving to {output_dir} ===")

        # スパイクソーティング
        log("  [1/8] Spike overview...")
        self.plot_spike_overview(
            save_path=os.path.join(output_dir, 'spike_overview.png'))
        plt.close()

        log("  [2/8] Firing rate by condition...")
        fig = self.plot_firing_rate_by_condition(
            save_path=os.path.join(output_dir, 'firing_rate_condition.png'))
        if fig:
            plt.close()

        # LFP
        log("  [3/8] CSD summary...")
        try:
            self.plot_csd_summary(
                save_path=os.path.join(output_dir, 'csd_summary.png'))
            plt.close()
        except Exception as e:
            log(f"    Skipped CSD: {e}")

        log("  [4/8] LFP depth analysis...")
        self.plot_lfp_depth_analysis(
            save_path=os.path.join(output_dir, 'lfp_depth_analysis.png'))
        plt.close()

        log("  [5/8] LFP overview...")
        self.plot_lfp_overview(
            save_path=os.path.join(output_dir, 'lfp_overview.png'))
        plt.close()

        # 統合解析
        log("  [6/8] Phase locking depth...")
        fig = self.plot_phase_locking_depth(
            save_path=os.path.join(output_dir, 'phase_locking_depth.png'))
        if fig:
            plt.close()

        log("  [7/8] STA depth...")
        fig = self.plot_sta_depth(
            save_path=os.path.join(output_dir, 'sta_depth.png'))
        if fig:
            plt.close()

        # グランドサマリー
        log("  [8/8] Grand summary...")
        self.plot_grand_summary(
            save_path=os.path.join(output_dir, 'grand_summary.png'))
        plt.close()

        # CSV & レポート
        self.save_spike_table_csv(output_dir)
        self.generate_report(output_dir)

        log(f"\n=== Complete: all outputs in {output_dir} ===")
