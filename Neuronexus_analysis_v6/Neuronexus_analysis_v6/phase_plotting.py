"""
phase_plotting.py - 位相ロック解析の可視化

極座標ヒストグラム、条件比較、時系列プロットなど
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from typing import Dict, List, Optional, Tuple, Any
import os

# 日本語フォント設定（環境に応じて調整）
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']


def plot_phase_histogram(
    spike_phases: np.ndarray,
    mrl: float = None,
    preferred_phase: float = None,
    title: str = "Phase Distribution",
    n_bins: int = 36,
    color: str = 'steelblue',
    ax: plt.Axes = None,
    show_stats: bool = True
) -> Tuple[plt.Figure, plt.Axes]:
    """
    位相分布の極座標ヒストグラム
    
    Parameters
    ----------
    spike_phases : np.ndarray
        スパイク位相（ラジアン）
    mrl : float or None
        Mean Resultant Length（Noneの場合は計算）
    preferred_phase : float or None
        平均位相（Noneの場合は計算）
    title : str
    n_bins : int
    color : str
    ax : matplotlib.axes.Axes or None
    show_stats : bool
        統計値を表示するか
    
    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='polar')
    else:
        fig = ax.figure
    
    # MRLと平均位相を計算（指定がない場合）
    if mrl is None or preferred_phase is None:
        mean_vector = np.mean(np.exp(1j * spike_phases))
        if mrl is None:
            mrl = np.abs(mean_vector)
        if preferred_phase is None:
            preferred_phase = np.angle(mean_vector)
    
    # ヒストグラム
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    counts, _ = np.histogram(spike_phases, bins=bins)
    
    # 正規化（確率密度）
    bin_width = 2 * np.pi / n_bins
    counts_norm = counts / (len(spike_phases) * bin_width)
    
    # バーの位置（ビンの中央）
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # 極座標バープロット
    bars = ax.bar(bin_centers, counts_norm, width=bin_width * 0.9, 
                  color=color, alpha=0.7, edgecolor='white', linewidth=0.5)
    
    # 平均ベクトルを矢印で描画
    max_count = np.max(counts_norm) if len(counts_norm) > 0 else 1
    arrow_length = mrl * max_count * 1.2  # MRLに比例した長さ
    
    ax.annotate('', xy=(preferred_phase, arrow_length), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
    
    # 設定
    ax.set_theta_zero_location('E')  # 0度を右に
    ax.set_theta_direction(1)  # 反時計回り
    ax.set_rlabel_position(45)
    
    # タイトルと統計情報
    if show_stats:
        title_text = f"{title}\nMRL={mrl:.3f}, θ={np.degrees(preferred_phase):.1f}°, n={len(spike_phases)}"
    else:
        title_text = title
    ax.set_title(title_text, fontsize=11, pad=15)
    
    return fig, ax


def plot_phase_locking_summary(
    results: Dict[str, Dict[int, Any]],
    band_names: List[str],
    channel_labels: List[str],
    output_dir: str,
    basename: str,
    show: bool = True,
    save: bool = True
) -> plt.Figure:
    """
    位相ロック結果のサマリープロット（ヒートマップ）
    
    Parameters
    ----------
    results : dict
        analyze_spike_lfp_couplingの出力
    band_names : list
        周波数帯域名のリスト
    channel_labels : list
        チャンネルラベル
    output_dir : str
    basename : str
    show, save : bool
    
    Returns
    -------
    fig
    """
    n_bands = len(band_names)
    n_channels = len(channel_labels)
    
    # MRLマトリクスを作成
    mrl_matrix = np.zeros((n_bands, n_channels))
    pvalue_matrix = np.ones((n_bands, n_channels))
    
    for i, band in enumerate(band_names):
        if band in results:
            for j in range(n_channels):
                if j in results[band] and results[band][j] is not None:
                    mrl_matrix[i, j] = results[band][j].mrl
                    pvalue_matrix[i, j] = results[band][j].p_value
    
    # プロット
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MRLヒートマップ
    im1 = axes[0].imshow(mrl_matrix, aspect='auto', cmap='YlOrRd', 
                         vmin=0, vmax=0.5)
    axes[0].set_xticks(range(n_channels))
    axes[0].set_xticklabels(channel_labels, rotation=45, ha='right')
    axes[0].set_yticks(range(n_bands))
    axes[0].set_yticklabels(band_names)
    axes[0].set_xlabel('Channel')
    axes[0].set_ylabel('Frequency Band')
    axes[0].set_title('Mean Resultant Length (MRL)')
    plt.colorbar(im1, ax=axes[0], label='MRL')
    
    # 有意性マーカー
    for i in range(n_bands):
        for j in range(n_channels):
            if pvalue_matrix[i, j] < 0.05:
                axes[0].text(j, i, '*', ha='center', va='center', 
                            fontsize=16, color='white', fontweight='bold')
            if pvalue_matrix[i, j] < 0.01:
                axes[0].text(j, i, '**', ha='center', va='center', 
                            fontsize=14, color='white', fontweight='bold')
    
    # p値ヒートマップ（-log10変換）
    log_pvalue = -np.log10(pvalue_matrix + 1e-10)
    im2 = axes[1].imshow(log_pvalue, aspect='auto', cmap='Blues', 
                         vmin=0, vmax=5)
    axes[1].set_xticks(range(n_channels))
    axes[1].set_xticklabels(channel_labels, rotation=45, ha='right')
    axes[1].set_yticks(range(n_bands))
    axes[1].set_yticklabels(band_names)
    axes[1].set_xlabel('Channel')
    axes[1].set_ylabel('Frequency Band')
    axes[1].set_title('Statistical Significance (-log₁₀ p-value)')
    cbar = plt.colorbar(im2, ax=axes[1], label='-log₁₀(p)')
    
    # 有意性閾値ライン
    axes[1].axhline(y=-0.5, color='gray', linestyle='--', alpha=0.3)
    
    plt.suptitle(f'{basename} - Phase Locking Summary', fontsize=12)
    plt.tight_layout()
    
    if save:
        filepath = os.path.join(output_dir, f'{basename}_phase_locking_summary.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"  保存: {filepath}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_condition_comparison(
    condition_results: Dict[str, Any],
    freq_band_name: str,
    unit_key: str,
    output_dir: str,
    basename: str,
    show: bool = True,
    save: bool = True
) -> plt.Figure:
    """
    条件間（baseline/stim/post）の位相ロック比較
    
    Parameters
    ----------
    condition_results : dict
        analyze_phase_locking_by_conditionの出力
    freq_band_name : str
        周波数帯域名
    unit_key : str
        ユニット識別子
    output_dir : str
    basename : str
    show, save : bool
    
    Returns
    -------
    fig
    """
    conditions = ['baseline', 'stim', 'post']
    colors = {'baseline': 'gray', 'stim': 'red', 'post': 'blue'}
    
    # 有効な条件数をカウント
    valid_conditions = [c for c in conditions if c in condition_results and condition_results[c] is not None]
    
    if len(valid_conditions) == 0:
        print("  警告: 有効な条件がありません")
        return None
    
    fig = plt.figure(figsize=(15, 5))
    
    # 上段: 各条件の極座標ヒストグラム
    for i, condition in enumerate(conditions):
        ax = fig.add_subplot(1, 4, i + 1, projection='polar')
        
        if condition in condition_results and condition_results[condition] is not None:
            result = condition_results[condition]
            plot_phase_histogram(
                result.spike_phases,
                mrl=result.mrl,
                preferred_phase=result.preferred_phase,
                title=f"{condition.capitalize()}\n(n={result.n_spikes})",
                color=colors[condition],
                ax=ax
            )
        else:
            ax.set_title(f"{condition.capitalize()}\n(insufficient data)")
    
    # 右端: MRL/PPCの比較バーグラフ
    ax_bar = fig.add_subplot(1, 4, 4)
    
    x = np.arange(len(conditions))
    width = 0.35
    
    mrl_values = []
    ppc_values = []
    for condition in conditions:
        if condition in condition_results and condition_results[condition] is not None:
            mrl_values.append(condition_results[condition].mrl)
            ppc_values.append(condition_results[condition].ppc)
        else:
            mrl_values.append(0)
            ppc_values.append(0)
    
    bars1 = ax_bar.bar(x - width/2, mrl_values, width, label='MRL', color='steelblue', alpha=0.8)
    bars2 = ax_bar.bar(x + width/2, ppc_values, width, label='PPC', color='coral', alpha=0.8)
    
    # 有意性マーカー
    for i, condition in enumerate(conditions):
        if condition in condition_results and condition_results[condition] is not None:
            if condition_results[condition].significant:
                ax_bar.text(i - width/2, mrl_values[i] + 0.02, '*', 
                           ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax_bar.set_ylabel('Value')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([c.capitalize() for c in conditions])
    ax_bar.legend(loc='upper right')
    ax_bar.set_ylim(0, max(max(mrl_values), max(ppc_values)) * 1.3 + 0.05)
    ax_bar.set_title('Comparison')
    
    plt.suptitle(f'{basename} - {unit_key} - {freq_band_name} band', fontsize=12)
    plt.tight_layout()
    
    if save:
        filepath = os.path.join(output_dir, f'{basename}_{unit_key}_{freq_band_name}_condition_comparison.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"  保存: {filepath}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_spike_lfp_relationship(
    lfp_data: np.ndarray,
    lfp_times: np.ndarray,
    spike_times: np.ndarray,
    lfp_phase: np.ndarray,
    t_window: Tuple[float, float] = (0, 5),
    fs: int = 1000,
    output_dir: str = None,
    basename: str = None,
    show: bool = True,
    save: bool = True
) -> plt.Figure:
    """
    スパイク-LFP関係の時系列プロット
    
    Parameters
    ----------
    lfp_data : np.ndarray
        生またはフィルタ済みLFP
    lfp_times : np.ndarray
    spike_times : np.ndarray
    lfp_phase : np.ndarray
    t_window : tuple
        表示する時間窓
    fs : int
    output_dir, basename : str
    show, save : bool
    
    Returns
    -------
    fig
    """
    t_start, t_end = t_window
    
    # 時間範囲のマスク
    time_mask = (lfp_times >= t_start) & (lfp_times <= t_end)
    times = lfp_times[time_mask]
    lfp = lfp_data[time_mask] if lfp_data.ndim == 1 else lfp_data[time_mask, 0]
    phase = lfp_phase[time_mask] if lfp_phase.ndim == 1 else lfp_phase[time_mask, 0]
    
    # 時間範囲内のスパイク
    spike_mask = (spike_times >= t_start) & (spike_times <= t_end)
    spikes = spike_times[spike_mask]
    
    # スパイク時点の位相を取得
    spike_indices = np.searchsorted(times, spikes)
    spike_indices = np.clip(spike_indices, 0, len(phase) - 1)
    spike_phases = phase[spike_indices]
    
    # プロット
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    
    # 上段: LFP
    axes[0].plot(times, lfp, 'k-', linewidth=0.5)
    axes[0].set_ylabel('LFP (μV)')
    axes[0].set_title('Local Field Potential')
    
    # スパイク時刻に縦線
    for spike_t in spikes:
        axes[0].axvline(spike_t, color='red', alpha=0.3, linewidth=0.5)
    
    # 中段: 瞬時位相
    axes[1].plot(times, phase, 'b-', linewidth=0.5)
    axes[1].set_ylabel('Phase (rad)')
    axes[1].set_ylim(-np.pi, np.pi)
    axes[1].set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    axes[1].set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    axes[1].set_title('Instantaneous Phase')
    
    # スパイク時点をマーカー
    axes[1].scatter(spikes, spike_phases, c='red', s=20, zorder=5)
    
    # 下段: スパイクラスタープロット（位相で色分け）
    if len(spikes) > 0:
        # 位相を色にマップ
        norm_phases = (spike_phases + np.pi) / (2 * np.pi)  # 0-1に正規化
        colors = plt.cm.hsv(norm_phases)
        
        axes[2].scatter(spikes, np.ones_like(spikes), c=colors, s=50, marker='|')
        axes[2].set_ylim(0.5, 1.5)
        axes[2].set_yticks([])
        axes[2].set_ylabel('Spikes')
        axes[2].set_title('Spike Raster (color = phase)')
        
        # カラーバー
        sm = plt.cm.ScalarMappable(cmap='hsv', norm=plt.Normalize(-np.pi, np.pi))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=axes[2], orientation='vertical', pad=0.02)
        cbar.set_label('Phase (rad)')
        cbar.set_ticks([-np.pi, 0, np.pi])
        cbar.set_ticklabels(['-π', '0', 'π'])
    
    axes[2].set_xlabel('Time (s)')
    
    plt.tight_layout()
    
    if save and output_dir and basename:
        filepath = os.path.join(output_dir, f'{basename}_spike_lfp_relationship.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"  保存: {filepath}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_preferred_phase_by_channel(
    results: Dict[str, Dict[int, Any]],
    band_name: str,
    channel_labels: List[str],
    output_dir: str,
    basename: str,
    show: bool = True,
    save: bool = True
) -> plt.Figure:
    """
    チャンネル別の平均位相（preferred phase）をプロット
    
    Parameters
    ----------
    results : dict
    band_name : str
    channel_labels : list
    output_dir, basename : str
    show, save : bool
    
    Returns
    -------
    fig
    """
    if band_name not in results:
        print(f"  警告: {band_name}のデータがありません")
        return None
    
    band_results = results[band_name]
    n_channels = len(channel_labels)
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')
    
    colors = plt.cm.viridis(np.linspace(0, 1, n_channels))
    
    for ch in range(n_channels):
        if ch in band_results and band_results[ch] is not None:
            result = band_results[ch]
            
            # 有意なもののみ描画
            if result.significant:
                # ベクトルを描画
                ax.annotate('', 
                           xy=(result.preferred_phase, result.mrl),
                           xytext=(0, 0),
                           arrowprops=dict(arrowstyle='->', color=colors[ch], lw=2))
                
                # チャンネルラベル
                ax.text(result.preferred_phase, result.mrl * 1.1, 
                       channel_labels[ch], ha='center', va='center',
                       fontsize=8, color=colors[ch])
    
    ax.set_ylim(0, 0.5)
    ax.set_title(f'{basename} - {band_name} band\nPreferred Phase by Channel (significant only)')
    
    if save:
        filepath = os.path.join(output_dir, f'{basename}_{band_name}_preferred_phase_channels.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"  保存: {filepath}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def save_phase_locking_csv(
    all_results: Dict[str, Dict[str, Dict]],
    unit_info_list: List,
    condition_results: Dict[str, Dict[str, Any]],
    output_dir: str,
    basename: str
) -> str:
    """
    位相ロック結果をCSVで保存
    
    Parameters
    ----------
    all_results : dict
        {unit_key: {band_name: {channel: PhaseLockingResult}}}
    unit_info_list : list
        UnitInfoのリスト
    condition_results : dict
        {unit_key: {condition: PhaseLockingResult}}
    output_dir : str
    basename : str
    
    Returns
    -------
    filepath : str
    """
    import csv
    
    filepath = os.path.join(output_dir, f'{basename}_phase_locking.csv')
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # ヘッダー
        writer.writerow([
            'unit_key', 'channel', 'unit_id', 'band', 'condition',
            'lfp_channel', 'n_spikes', 'mrl', 'ppc', 'p_value',
            'preferred_phase_deg', 'significant'
        ])
        
        # 全体解析結果
        for unit_key, band_results in all_results.items():
            # unit_info から情報を取得
            unit_info = next((u for u in unit_info_list if u.unit_key == unit_key), None)
            channel = unit_info.channel if unit_info else -1
            unit_id = unit_info.unit_id if unit_info else -1
            
            for band_name, channel_results in band_results.items():
                for lfp_ch, result in channel_results.items():
                    if result is not None:
                        writer.writerow([
                            unit_key, channel, unit_id, band_name, 'all',
                            lfp_ch, result.n_spikes, f'{result.mrl:.4f}',
                            f'{result.ppc:.4f}', f'{result.p_value:.6f}',
                            f'{result.preferred_phase_deg:.1f}', result.significant
                        ])
        
        # 条件別結果
        for unit_key, cond_results in condition_results.items():
            unit_info = next((u for u in unit_info_list if u.unit_key == unit_key), None)
            channel = unit_info.channel if unit_info else -1
            unit_id = unit_info.unit_id if unit_info else -1
            
            for condition, result in cond_results.items():
                if result is not None:
                    writer.writerow([
                        unit_key, channel, unit_id, 'theta', condition,
                        0, result.n_spikes, f'{result.mrl:.4f}',
                        f'{result.ppc:.4f}', f'{result.p_value:.6f}',
                        f'{result.preferred_phase_deg:.1f}', result.significant
                    ])
    
    print(f"  CSV保存: {filepath}")
    return filepath
