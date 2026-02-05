"""
waveform_browser.py - 個々のスパイク波形を確認するブラウザ (v2)

修正点:
- 親ウィンドウを正しく受け取る
- Toplevelを適切に使用
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from typing import Dict, List, Optional
import random

from spike_sorting import ChannelSortResult, SpikeUnit


class WaveformBrowserWindow:
    """個々のスパイク波形を確認するブラウザウィンドウ"""
    
    def __init__(self, parent, result: ChannelSortResult, unit: SpikeUnit = None):
        """
        Parameters
        ----------
        parent : tk.Tk or tk.Toplevel
            親ウィンドウ
        result : ChannelSortResult
            チャンネルのソーティング結果
        unit : SpikeUnit, optional
            表示するユニット（Noneで全スパイク）
        """
        self.result = result
        self.current_unit = unit
        
        # 表示するスパイクデータ
        if unit is not None:
            self.waveforms = unit.waveforms
            self.spike_indices = unit.spike_indices
            self.spike_times = unit.spike_times
            self.title = f"Unit {unit.unit_id}"
        else:
            self.waveforms = result.all_waveforms
            self.spike_indices = result.all_spike_indices
            self.spike_times = result.all_spike_indices / result.fs
            self.title = "All Spikes"
        
        self.n_spikes = len(self.waveforms) if self.waveforms is not None else 0
        self.current_idx = 0
        
        # マークされたスパイク
        self.marked_good: set = set()
        self.marked_bad: set = set()
        
        # 時間軸
        if result.waveform_time_ms is not None:
            self.time_ms = result.waveform_time_ms
        else:
            n_samples = self.waveforms.shape[1] if self.waveforms is not None and len(self.waveforms) > 0 else 60
            self.time_ms = np.linspace(-0.5, 1.0, n_samples)
        
        # ウィンドウ作成
        self.window = tk.Toplevel(parent)
        self.window.title(f"Waveform Browser - {self.title}")
        self.window.geometry("1000x700")
        self.window.transient(parent)
        
        self._build_gui()
        
        if self.n_spikes > 0:
            self._update_display()
    
    def _build_gui(self):
        """GUIを構築"""
        # === 上部: コントロール ===
        control_frame = ttk.Frame(self.window)
        control_frame.pack(fill='x', padx=5, pady=5)
        
        # ナビゲーション
        ttk.Button(control_frame, text="<<-10", command=lambda: self._move(-10), width=6).pack(side='left', padx=2)
        ttk.Button(control_frame, text="<Prev", command=lambda: self._move(-1), width=6).pack(side='left', padx=2)
        
        self.idx_var = tk.StringVar(value="1")
        self.idx_entry = ttk.Entry(control_frame, textvariable=self.idx_var, width=8)
        self.idx_entry.pack(side='left', padx=5)
        self.idx_entry.bind('<Return>', self._on_idx_enter)
        
        self.total_label = ttk.Label(control_frame, text=f"/ {self.n_spikes}")
        self.total_label.pack(side='left')
        
        ttk.Button(control_frame, text="Next>", command=lambda: self._move(1), width=6).pack(side='left', padx=2)
        ttk.Button(control_frame, text="+10>>", command=lambda: self._move(10), width=6).pack(side='left', padx=2)
        
        ttk.Separator(control_frame, orient='vertical').pack(side='left', fill='y', padx=10)
        
        ttk.Button(control_frame, text="Random", command=self._random_spike, width=8).pack(side='left', padx=5)
        
        ttk.Separator(control_frame, orient='vertical').pack(side='left', fill='y', padx=10)
        
        # マークボタン
        ttk.Button(control_frame, text="Good (G)", command=self._mark_good, width=8).pack(side='left', padx=2)
        ttk.Button(control_frame, text="Bad (B)", command=self._mark_bad, width=8).pack(side='left', padx=2)
        ttk.Button(control_frame, text="Clear", command=self._clear_mark, width=6).pack(side='left', padx=2)
        
        # 統計表示
        self.stats_label = ttk.Label(control_frame, text="Good: 0, Bad: 0")
        self.stats_label.pack(side='right', padx=10)
        
        # === 中央: プロット ===
        plot_frame = ttk.Frame(self.window)
        plot_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # === 下部: 情報 ===
        info_frame = ttk.Frame(self.window)
        info_frame.pack(fill='x', padx=5, pady=5)
        
        self.info_var = tk.StringVar(value="")
        ttk.Label(info_frame, textvariable=self.info_var).pack(side='left')
        
        # キーバインド
        self.window.bind('<Left>', lambda e: self._move(-1))
        self.window.bind('<Right>', lambda e: self._move(1))
        self.window.bind('<Up>', lambda e: self._move(-10))
        self.window.bind('<Down>', lambda e: self._move(10))
        self.window.bind('g', lambda e: self._mark_good())
        self.window.bind('G', lambda e: self._mark_good())
        self.window.bind('b', lambda e: self._mark_bad())
        self.window.bind('B', lambda e: self._mark_bad())
        self.window.bind('r', lambda e: self._random_spike())
        self.window.bind('R', lambda e: self._random_spike())
        self.window.bind('<space>', lambda e: self._move(1))
        
        # フォーカス設定
        self.window.focus_set()
    
    def _move(self, delta: int):
        """スパイク位置を移動"""
        if self.n_spikes == 0:
            return
        self.current_idx = (self.current_idx + delta) % self.n_spikes
        self._update_display()
    
    def _on_idx_enter(self, event=None):
        """インデックス直接入力"""
        try:
            idx = int(self.idx_var.get()) - 1
            if 0 <= idx < self.n_spikes:
                self.current_idx = idx
                self._update_display()
        except ValueError:
            pass
    
    def _random_spike(self):
        """ランダムなスパイクを表示"""
        if self.n_spikes > 0:
            self.current_idx = random.randint(0, self.n_spikes - 1)
            self._update_display()
    
    def _mark_good(self):
        """現在のスパイクを良いとマーク"""
        self.marked_good.add(self.current_idx)
        self.marked_bad.discard(self.current_idx)
        self._update_stats()
        self._move(1)
    
    def _mark_bad(self):
        """現在のスパイクを悪いとマーク"""
        self.marked_bad.add(self.current_idx)
        self.marked_good.discard(self.current_idx)
        self._update_stats()
        self._move(1)
    
    def _clear_mark(self):
        """現在のスパイクのマークをクリア"""
        self.marked_good.discard(self.current_idx)
        self.marked_bad.discard(self.current_idx)
        self._update_stats()
    
    def _update_stats(self):
        """統計表示を更新"""
        self.stats_label.config(text=f"Good: {len(self.marked_good)}, Bad: {len(self.marked_bad)}")
    
    def _update_display(self):
        """表示を更新"""
        self.fig.clear()
        
        if self.n_spikes == 0:
            self.canvas.draw()
            return
        
        # インデックス表示更新
        self.idx_var.set(str(self.current_idx + 1))
        
        # 現在の波形
        current_wf = self.waveforms[self.current_idx]
        current_time = self.spike_times[self.current_idx]
        current_idx_sample = self.spike_indices[self.current_idx]
        
        # マーク状態
        if self.current_idx in self.marked_good:
            mark_status = "GOOD"
            mark_color = 'green'
        elif self.current_idx in self.marked_bad:
            mark_status = "BAD"
            mark_color = 'red'
        else:
            mark_status = ""
            mark_color = 'black'
        
        # サブプロット
        gs = self.fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. 現在の波形（大きく）
        ax_main = self.fig.add_subplot(gs[0, :2])
        ax_main.plot(self.time_ms, current_wf, 'b-', linewidth=2, label='Current')
        ax_main.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax_main.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax_main.axhline(y=self.result.threshold, color='red', linestyle='--', alpha=0.5, label='Threshold')
        ax_main.set_xlabel('Time (ms)')
        ax_main.set_ylabel('Amplitude')
        
        title_str = f'Spike {self.current_idx + 1} / {self.n_spikes}'
        if mark_status:
            title_str += f'  [{mark_status}]'
        ax_main.set_title(title_str, color=mark_color, fontweight='bold')
        ax_main.legend(loc='upper right')
        
        # ピーク振幅を表示
        peak_amp = np.min(current_wf)
        peak_idx = np.argmin(current_wf)
        ax_main.scatter([self.time_ms[peak_idx]], [peak_amp], c='red', s=50, zorder=5)
        ax_main.annotate(f'{peak_amp:.4f}', (self.time_ms[peak_idx], peak_amp),
                        textcoords='offset points', xytext=(10, -10), fontsize=9)
        
        # 2. 前後10スパイクとの比較
        ax_context = self.fig.add_subplot(gs[0, 2])
        start_idx = max(0, self.current_idx - 5)
        end_idx = min(self.n_spikes, self.current_idx + 6)
        
        for i in range(start_idx, end_idx):
            if i == self.current_idx:
                ax_context.plot(self.time_ms, self.waveforms[i], 'b-', linewidth=2, alpha=0.9)
            else:
                ax_context.plot(self.time_ms, self.waveforms[i], 'gray', linewidth=0.5, alpha=0.3)
        
        ax_context.set_xlabel('Time (ms)')
        ax_context.set_title('Context (+/- 5 spikes)')
        
        # 3. 全スパイクの中での位置（振幅分布）
        ax_amp = self.fig.add_subplot(gs[1, 0])
        all_peaks = np.min(self.waveforms, axis=1)
        ax_amp.hist(all_peaks, bins=50, color='gray', alpha=0.7, edgecolor='black')
        ax_amp.axvline(x=peak_amp, color='blue', linewidth=2, label='Current')
        ax_amp.axvline(x=self.result.threshold, color='red', linestyle='--', label='Threshold')
        ax_amp.set_xlabel('Peak Amplitude')
        ax_amp.set_ylabel('Count')
        ax_amp.set_title('Amplitude Distribution')
        ax_amp.legend(fontsize=8)
        
        # 4. 時系列でのスパイク位置
        ax_time = self.fig.add_subplot(gs[1, 1])
        
        window_sec = 1.0
        mask = np.abs(self.spike_times - current_time) < window_sec
        nearby_times = self.spike_times[mask]
        nearby_amps = np.min(self.waveforms[mask], axis=1)
        
        ax_time.scatter(nearby_times - current_time, nearby_amps, c='gray', s=20, alpha=0.5)
        ax_time.scatter([0], [peak_amp], c='blue', s=100, zorder=5, label='Current')
        ax_time.axhline(y=self.result.threshold, color='red', linestyle='--', alpha=0.5)
        ax_time.set_xlabel('Time from current (s)')
        ax_time.set_ylabel('Amplitude')
        ax_time.set_title(f'Nearby spikes (+/- {window_sec}s)')
        ax_time.set_xlim(-window_sec, window_sec)
        
        # 5. 情報パネル
        ax_info = self.fig.add_subplot(gs[1, 2])
        ax_info.axis('off')
        
        info_text = f"=== Spike Info ===\n\n"
        info_text += f"Index: {self.current_idx + 1} / {self.n_spikes}\n"
        info_text += f"Time: {current_time:.4f} s\n"
        info_text += f"Sample: {current_idx_sample}\n"
        info_text += f"Peak Amp: {peak_amp:.4f}\n"
        info_text += f"Threshold: {self.result.threshold:.4f}\n"
        info_text += f"\n"
        info_text += f"=== Keyboard ===\n"
        info_text += f"Left/Right : +/- 1\n"
        info_text += f"Up/Down    : +/- 10\n"
        info_text += f"Space      : Next\n"
        info_text += f"G          : Mark Good\n"
        info_text += f"B          : Mark Bad\n"
        info_text += f"R          : Random\n"
        
        ax_info.text(0.1, 0.95, info_text, transform=ax_info.transAxes,
                    fontsize=9, verticalalignment='top', fontfamily='monospace')
        
        self.fig.suptitle(f'{self.title} - Waveform Browser', fontsize=12)
        self.canvas.draw()
        
        # 情報バー更新
        self.info_var.set(f"Time: {current_time:.4f}s | Amplitude: {peak_amp:.4f}")


class MultiWaveformWindow:
    """複数スパイクを一度に表示するビューア"""
    
    def __init__(self, parent, result: ChannelSortResult, unit: SpikeUnit = None,
                 n_display: int = 25):
        self.result = result
        self.unit = unit
        self.n_display = n_display
        
        if unit is not None:
            self.waveforms = unit.waveforms
            self.spike_indices = unit.spike_indices
            self.title = f"Unit {unit.unit_id}"
        else:
            self.waveforms = result.all_waveforms
            self.spike_indices = result.all_spike_indices
            self.title = "All Spikes"
        
        self.n_spikes = len(self.waveforms) if self.waveforms is not None else 0
        self.current_page = 0
        self.n_pages = max(1, (self.n_spikes + n_display - 1) // n_display)
        
        # 時間軸
        if result.waveform_time_ms is not None:
            self.time_ms = result.waveform_time_ms
        else:
            n_samples = self.waveforms.shape[1] if self.waveforms is not None and len(self.waveforms) > 0 else 60
            self.time_ms = np.linspace(-0.5, 1.0, n_samples)
        
        # ウィンドウ
        self.window = tk.Toplevel(parent)
        self.window.title(f"Multi-Waveform Viewer - {self.title}")
        self.window.geometry("1200x800")
        self.window.transient(parent)
        
        self._build_gui()
        
        if self.n_spikes > 0:
            self._update_display()
    
    def _build_gui(self):
        """GUIを構築"""
        # コントロール
        control_frame = ttk.Frame(self.window)
        control_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(control_frame, text="<< First", command=self._first_page, width=8).pack(side='left', padx=2)
        ttk.Button(control_frame, text="< Prev", command=self._prev_page, width=8).pack(side='left', padx=2)
        
        self.page_label = ttk.Label(control_frame, text=f"Page 1 / {self.n_pages}")
        self.page_label.pack(side='left', padx=10)
        
        ttk.Button(control_frame, text="Next >", command=self._next_page, width=8).pack(side='left', padx=2)
        ttk.Button(control_frame, text="Last >>", command=self._last_page, width=8).pack(side='left', padx=2)
        
        ttk.Separator(control_frame, orient='vertical').pack(side='left', fill='y', padx=10)
        
        ttk.Button(control_frame, text="Random", command=self._random_page, width=8).pack(side='left', padx=5)
        
        ttk.Label(control_frame, text=f"Total: {self.n_spikes} spikes").pack(side='right', padx=10)
        
        # プロット
        plot_frame = ttk.Frame(self.window)
        plot_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # キーバインド
        self.window.bind('<Left>', lambda e: self._prev_page())
        self.window.bind('<Right>', lambda e: self._next_page())
        self.window.bind('<Home>', lambda e: self._first_page())
        self.window.bind('<End>', lambda e: self._last_page())
        
        self.window.focus_set()
    
    def _first_page(self):
        self.current_page = 0
        self._update_display()
    
    def _last_page(self):
        self.current_page = self.n_pages - 1
        self._update_display()
    
    def _prev_page(self):
        self.current_page = max(0, self.current_page - 1)
        self._update_display()
    
    def _next_page(self):
        self.current_page = min(self.n_pages - 1, self.current_page + 1)
        self._update_display()
    
    def _random_page(self):
        if self.n_pages > 0:
            self.current_page = random.randint(0, self.n_pages - 1)
            self._update_display()
    
    def _update_display(self):
        """表示を更新"""
        self.fig.clear()
        
        if self.n_spikes == 0:
            self.canvas.draw()
            return
        
        # ページラベル更新
        self.page_label.config(text=f"Page {self.current_page + 1} / {self.n_pages}")
        
        # 表示するスパイクの範囲
        start_idx = self.current_page * self.n_display
        end_idx = min(start_idx + self.n_display, self.n_spikes)
        
        # グリッド計算
        n_cols = 5
        n_rows = (self.n_display + n_cols - 1) // n_cols
        
        # 平均波形（参照用）
        mean_wf = np.mean(self.waveforms, axis=0)
        
        # Y軸の範囲を統一
        y_min = self.result.threshold * 1.5
        y_max = -self.result.threshold * 0.5
        
        for i, idx in enumerate(range(start_idx, end_idx)):
            ax = self.fig.add_subplot(n_rows, n_cols, i + 1)
            
            # 平均波形（薄いグレー）
            ax.plot(self.time_ms, mean_wf, 'gray', linewidth=1, alpha=0.5)
            
            # 現在の波形
            ax.plot(self.time_ms, self.waveforms[idx], 'b-', linewidth=1)
            
            # 閾値
            ax.axhline(y=self.result.threshold, color='red', linestyle='--', 
                      alpha=0.3, linewidth=0.5)
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
            
            # ラベル
            peak_amp = np.min(self.waveforms[idx])
            ax.set_title(f'#{idx+1} ({peak_amp:.3f})', fontsize=8)
            
            # 軸を簡略化
            ax.set_xticks([])
            ax.set_yticks([])
            
            # 統一スケール
            ax.set_ylim(y_min, y_max)
        
        self.fig.suptitle(f'{self.title} - Page {self.current_page + 1}/{self.n_pages} '
                         f'(Spikes {start_idx+1}-{end_idx})', fontsize=12)
        
        self.fig.tight_layout(rect=[0, 0, 1, 0.96])
        self.canvas.draw()
