"""
spike_sorting_gui.py - スパイクソーティング GUI

機能:
- 波形・PCA・ISI表示
- ユニットの選択・編集
- Merge / Delete / Mark as MUA
- 再クラスタリング
- 結果保存
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import os
from typing import Dict, List, Optional

# 自作モジュール
from spike_sorting import (
    SortingConfig, ChannelSortResult, SpikeUnit,
    sort_all_channels, merge_units, delete_unit, undelete_unit,
    mark_as_mua, unmark_mua, recluster, save_sorting_results,
    export_spike_times_csv, compute_isi_histogram, compute_autocorrelogram
)


class SpikeSortingGUI:
    """スパイクソーティングGUI"""
    
    def __init__(self, results: Dict[int, ChannelSortResult] = None,
                 wideband_data: np.ndarray = None, fs: float = None):
        """
        Parameters
        ----------
        results : Dict[int, ChannelSortResult], optional
            既存のソーティング結果
        wideband_data : np.ndarray, optional
            生データ（新規ソーティング用）
        fs : float, optional
            サンプリングレート
        """
        self.results = results or {}
        self.wideband_data = wideband_data
        self.fs = fs
        
        self.current_channel = 0
        self.selected_units: List[int] = []
        
        # メインウィンドウ
        self.root = tk.Tk()
        self.root.title("Spike Sorting GUI")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 700)
        
        # スタイル設定
        style = ttk.Style()
        style.configure("Good.TLabel", foreground="green")
        style.configure("Warning.TLabel", foreground="orange")
        style.configure("Bad.TLabel", foreground="red")
        
        self._build_gui()
        
        # 初期表示
        if self.results:
            self._update_channel_list()
            self._update_display()
    
    def _build_gui(self):
        """GUIを構築"""
        # === 上部: コントロールパネル ===
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill='x', padx=5, pady=5)
        
        # チャンネル選択
        ttk.Label(control_frame, text="Channel:").pack(side='left', padx=5)
        
        self.channel_var = tk.StringVar()
        self.channel_combo = ttk.Combobox(control_frame, textvariable=self.channel_var,
                                          state='readonly', width=10)
        self.channel_combo.pack(side='left', padx=5)
        self.channel_combo.bind('<<ComboboxSelected>>', self._on_channel_change)
        
        ttk.Button(control_frame, text="◀ Prev", command=self._prev_channel).pack(side='left', padx=2)
        ttk.Button(control_frame, text="Next ▶", command=self._next_channel).pack(side='left', padx=2)
        
        ttk.Separator(control_frame, orient='vertical').pack(side='left', fill='y', padx=10)
        
        # 再クラスタリング
        ttk.Label(control_frame, text="Clusters:").pack(side='left', padx=5)
        self.n_clusters_var = tk.StringVar(value="4")
        self.n_clusters_spin = ttk.Spinbox(control_frame, from_=1, to=10, width=5,
                                           textvariable=self.n_clusters_var)
        self.n_clusters_spin.pack(side='left', padx=2)
        ttk.Button(control_frame, text="Re-cluster", command=self._recluster).pack(side='left', padx=5)
        
        ttk.Separator(control_frame, orient='vertical').pack(side='left', fill='y', padx=10)
        
        # 保存ボタン
        ttk.Button(control_frame, text="💾 Save NPZ", command=self._save_npz).pack(side='right', padx=5)
        ttk.Button(control_frame, text="📄 Export CSV", command=self._export_csv).pack(side='right', padx=5)
        
        # === 中央: プロットエリア ===
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Figure作成
        self.fig = Figure(figsize=(14, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # ナビゲーションツールバー
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(fill='x')
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        
        # === 下部: ユニットリスト & 操作 ===
        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(fill='x', padx=5, pady=5)
        
        # ユニットリスト（左側）
        list_frame = ttk.LabelFrame(bottom_frame, text="Units")
        list_frame.pack(side='left', fill='both', expand=True, padx=5)
        
        # Treeview
        columns = ('unit', 'n_spikes', 'amplitude', 'snr', 'isi_viol', 'status')
        self.unit_tree = ttk.Treeview(list_frame, columns=columns, show='headings', 
                                       height=6, selectmode='extended')
        
        self.unit_tree.heading('unit', text='Unit')
        self.unit_tree.heading('n_spikes', text='Spikes')
        self.unit_tree.heading('amplitude', text='Amplitude')
        self.unit_tree.heading('snr', text='SNR')
        self.unit_tree.heading('isi_viol', text='ISI viol%')
        self.unit_tree.heading('status', text='Status')
        
        self.unit_tree.column('unit', width=60, anchor='center')
        self.unit_tree.column('n_spikes', width=80, anchor='center')
        self.unit_tree.column('amplitude', width=100, anchor='center')
        self.unit_tree.column('snr', width=60, anchor='center')
        self.unit_tree.column('isi_viol', width=80, anchor='center')
        self.unit_tree.column('status', width=100, anchor='center')
        
        self.unit_tree.pack(side='left', fill='both', expand=True)
        
        # スクロールバー
        scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=self.unit_tree.yview)
        scrollbar.pack(side='right', fill='y')
        self.unit_tree.configure(yscrollcommand=scrollbar.set)
        
        # 選択イベント
        self.unit_tree.bind('<<TreeviewSelect>>', self._on_unit_select)
        
        # 操作ボタン（右側）- 2列構成
        action_frame = ttk.LabelFrame(bottom_frame, text="Actions")
        action_frame.pack(side='right', fill='y', padx=5)
        
        # 上段ボタン
        btn_row1 = ttk.Frame(action_frame)
        btn_row1.pack(fill='x', pady=2)
        ttk.Button(btn_row1, text="Merge", 
                   command=self._merge_selected, width=10).pack(side='left', padx=2)
        ttk.Button(btn_row1, text="Delete", 
                   command=self._delete_selected, width=10).pack(side='left', padx=2)
        
        # 中段ボタン
        btn_row2 = ttk.Frame(action_frame)
        btn_row2.pack(fill='x', pady=2)
        ttk.Button(btn_row2, text="Undelete", 
                   command=self._undelete_selected, width=10).pack(side='left', padx=2)
        ttk.Button(btn_row2, text="Mark MUA", 
                   command=self._mark_mua, width=10).pack(side='left', padx=2)
        
        # 下段ボタン
        btn_row3 = ttk.Frame(action_frame)
        btn_row3.pack(fill='x', pady=2)
        ttk.Button(btn_row3, text="Unmark MUA", 
                   command=self._unmark_mua, width=10).pack(side='left', padx=2)
        
        # 波形ブラウザボタン
        ttk.Separator(action_frame, orient='horizontal').pack(fill='x', pady=5)
        
        btn_row4 = ttk.Frame(action_frame)
        btn_row4.pack(fill='x', pady=2)
        ttk.Button(btn_row4, text="🔍 Browse 1by1", 
                   command=self._open_waveform_browser, width=12).pack(side='left', padx=2)
        ttk.Button(btn_row4, text="📊 View Grid", 
                   command=self._open_multi_viewer, width=10).pack(side='left', padx=2)
        
        # ステータスバー
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief='sunken')
        status_bar.pack(fill='x', side='bottom')
    
    def _update_channel_list(self):
        """チャンネルリストを更新"""
        channels = sorted(self.results.keys())
        self.channel_combo['values'] = [f"Ch {ch}" for ch in channels]
        
        if channels:
            self.current_channel = channels[0]
            self.channel_combo.set(f"Ch {self.current_channel}")
    
    def _on_channel_change(self, event=None):
        """チャンネル変更時"""
        selection = self.channel_var.get()
        if selection:
            self.current_channel = int(selection.replace("Ch ", ""))
            self._update_display()
    
    def _prev_channel(self):
        """前のチャンネルへ"""
        channels = sorted(self.results.keys())
        if not channels:
            return
        
        idx = channels.index(self.current_channel) if self.current_channel in channels else 0
        idx = (idx - 1) % len(channels)
        self.current_channel = channels[idx]
        self.channel_combo.set(f"Ch {self.current_channel}")
        self._update_display()
    
    def _next_channel(self):
        """次のチャンネルへ"""
        channels = sorted(self.results.keys())
        if not channels:
            return
        
        idx = channels.index(self.current_channel) if self.current_channel in channels else 0
        idx = (idx + 1) % len(channels)
        self.current_channel = channels[idx]
        self.channel_combo.set(f"Ch {self.current_channel}")
        self._update_display()
    
    def _update_display(self):
        """表示を更新"""
        if self.current_channel not in self.results:
            return
        
        result = self.results[self.current_channel]
        
        # ユニットリスト更新
        self._update_unit_list(result)
        
        # プロット更新
        self._update_plots(result)
        
        # ステータス更新
        n_units = len([u for u in result.units if not u.is_noise])
        n_spikes = sum(u.n_spikes for u in result.units if not u.is_noise)
        self.status_var.set(f"Channel {self.current_channel}: {n_units} units, {n_spikes} spikes")
    
    def _update_unit_list(self, result: ChannelSortResult):
        """ユニットリストを更新"""
        # 既存のアイテムを削除
        for item in self.unit_tree.get_children():
            self.unit_tree.delete(item)
        
        # ユニットを追加
        for unit in result.units:
            # ステータス決定
            if unit.is_noise:
                status = "🗑️ Noise"
            elif unit.is_mua:
                status = "📊 MUA"
            elif unit.isi_violation_rate < 2:
                status = "✓ Good"
            elif unit.isi_violation_rate < 5:
                status = "⚠ Questionable"
            else:
                status = "⚠ Poor"
            
            self.unit_tree.insert('', 'end', iid=str(unit.unit_id),
                                  values=(f"Unit {unit.unit_id}",
                                         unit.n_spikes,
                                         f"{unit.mean_amplitude:.4f}",
                                         f"{unit.snr:.1f}",
                                         f"{unit.isi_violation_rate:.1f}%",
                                         status))
    
    def _on_unit_select(self, event=None):
        """ユニット選択時"""
        selection = self.unit_tree.selection()
        self.selected_units = [int(iid) for iid in selection]
        
        # 選択されたユニットをハイライト
        self._update_plots(self.results.get(self.current_channel))
    
    def _update_plots(self, result: ChannelSortResult):
        """プロットを更新"""
        self.fig.clear()
        
        if result is None or not result.units:
            self.canvas.draw()
            return
        
        # サブプロット作成
        gs = self.fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        ax_waveform = self.fig.add_subplot(gs[0, 0])
        ax_pca12 = self.fig.add_subplot(gs[0, 1])
        ax_pca13 = self.fig.add_subplot(gs[0, 2])
        ax_isi = self.fig.add_subplot(gs[1, 0])
        ax_autocorr = self.fig.add_subplot(gs[1, 1])
        ax_info = self.fig.add_subplot(gs[1, 2])
        
        # 波形時間軸
        if result.waveform_time_ms is not None:
            time_ms = result.waveform_time_ms
        else:
            time_ms = np.linspace(-0.5, 1.0, result.all_waveforms.shape[1] if result.all_waveforms is not None else 60)
        
        # 各ユニットをプロット
        for unit in result.units:
            alpha = 0.8 if unit.unit_id in self.selected_units or not self.selected_units else 0.3
            linewidth = 2 if unit.unit_id in self.selected_units else 1
            
            if unit.is_noise:
                color = 'gray'
                alpha = 0.2
            else:
                color = unit.color
            
            # 1. 波形
            mean_wf = np.mean(unit.waveforms, axis=0)
            std_wf = np.std(unit.waveforms, axis=0)
            ax_waveform.plot(time_ms, mean_wf, color=color, linewidth=linewidth, 
                            alpha=alpha, label=f'Unit {unit.unit_id}')
            ax_waveform.fill_between(time_ms, mean_wf - std_wf, mean_wf + std_wf, 
                                    color=color, alpha=alpha * 0.2)
            
            # 2. PCA (PC1 vs PC2)
            ax_pca12.scatter(unit.pca_features[:, 0], unit.pca_features[:, 1],
                            c=color, s=10, alpha=alpha, label=f'Unit {unit.unit_id}')
            
            # 3. PCA (PC1 vs PC3)
            if unit.pca_features.shape[1] > 2:
                ax_pca13.scatter(unit.pca_features[:, 0], unit.pca_features[:, 2],
                                c=color, s=10, alpha=alpha)
            
            # 4. ISI (選択されたユニットのみ、または全て)
            if unit.unit_id in self.selected_units or not self.selected_units:
                if not unit.is_noise:
                    bins, hist = compute_isi_histogram(unit.spike_times)
                    if len(bins) > 0:
                        ax_isi.bar(bins, hist, width=1.0, color=color, alpha=alpha * 0.7,
                                  label=f'Unit {unit.unit_id}')
            
            # 5. 自己相関（選択されたユニットのみ）
            if unit.unit_id in self.selected_units and not unit.is_noise:
                bins, autocorr = compute_autocorrelogram(unit.spike_times)
                if len(bins) > 0:
                    ax_autocorr.bar(bins, autocorr, width=1.0, color=color, alpha=0.7)
        
        # 軸ラベル等の設定
        ax_waveform.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax_waveform.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax_waveform.set_xlabel('Time (ms)')
        ax_waveform.set_ylabel('Amplitude')
        ax_waveform.set_title('Mean Waveforms')
        ax_waveform.legend(loc='upper right', fontsize=8)
        
        ax_pca12.set_xlabel('PC1')
        ax_pca12.set_ylabel('PC2')
        ax_pca12.set_title('PCA: PC1 vs PC2')
        
        ax_pca13.set_xlabel('PC1')
        ax_pca13.set_ylabel('PC3')
        ax_pca13.set_title('PCA: PC1 vs PC3')
        
        ax_isi.axvline(x=2, color='red', linestyle='--', alpha=0.7, label='2ms')
        ax_isi.set_xlabel('ISI (ms)')
        ax_isi.set_ylabel('Count')
        ax_isi.set_title('ISI Histogram')
        ax_isi.set_xlim(0, 100)
        ax_isi.legend(loc='upper right', fontsize=8)
        
        ax_autocorr.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax_autocorr.set_xlabel('Lag (ms)')
        ax_autocorr.set_ylabel('Count')
        ax_autocorr.set_title('Autocorrelogram')
        
        # 情報パネル
        ax_info.axis('off')
        info_text = f"Channel {self.current_channel}\n"
        info_text += f"Threshold: {result.threshold:.4f}\n"
        info_text += f"Sigma: {result.sigma:.4f}\n\n"
        
        if result.pca_explained_variance is not None:
            var = result.pca_explained_variance
            info_text += f"PCA Variance:\n  PC1: {var[0]:.1%}\n  PC2: {var[1]:.1%}\n  PC3: {var[2]:.1%}\n\n"
        
        info_text += "Selected Units:\n"
        for unit in result.units:
            if unit.unit_id in self.selected_units:
                info_text += f"  Unit {unit.unit_id}: n={unit.n_spikes}, ISI={unit.isi_violation_rate:.1f}%\n"
        
        ax_info.text(0.1, 0.9, info_text, transform=ax_info.transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        self.fig.suptitle(f"Channel {self.current_channel} - Spike Sorting", fontsize=12)
        
        self.canvas.draw()
    
    def _merge_selected(self):
        """選択されたユニットをマージ"""
        if len(self.selected_units) < 2:
            messagebox.showwarning("Warning", "Select at least 2 units to merge")
            return
        
        result = self.results.get(self.current_channel)
        if result:
            self.results[self.current_channel] = merge_units(result, self.selected_units)
            self.selected_units = []
            self._update_display()
            self.status_var.set(f"Merged units")
    
    def _delete_selected(self):
        """選択されたユニットを削除"""
        if not self.selected_units:
            return
        
        result = self.results.get(self.current_channel)
        if result:
            for uid in self.selected_units:
                delete_unit(result, uid)
            self.selected_units = []
            self._update_display()
            self.status_var.set(f"Deleted units")
    
    def _undelete_selected(self):
        """削除を取り消し"""
        if not self.selected_units:
            return
        
        result = self.results.get(self.current_channel)
        if result:
            for uid in self.selected_units:
                undelete_unit(result, uid)
            self.selected_units = []
            self._update_display()
            self.status_var.set(f"Undeleted units")
    
    def _mark_mua(self):
        """MUAとしてマーク"""
        if not self.selected_units:
            return
        
        result = self.results.get(self.current_channel)
        if result:
            for uid in self.selected_units:
                mark_as_mua(result, uid)
            self._update_display()
            self.status_var.set(f"Marked as MUA")
    
    def _unmark_mua(self):
        """MUAマークを解除"""
        if not self.selected_units:
            return
        
        result = self.results.get(self.current_channel)
        if result:
            for uid in self.selected_units:
                unmark_mua(result, uid)
            self._update_display()
            self.status_var.set(f"Unmarked MUA")
    
    def _recluster(self):
        """再クラスタリング"""
        try:
            n_clusters = int(self.n_clusters_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid number of clusters")
            return
        
        result = self.results.get(self.current_channel)
        if result:
            self.results[self.current_channel] = recluster(result, n_clusters)
            self.selected_units = []
            self._update_display()
            self.status_var.set(f"Re-clustered with {n_clusters} clusters")
    
    def _save_npz(self):
        """NPZで保存"""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".npz",
            filetypes=[("NPZ files", "*.npz"), ("All files", "*.*")],
            initialfile="spike_sorting_result.npz"
        )
        
        if filepath:
            try:
                save_sorting_results(self.results, filepath)
                messagebox.showinfo("Success", f"Saved to {filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Save failed: {e}")
    
    def _export_csv(self):
        """CSVでエクスポート"""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="spike_times.csv"
        )
        
        if filepath:
            try:
                export_spike_times_csv(self.results, filepath)
                messagebox.showinfo("Success", f"Exported to {filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")
    
    def _open_waveform_browser(self):
        """波形ブラウザを開く（選択されたユニット or 全スパイク）"""
        result = self.results.get(self.current_channel)
        if result is None or result.all_waveforms is None:
            messagebox.showwarning("Warning", "No spike data available")
            return
        
        try:
            from waveform_browser import WaveformBrowser
            
            # 選択されたユニットがあればそのユニット、なければ全スパイク
            unit = None
            if self.selected_units:
                for u in result.units:
                    if u.unit_id == self.selected_units[0]:
                        unit = u
                        break
            
            WaveformBrowser(result, unit)
        except ImportError as e:
            messagebox.showerror("Error", f"Could not load waveform browser: {e}")
    
    def _open_multi_viewer(self):
        """マルチ波形ビューアを開く"""
        result = self.results.get(self.current_channel)
        if result is None or result.all_waveforms is None:
            messagebox.showwarning("Warning", "No spike data available")
            return
        
        try:
            from waveform_browser import MultiWaveformViewer
            
            # 選択されたユニットがあればそのユニット、なければ全スパイク
            unit = None
            if self.selected_units:
                for u in result.units:
                    if u.unit_id == self.selected_units[0]:
                        unit = u
                        break
            
            MultiWaveformViewer(result, unit)
        except ImportError as e:
            messagebox.showerror("Error", f"Could not load multi viewer: {e}")
    
    def run(self):
        """GUIを実行"""
        self.root.mainloop()


def launch_spike_sorting_gui(wideband_data: np.ndarray = None, 
                             fs: float = None,
                             results: Dict[int, ChannelSortResult] = None):
    """
    スパイクソーティングGUIを起動
    
    Parameters
    ----------
    wideband_data : np.ndarray, optional
        生データ (n_samples, n_channels)
    fs : float, optional
        サンプリングレート
    results : Dict[int, ChannelSortResult], optional
        既存のソーティング結果
    """
    # データがあれば自動ソーティング
    if results is None and wideband_data is not None and fs is not None:
        print("自動ソーティング実行中...")
        config = SortingConfig()
        results = sort_all_channels(wideband_data, fs, config, verbose=True)
    
    # GUI起動
    gui = SpikeSortingGUI(results=results, wideband_data=wideband_data, fs=fs)
    gui.run()


# === スタンドアロン実行用 ===
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Spike Sorting GUI")
    parser.add_argument("--plx", type=str, help="PLX file path")
    args = parser.parse_args()
    
    if args.plx:
        import neo
        
        print(f"Loading {args.plx}...")
        plx = neo.io.PlexonIO(filename=args.plx)
        data = plx.read()
        seg = data[0].segments[0]
        
        # 40kHzデータを探す
        wideband = None
        fs = None
        for sig in seg.analogsignals:
            if float(sig.sampling_rate) >= 20000:
                wideband = np.array(sig)
                fs = float(sig.sampling_rate)
                break
        
        if wideband is not None:
            launch_spike_sorting_gui(wideband, fs)
        else:
            print("Error: No wideband data found")
    else:
        # デモモード（ダミーデータ）
        print("Demo mode - no data loaded")
        print("Usage: python spike_sorting_gui.py --plx your_file.plx")
