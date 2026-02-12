"""
font_manager.py - フォントサイズ一括管理

GUIウィジェットとmatplotlibプロットのフォントサイズを統一制御。
"""
import tkinter as tk
from tkinter import ttk
import matplotlib as mpl
from contextlib import contextmanager


# matplotlib デフォルト値（復元用）
_MPL_DEFAULTS = {
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 12,
}


def apply_gui_font_size(root: tk.Tk, size: int):
    """GUIウィジェット全体のフォントサイズを変更"""
    style = ttk.Style(root)
    default_font = ('', size)
    bold_font = ('', size, 'bold')

    style.configure('.', font=default_font)
    style.configure('TLabel', font=default_font)
    style.configure('TButton', font=default_font)
    style.configure('TCheckbutton', font=default_font)
    style.configure('TRadiobutton', font=default_font)
    style.configure('TCombobox', font=default_font)
    style.configure('TEntry', font=default_font)
    style.configure('TLabelframe.Label', font=bold_font)

    # Heading用
    style.configure('Heading.TLabel', font=bold_font)
    style.configure('SidebarStep.TButton', font=default_font, padding=6)
    style.configure('Run.TButton', font=bold_font, padding=8)

    # Treeview
    style.configure('Treeview', font=default_font, rowheight=int(size * 2))
    style.configure('Treeview.Heading', font=bold_font)

    root.option_add('*Font', f'TkDefaultFont {size}')


def apply_plot_font_size(base_size: int, scale: float = 1.0):
    """matplotlibのフォントサイズをグローバルに設定"""
    s = max(4, int(base_size * scale))
    mpl.rcParams['font.size'] = s
    mpl.rcParams['axes.titlesize'] = s + 2
    mpl.rcParams['axes.labelsize'] = s
    mpl.rcParams['xtick.labelsize'] = s - 1
    mpl.rcParams['ytick.labelsize'] = s - 1
    mpl.rcParams['legend.fontsize'] = s - 1
    mpl.rcParams['figure.titlesize'] = s + 4


@contextmanager
def plot_font_context(base_size: int, scale: float = 1.0):
    """プロット呼び出し時に一時的にフォントサイズを変更するコンテキストマネージャ"""
    old = {k: mpl.rcParams[k] for k in _MPL_DEFAULTS}
    apply_plot_font_size(base_size, scale)
    try:
        yield
    finally:
        for k, v in old.items():
            mpl.rcParams[k] = v


def restore_mpl_defaults():
    """matplotlibのフォント設定をデフォルトに戻す"""
    for k, v in _MPL_DEFAULTS.items():
        mpl.rcParams[k] = v
