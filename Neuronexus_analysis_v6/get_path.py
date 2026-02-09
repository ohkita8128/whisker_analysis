import tkinter as tk
from tkinter import filedialog
import os

def get_path(file_type='all', mode='file', initial_dir=".", 
             multiple=False, title=None):
    """
    ファイル/フォルダ選択の統合版
    
    Parameters:
    mode: 'file' (ファイル), 'folder' (フォルダ), 'save' (保存先)
    file_type: ファイルの種類
    multiple: 複数選択（modeが'file'の時のみ有効）
    """
    filetypes_dict = {
        'excel': [("Excel Files", "*.xlsx *.xls"), ("All Files", "*.*")],
        'csv': [("CSV Files", "*.csv"), ("All Files", "*.*")],
        'image': [("Image Files", "*.png *.jpg *.jpeg"), ("All Files", "*.*")],
        'python': [("Python Files", "*.py"), ("All Files", "*.*")],
        'plx': [("Plexon Files", "*.plx"), ("All Files", "*.*")],
        'all': [("All Files", "*.*")]
    }
    
    root = tk.Tk()
    root.withdraw()
    
    top = tk.Toplevel()
    top.withdraw()
    top.attributes('-topmost', True)
    
    result = None
    
    # モードに応じた処理
    if mode == 'file':
        if title is None:
            title = "ファイルを選択" + (" (複数可)" if multiple else "")
        
        filetypes = filetypes_dict.get(file_type, filetypes_dict['all'])
        
        if multiple:
            result = filedialog.askopenfilenames(
                parent=top, title=title, 
                initialdir=initial_dir, filetypes=filetypes
            )
            result = list(result) if result else None
        else:
            result = filedialog.askopenfilename(
                parent=top, title=title,
                initialdir=initial_dir, filetypes=filetypes
            )
            result = result if result else None
    
    elif mode == 'folder':
        if title is None:
            title = "フォルダを選択"
        result = filedialog.askdirectory(
            parent=top, title=title, initialdir=initial_dir
        )
        result = result if result else None
    
    elif mode == 'save':

        if title is None:
            title = "保存先を選択"
        filetypes = filetypes_dict.get(file_type, filetypes_dict['all'])

        default_extensions = {
            'excel': '.xlsx',
            'csv': '.csv',
            'plx': '.plx',
            'image': '.png',
            'python': '.py',
            'json': '.json',
            'all':''
        }


        result = filedialog.asksaveasfilename(
            parent=top, title=title,
            initialdir=initial_dir, filetypes=filetypes,
            defaultextension=default_extensions[file_type]
        )
        result = result if result else None
    
    top.destroy()
    root.destroy()
    
    # 結果表示
    if result:
        if isinstance(result, list):
            print(f" {len(result)} 個選択された。")
        else:
            print(f" 選択: {os.path.basename(result)}")
    else:
        print(" 選択されなかった。")
    
    return result