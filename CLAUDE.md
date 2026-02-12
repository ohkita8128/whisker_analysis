# Neuronexus Whisker Analysis - Coding Rules

## tkinter Variable: master= 必須

tkinter の変数クラス (`StringVar`, `BooleanVar`, `IntVar`, `DoubleVar`) を生成する際は、
**必ず `master=` パラメータを指定すること**。省略すると Entry / Combobox / Checkbutton 等の
ウィジェットに初期値が表示されないバグが発生する。

```python
# OK
tk.StringVar(master=self.root, value="default")
tk.StringVar(master=self.winfo_toplevel(), value="default")
tk.BooleanVar(master=self.winfo_toplevel(), value=True)
tk.IntVar(master=self.root, value=9)
tk.DoubleVar(master=self._toplevel, value=1.0)

# NG - 初期値が表示されない
tk.StringVar(value="default")
tk.BooleanVar(value=True)
```

- `MainGUI` クラス内では `master=self.root`
- `StepPanel` サブクラス内では `master=self.winfo_toplevel()`
- `BandEditorFrame` 等の独立ウィジェット内では `master=self._toplevel` (コンストラクタで `self._toplevel = parent.winfo_toplevel()` を保持)

このルールは v6 の `phase_gui.py` および v7 の全ファイルで過去にバグとして発見・修正済み。
新しいtkinterコードを書く際は必ず守ること。
