import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

from components.widgets import header, Card, Button, label, spacer
from components.styles import COLORS, SPACE

class InputSelectScreen(tk.Frame):
    def __init__(self, parent, mode, model_path, on_process, on_back):
        """
        on_process(sources) -> start processing
        """
        super().__init__(parent, bg=COLORS["light"])
        self.mode = mode
        self.model_path = model_path
        self.on_process = on_process
        self.on_back = on_back
        self.sources = []  # list[Path]

        title = f"{mode.title()} — Select Input"
        header(self, title).pack(fill="x")

        wrap = Card(self, bg=COLORS["white"])
        wrap.pack(padx=SPACE["xl"], pady=SPACE["xl"], fill="x")

        # File/folder row
        row = tk.Frame(wrap, bg=wrap.cget("bg"))
        row.pack(fill="x")
        Button(row, text="Choose File(s)", command=self._choose_files).pack(side="left")
        Button(row, text="Choose Folder", command=self._choose_folder, bg=COLORS["dark"]).pack(side="left", padx=8)

        # (Live feed kept for future; for now we focus on file/folder)
        # Button(row, text="Live Camera (beta)", command=self._live_camera, bg=COLORS["warning"]).pack(side="left", padx=8)

        spacer(wrap, 10).pack()
        self.sel_lbl = label(wrap, "No input selected", fg=COLORS["dark"])
        self.sel_lbl.pack(anchor="w")

        spacer(wrap, 20).pack()
        actions = tk.Frame(wrap, bg=wrap.cget("bg"))
        actions.pack(fill="x")
        Button(actions, text="← Back", command=self.on_back, bg=COLORS["dark"]).pack(side="left")
        Button(actions, text="Process →", command=self._process, bg=COLORS["success"]).pack(side="right")

    def _choose_files(self):
        paths = filedialog.askopenfilenames(
            title="Select image(s)",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if paths:
            self.sources = [Path(p) for p in paths]
            self.sel_lbl.config(text=f"Selected {len(self.sources)} file(s).")

    def _choose_folder(self):
        p = filedialog.askdirectory(title="Select folder of images")
        if p:
            self.sources = [Path(p)]
            self.sel_lbl.config(text=f"Selected folder: {p}")

    def _process(self):
        if not self.sources:
            messagebox.showerror("No input", "Please select file(s) or a folder first.")
            return
        self.on_process(self.sources)
