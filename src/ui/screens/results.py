# src/ui/screens/results.py
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from pathlib import Path
from components.widgets import header, Button, card
from components.styles import SPACE

THUMB = (360, 240)

class ResultsScreen(ttk.Frame):
    def __init__(self, master, mode: str, output_dir: str, files, go_home):
        super().__init__(master, style="TFrame")
        self.mode = mode
        self.output_dir = Path(output_dir)
        self.files = files
        self.go_home = go_home
        self._thumb_cache = []

        title = "Classification" if mode == "classification" else "Detection"
        h = header(self, f"{title} – Results", f"Saved to: {self.output_dir}")
        h.grid(row=0, column=0, sticky="ew", padx=SPACE.lg, pady=(SPACE.lg, SPACE.md))

        wrap = card(self)
        wrap.grid(row=1, column=0, sticky="nsew", padx=SPACE.lg, pady=SPACE.lg)
        wrap.columnconfigure(0, weight=1)
        wrap.rowconfigure(1, weight=1)

        ttk.Label(wrap, text=f"{len(self.files)} output file(s)", style="H2.TLabel").grid(row=0, column=0, sticky="w")

        canvas = tk.Canvas(wrap, bg=wrap.cget("background"), highlightthickness=0)
        vsb = ttk.Scrollbar(wrap, orient="vertical", command=canvas.yview)
        self.gridf = ttk.Frame(canvas, style="Card.TFrame")
        self.gridf.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=self.gridf, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)

        canvas.grid(row=1, column=0, sticky="nsew")
        vsb.grid(row=1, column=1, sticky="ns")

        # thumbnails
        cols = 2
        r = c = 0
        for f in self.files:
            if Path(f).suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}:
                tile = ttk.Frame(self.gridf, style="Muted.TFrame", padding=SPACE.sm)
                tile.grid(row=r, column=c, sticky="nsew", padx=SPACE.sm, pady=SPACE.sm)
                try:
                    im = Image.open(f).convert("RGB")
                    im.thumbnail(THUMB)
                    tkim = ImageTk.PhotoImage(im)
                    self._thumb_cache.append(tkim)
                    lbl = ttk.Label(tile, image=tkim)
                    lbl.grid(row=0, column=0)
                    ttk.Label(tile, text=os.path.basename(f)).grid(row=1, column=0, sticky="w")
                except Exception:
                    ttk.Label(tile, text=os.path.basename(f)).grid(row=0, column=0, sticky="w")
            c += 1
            if c >= cols:
                c = 0; r += 1

        nav = ttk.Frame(self, style="TFrame", padding=SPACE.lg)
        nav.grid(row=2, column=0, sticky="ew")
        Button(nav, text="Open Folder", command=lambda: os.startfile(self.output_dir)).grid(row=0, column=0, sticky="w")
        Button(nav, text="Done (Home)", command=self.go_home, kind="primary").grid(row=0, column=1, padx=(SPACE.sm,0), sticky="w")
