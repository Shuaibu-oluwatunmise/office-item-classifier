# src/ui/screens/processing.py
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from components.widgets import header, Button, card
from components.styles import SPACE
from utils.handlers import classify_path, detect_path

class ProcessingScreen(ttk.Frame):
    def __init__(self, master, mode: str, input_path: str, go_back, go_results):
        super().__init__(master, style="TFrame")
        self.mode = mode
        self.input_path = Path(input_path)
        self.go_back = go_back
        self.go_results = go_results

        title = "Classification" if mode == "classification" else "Detection"
        h = header(self, f"{title} – Processing", f"Running model on: {self.input_path}")
        h.grid(row=0, column=0, sticky="ew", padx=SPACE.lg, pady=(SPACE.lg, SPACE.md))

        self.card = card(self)
        self.card.grid(row=1, column=0, sticky="nsew", padx=SPACE.lg, pady=SPACE.lg)
        self.info = ttk.Label(self.card, text="Starting...", style="Sub.TLabel")
        self.info.grid(row=0, column=0, sticky="w")

        nav = ttk.Frame(self, style="TFrame", padding=SPACE.lg)
        nav.grid(row=2, column=0, sticky="ew")
        Button(nav, text="Cancel", command=self.go_back).grid(row=0, column=0, sticky="w")

        self.after(100, self._kickoff)

    def _kickoff(self):
        t = threading.Thread(target=self._do_work, daemon=True)
        t.start()

    def _do_work(self):
        try:
            if self.mode == "classification":
                out_dir, saved = classify_path(self.input_path)
            else:
                out_dir, saved = detect_path(self.input_path)
            self.after(0, lambda: self._done(out_dir, saved))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", str(e)))

    def _done(self, out_dir, saved):
        self.info.configure(text=f"Finished. Saved {len(saved)} file(s) to:\n{out_dir}")
        self.go_results(mode=self.mode, output_dir=str(out_dir), files=[str(p) for p in saved])
