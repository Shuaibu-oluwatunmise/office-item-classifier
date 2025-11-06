import tkinter as tk
from tkinter import ttk
from threading import Thread
from pathlib import Path

from components.widgets import header, Card, Button, label, spacer
from components.styles import COLORS, SPACE
from utils.handlers import run_inference

class ProcessingScreen(tk.Frame):
    def __init__(self, parent, mode, model_path, sources, on_done, on_cancel):
        """
        on_done(output_dir: Path) -> go show results
        """
        super().__init__(parent, bg=COLORS["light"])
        self.mode = mode
        self.model_path = model_path
        self.sources = sources
        self.on_done = on_done
        self.on_cancel = on_cancel

        header(self, "Processing...").pack(fill="x")

        wrap = Card(self, bg=COLORS["white"])
        wrap.pack(padx=SPACE["xl"], pady=SPACE["xl"], fill="x")

        label(wrap, f"Running {mode} with model:\n{model_path}").pack(anchor="w")
        spacer(wrap, 12).pack()

        self.prog = ttk.Progressbar(wrap, mode="indeterminate", length=400)
        self.prog.pack(pady=(4,12))
        self.prog.start(12)

        self.status = label(wrap, "Please wait...")
        self.status.pack(anchor="w")

        spacer(wrap, 20).pack()
        Button(wrap, text="Cancel", command=self._cancel, bg=COLORS["danger"]).pack(anchor="e")

        # Start background job
        Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            out_dir = run_inference(self.mode, self.model_path, self.sources)
            self.after(0, lambda: self._done(out_dir))
        except Exception as e:
            self.after(0, lambda: self.status.config(text=f"Error: {e}"))
            self.after(3000, self.on_cancel)

    def _done(self, out_dir: Path):
        self.prog.stop()
        self.on_done(out_dir)

    def _cancel(self):
        # We can't easily stop YOLO mid-run here, so just return to previous screen.
        self.on_cancel()
