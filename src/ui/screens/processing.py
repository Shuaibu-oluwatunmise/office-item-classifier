import tkinter as tk
from tkinter import ttk
from threading import Thread
from pathlib import Path

from components.widgets import header, Card, Button, label, heading, spacer
from components.styles import COLORS, SPACE
from utils.handlers import run_inference

class ProcessingScreen(tk.Frame):
    def __init__(self, parent, mode, model_path, sources, input_type, on_done, on_cancel):
        """
        on_done(output_dir: Path) -> go show results
        """
        super().__init__(parent, bg=COLORS["bg_darkest"])
        self.mode = mode
        self.model_path = model_path
        self.sources = sources
        self.input_type = input_type
        self.on_done = on_done
        self.on_cancel = on_cancel
        self.is_cancelled = False

        header(self, f"Processing {mode.title()}...").pack(fill="x")

        # Main content
        content = tk.Frame(self, bg=COLORS["bg_darkest"])
        content.pack(fill="both", expand=True, padx=SPACE["xxl"], pady=SPACE["xxl"])

        # Processing card with centered content
        process_card = Card(content, shadow="md", glow=True)
        process_card.pack(fill="both", expand=True)

        # Center container
        center = tk.Frame(process_card.outer, bg=process_card.outer.cget("bg"))
        center.place(relx=0.5, rely=0.5, anchor="center")

        # Processing icon/indicator with neon color
        icon_label = label(
            center,
            "⚙️",
            variant="neon",
            size=56
        )
        icon_label.pack()

        spacer(center, SPACE["xl"]).pack()

        heading(center, "Processing Your Media", level=1).pack()
        spacer(center, SPACE["sm"]).pack()

        # Model info
        model_info = label(
            center,
            f"Model: {self.model_path.parent.parent.name}",
            variant="secondary",
            size=11
        )
        model_info.pack()

        spacer(center, SPACE["xl"]).pack()

        # Progress bar with neon styling
        prog_frame = tk.Frame(center, bg=center.cget("bg"))
        prog_frame.pack()

        style = ttk.Style()
        style.theme_use('clam')
        style.configure(
            "Neon.Horizontal.TProgressbar",
            troughcolor=COLORS["bg_darker"],
            background=COLORS["primary"],
            borderwidth=0,
            thickness=10
        )

        self.prog = ttk.Progressbar(
            prog_frame,
            mode="determinate",
            length=450,
            maximum=100,
            style="Neon.Horizontal.TProgressbar"
        )
        self.prog.pack()

        spacer(center, SPACE["md"]).pack()

        # Progress percentage
        self.progress_pct = label(
            center,
            "0%",
            variant="neon",
            weight="bold",
            size=14
        )
        self.progress_pct.pack()

        spacer(center, SPACE["lg"]).pack()

        # Status message
        self.status = label(
            center,
            "Initializing...",
            variant="secondary",
            size=12
        )
        self.status.pack()

        spacer(center, SPACE["xl"]).pack()

        # Cancel button
        Button(
            center,
            text="Cancel",
            command=self._cancel,
            variant="danger",
            size="md"
        ).pack()

        # Start background job
        Thread(target=self._worker, daemon=True).start()

    def _update_progress(self, current, total, status_text):
        """Progress callback from worker"""
        if self.is_cancelled:
            return
        
        pct = int((current / total) * 100) if total > 0 else 0
        self.prog['value'] = pct
        self.progress_pct.config(text=f"{pct}%")
        self.status.config(text=status_text)

    def _worker(self):
        try:
            self.after(0, lambda: self.status.config(text="Loading model..."))
            
            out_dir = run_inference(
                self.mode, 
                self.model_path, 
                self.sources,
                self.input_type,
                progress_callback=lambda c, t, s: self.after(0, lambda: self._update_progress(c, t, s))
            )
            
            if not self.is_cancelled:
                self.after(0, lambda: self.status.config(text="✓ Complete!", fg=COLORS["primary"]))
                self.after(0, lambda: self.progress_pct.config(text="100%"))
                self.after(500, lambda: self._done(out_dir))
        except Exception as e:
            if not self.is_cancelled:
                self.after(0, lambda: self.status.config(
                    text=f"❌ Error: {str(e)}",
                    fg=COLORS["danger"]
                ))
                self.after(3000, self.on_cancel)

    def _done(self, out_dir: Path):
        self.on_done(out_dir)

    def _cancel(self):
        self.is_cancelled = True
        self.status.config(text="Cancelled by user", fg=COLORS["warning"])
        self.after(500, self.on_cancel)