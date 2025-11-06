import tkinter as tk
from tkinter import ttk
from threading import Thread
from pathlib import Path

from components.widgets import header, Card, Button, label, heading, spacer
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
        self.is_cancelled = False

        header(self, f"Processing {mode.title()}...").pack(fill="x")

        # Main content
        content = tk.Frame(self, bg=COLORS["light"])
        content.pack(fill="both", expand=True, padx=SPACE["xxl"], pady=SPACE["xxl"])

        # Processing card with centered content
        process_card = Card(content, shadow="md")
        process_card.pack(fill="both", expand=True)

        # Center container
        center = tk.Frame(process_card.master, bg=process_card.master.cget("bg"))
        center.place(relx=0.5, rely=0.5, anchor="center")

        # Processing icon/indicator
        icon_label = label(
            center,
            "⚙️",
            variant="primary",
            size=48
        )
        icon_label.pack()

        spacer(center, SPACE["lg"]).pack()

        heading(center, "Processing Your Images", level=1).pack()
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

        # Progress bar with custom styling
        prog_frame = tk.Frame(center, bg=center.cget("bg"))
        prog_frame.pack()

        style = ttk.Style()
        style.theme_use('clam')
        style.configure(
            "Custom.Horizontal.TProgressbar",
            troughcolor=COLORS["light"],
            background=COLORS["primary"],
            borderwidth=0,
            thickness=8
        )

        self.prog = ttk.Progressbar(
            prog_frame,
            mode="indeterminate",
            length=400,
            style="Custom.Horizontal.TProgressbar"
        )
        self.prog.pack()
        self.prog.start(10)

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

    def _worker(self):
        try:
            self.after(0, lambda: self.status.config(text="Loading model..."))
            self.after(500, lambda: self.status.config(text="Processing images..."))
            
            out_dir = run_inference(self.mode, self.model_path, self.sources)
            
            if not self.is_cancelled:
                self.after(0, lambda: self.status.config(text="✓ Complete!"))
                self.after(0, lambda: self.prog.stop())
                self.after(500, lambda: self._done(out_dir))
        except Exception as e:
            if not self.is_cancelled:
                self.after(0, lambda: self.prog.stop())
                self.after(0, lambda: self.status.config(
                    text=f"❌ Error: {str(e)}",
                    fg=COLORS["danger"]
                ))
                self.after(3000, self.on_cancel)

    def _done(self, out_dir: Path):
        self.prog.stop()
        self.on_done(out_dir)

    def _cancel(self):
        self.is_cancelled = True
        self.prog.stop()
        self.status.config(text="Cancelled by user")
        self.after(500, self.on_cancel)