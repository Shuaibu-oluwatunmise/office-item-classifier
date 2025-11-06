import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

from components.widgets import header, Card, Button, label, spacer
from components.styles import COLORS, FONTS, SPACE
from utils.handlers import discover_models

class HomeScreen(tk.Frame):
    def __init__(self, parent, on_go_next):
        super().__init__(parent, bg=COLORS["light"])
        self.on_go_next = on_go_next  # (mode, model_path) -> None

        # Header
        header(self, "Office Item Classifier — Home").pack(fill="x")

        wrap = Card(self, bg=COLORS["white"])
        wrap.pack(padx=SPACE["xl"], pady=SPACE["xl"], fill="x")

        label(wrap, "Select Task", fg=COLORS["dark"]).pack(anchor="w")
        spacer(wrap, 4).pack()
        self.mode_var = tk.StringVar(value="classification")
        modes = [("Classification", "classification"), ("Detection", "detection")]
        row = tk.Frame(wrap, bg=wrap.cget("bg"))
        row.pack(anchor="w")
        for text, val in modes:
            tk.Radiobutton(row, text=text, value=val, variable=self.mode_var,
                           bg=wrap.cget("bg"), font=(FONTS["base"], 10)).pack(side="left", padx=12)

        spacer(wrap, 14).pack()
        self.models = discover_models()

        # Model dropdown (auto-updating by mode)
        label(wrap, "Model").pack(anchor="w")
        self.model_var = tk.StringVar(value="")
        self.model_paths = {}  # display -> Path

        self.model_combo = ttk.Combobox(
            wrap, textvariable=self.model_var, state="readonly", width=46
        )
        self.model_combo.pack(fill="x", pady=(4,10))

        # Populate initial based on default mode
        self._populate_models()
        self.mode_var.trace_add("write", lambda *_: self._populate_models())

        # Continue button
        Button(wrap, text="Continue →", command=self._continue).pack(anchor="e", pady=(8,0))

    def _populate_models(self):
        mode = self.mode_var.get()
        items = self.models.get(mode, [])
        display = []
        self.model_paths.clear()

        if not items:
            self.model_combo["values"] = []
            self.model_var.set("")
            return

        for run_name, p in items:
            disp = f"{run_name}  ({p.as_posix()})"
            display.append(disp)
            self.model_paths[disp] = p

        # default: if only one, pre-select
        self.model_combo["values"] = display
        if len(display) == 1:
            self.model_var.set(display[0])
        else:
            if self.model_var.get() not in display:
                self.model_var.set(display[0])

    def _continue(self):
        mode = self.mode_var.get()
        disp = self.model_var.get()
        if not disp:
            messagebox.showerror("Select Model", "Please choose a model.")
            return
        model_path = self.model_paths[disp]
        self.on_go_next(mode, model_path)
