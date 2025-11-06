# src/ui/screens/input_select.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from components.widgets import header, Button, card
from components.styles import SPACE
from utils.handlers import start_live_classification, start_live_detection

class InputSelectScreen(ttk.Frame):
    def __init__(self, master, mode: str, go_back, go_process):
        super().__init__(master, style="TFrame")
        self.mode = mode
        self.go_back = go_back
        self.go_process = go_process

        title = "Classification" if mode == "classification" else "Detection"
        h = header(self, f"{title} – Pick input", "Choose files/folder or start live camera")
        h.grid(row=0, column=0, sticky="ew", padx=SPACE.lg, pady=(SPACE.lg, SPACE.md))

        wrap = card(self)
        wrap.grid(row=1, column=0, sticky="nsew", padx=SPACE.lg, pady=SPACE.lg)
        wrap.columnconfigure(0, weight=1)
        wrap.columnconfigure(1, weight=1)

        # Files/folder
        left = ttk.Frame(wrap, style="Muted.TFrame", padding=SPACE.lg)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, SPACE.lg))
        ttk.Label(left, text="Process Files", style="H2.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(left, text="Select an image/video or a folder containing media.", style="Sub.TLabel").grid(row=1, column=0, sticky="w", pady=(SPACE.xs, SPACE.lg))
        Button(left, text="Choose File", command=self._choose_file).grid(row=2, column=0, sticky="w", pady=(0, SPACE.sm))
        Button(left, text="Choose Folder", command=self._choose_folder).grid(row=3, column=0, sticky="w")
        self.sel_label = ttk.Label(left, text="", style="Sub.TLabel")
        self.sel_label.grid(row=4, column=0, sticky="w", pady=(SPACE.md,0))
        Button(left, text="Process", kind="primary", command=self._process).grid(row=5, column=0, sticky="w", pady=(SPACE.lg,0))

        # Live feed
        right = ttk.Frame(wrap, style="Muted.TFrame", padding=SPACE.lg)
        right.grid(row=0, column=1, sticky="nsew")
        ttk.Label(right, text="Live Camera", style="H2.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(right, text="Launch the live stream window with predictions.", style="Sub.TLabel").grid(row=1, column=0, sticky="w", pady=(SPACE.xs, SPACE.lg))
        Button(right, text="Start Live Feed", command=self._start_live).grid(row=2, column=0, sticky="w")

        # Nav
        nav = ttk.Frame(self, style="TFrame", padding=SPACE.lg)
        nav.grid(row=2, column=0, sticky="ew")
        Button(nav, text="Back", command=self.go_back).grid(row=0, column=0, sticky="w")

        self._selected_path = None

    def _choose_file(self):
        p = filedialog.askopenfilename(title="Select image/video")
        if p:
            self._selected_path = p
            self.sel_label.configure(text=p)

    def _choose_folder(self):
        p = filedialog.askdirectory(title="Select folder")
        if p:
            self._selected_path = p
            self.sel_label.configure(text=p)

    def _process(self):
        if not self._selected_path:
            messagebox.showwarning("No selection", "Please choose a file or folder.")
            return
        self.go_process(mode=self.mode, input_path=self._selected_path)

    def _start_live(self):
        # Spawn your existing live scripts
        if self.mode == "classification":
            start_live_classification()
        else:
            start_live_detection()
