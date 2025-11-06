# src/ui/screens/home.py
import tkinter as tk
from tkinter import ttk
from components.widgets import header, Button, card
from components.styles import COLORS, SPACE

class HomeScreen(ttk.Frame):
    def __init__(self, master, go_next):
        super().__init__(master, style="TFrame")
        self.go_next = go_next

        h = header(self, "Office Item Classifier", "Choose a mode to get started")
        h.grid(row=0, column=0, sticky="ew", padx=SPACE.lg, pady=(SPACE.lg, SPACE.md))

        wrap = card(self)
        wrap.grid(row=1, column=0, sticky="nsew", padx=SPACE.lg, pady=SPACE.lg)
        wrap.columnconfigure(0, weight=1)
        wrap.columnconfigure(1, weight=1)

        ttk.Label(wrap, text="Select Mode", style="H2.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Separator(wrap).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(SPACE.sm, SPACE.lg))

        c1 = ttk.Frame(wrap, style="Muted.TFrame", padding=SPACE.lg)
        c1.grid(row=2, column=0, sticky="nsew", padx=(0, SPACE.lg))
        c2 = ttk.Frame(wrap, style="Muted.TFrame", padding=SPACE.lg)
        c2.grid(row=2, column=1, sticky="nsew")

        ttk.Label(c1, text="Classification", style="H2.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(c1, text="Predict a single class for each image/video.", style="Sub.TLabel").grid(row=1, column=0, sticky="w", pady=(SPACE.xs, SPACE.lg))
        Button(c1, text="Use Classification", command=lambda: self.go_next(mode="classification")).grid(row=2, column=0, sticky="w")

        ttk.Label(c2, text="Detection", style="H2.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(c2, text="Find and label objects with bounding boxes.", style="Sub.TLabel").grid(row=1, column=0, sticky="w", pady=(SPACE.xs, SPACE.lg))
        Button(c2, text="Use Detection", command=lambda: self.go_next(mode="detection")).grid(row=2, column=0, sticky="w")
