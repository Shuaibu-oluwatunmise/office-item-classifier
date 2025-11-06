# src/ui/components/widgets.py
import tkinter as tk
from tkinter import ttk
from components.styles import COLORS, FONTS, SPACE

def configure_root(root: tk.Tk, title="Office Item Classifier"):
    root.title(title)
    root.configure(bg=COLORS["bg"])
    root.geometry("980x650")
    root.minsize(900, 580)

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TFrame", background=COLORS["bg"])
    style.configure("Panel.TFrame", background=COLORS["panel"])
    style.configure("Muted.TFrame", background=COLORS["muted"])
    style.configure("TLabel", background=COLORS["bg"], foreground=COLORS["text"], font=FONTS["p"])
    style.configure("H1.TLabel", font=FONTS["h1"])
    style.configure("H2.TLabel", font=FONTS["h2"], foreground=COLORS["text"])
    style.configure("Sub.TLabel", foreground=COLORS["subtext"])
    style.configure("TSeparator", background=COLORS["border"])
    style.configure("Card.TFrame", background=COLORS["muted"])

class Button(ttk.Button):
    def __init__(self, master, text, command=None, kind="primary", **kw):
        super().__init__(master, text=text, command=command, **kw)
        self.kind = kind
        self.default_bg = COLORS["accent"] if kind == "primary" else COLORS["muted"]
        self.hover_bg = COLORS["accent_hover"] if kind == "primary" else "#374151"
        self.configure(style="Accent.TButton")
        self._apply_style()
        self.bind("<Enter>", lambda e: self.configure(background=self.hover_bg))
        self.bind("<Leave>", lambda e: self.configure(background=self.default_bg))

    def _apply_style(self):
        style = ttk.Style(self)
        style.configure("Accent.TButton",
                        background=self.default_bg,
                        foreground="white",
                        font=FONTS["btn"],
                        borderwidth=0,
                        padding=(14, 8))
        style.map("Accent.TButton",
                  background=[('active', self.hover_bg)])

def card(master, **grid):
    frame = ttk.Frame(master, style="Card.TFrame", padding=SPACE.lg)
    if grid:
        frame.grid(**grid)
    return frame

def header(master, title, subtitle=None):
    wrap = ttk.Frame(master, style="TFrame", padding=(SPACE.lg, SPACE.md))
    ttk.Label(wrap, text=title, style="H1.TLabel").grid(row=0, column=0, sticky="w")
    if subtitle:
        ttk.Label(wrap, text=subtitle, style="Sub.TLabel").grid(row=1, column=0, sticky="w")
    return wrap
