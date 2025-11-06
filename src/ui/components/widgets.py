import tkinter as tk
from tkinter import ttk
from components.styles import COLORS, FONTS, SPACE

# ---- Containers -------------------------------------------------------------

class Card(tk.Frame):
    def __init__(self, parent, **kw):
        bg = kw.pop("bg", COLORS["light"])
        super().__init__(parent, bg=bg, highlightthickness=0, bd=0)
        self.configure(padx=SPACE["lg"], pady=SPACE["md"])

def header(parent, text):
    wrap = tk.Frame(parent, bg=COLORS["dark"])
    lbl = tk.Label(
        wrap, text=text, bg=COLORS["dark"], fg="white",
        font=(FONTS["heading"], 18, "bold"), pady=12
    )
    lbl.pack()
    return wrap

# ---- Buttons (tk.Button for hover color support) ---------------------------

class Button(tk.Button):
    def __init__(self, parent, text, command=None, **kw):
        self.default_bg = kw.pop("bg", COLORS["primary"])
        self.hover_bg = kw.pop("hover_bg", COLORS["secondary"])
        self.fg = kw.pop("fg", "white")
        super().__init__(
            parent, text=text, command=command, cursor="hand2",
            bg=self.default_bg, fg=self.fg, bd=0, relief="flat",
            activebackground=self.hover_bg, activeforeground=self.fg,
            font=(FONTS["base"], 11, "bold"), padx=14, pady=8
        )
        # hover effects
        self.bind("<Enter>", lambda e: self.configure(bg=self.hover_bg))
        self.bind("<Leave>", lambda e: self.configure(bg=self.default_bg))

# ---- Inputs ----------------------------------------------------------------

def label(parent, text, **kw):
    return tk.Label(parent, text=text, bg=kw.get("bg", parent.cget("bg")),
                    fg=kw.get("fg", COLORS["dark"]), font=(FONTS["base"], 11))

def spacer(parent, h=8):
    f = tk.Frame(parent, height=h, bg=parent.cget("bg"))
    f.pack_propagate(False)
    return f

# ---- Scrollable frame (for results gallery) --------------------------------

class ScrollFrame(tk.Frame):
    def __init__(self, parent, **kw):
        bg = kw.pop("bg", parent.cget("bg"))
        super().__init__(parent, bg=bg)
        canvas = tk.Canvas(self, bg=bg, highlightthickness=0)
        vsb = tk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.inner = tk.Frame(canvas, bg=bg)
        self.inner.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.inner, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)

        canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        # Mouse wheel
        def _wheel(evt):
            canvas.yview_scroll(int(-1*(evt.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _wheel)
