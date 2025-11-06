import tkinter as tk
from tkinter import ttk
from components.styles import COLORS, FONTS, SPACE, RADIUS

# ---- Modern Card with Shadow Effect ----------------------------------------

class Card(tk.Frame):
    """Enhanced card with rounded appearance and shadow simulation"""
    def __init__(self, parent, shadow="md", padding=None, **kw):
        bg = kw.pop("bg", COLORS["card_bg"])
        
        # Outer frame for shadow effect
        self.outer = tk.Frame(parent, bg=COLORS["light"])
        super().__init__(self.outer, bg=bg, highlightthickness=0, bd=0)
        
        # Padding
        pad = padding if padding else SPACE["lg"]
        self.configure(padx=pad, pady=pad)
        
        # Add subtle border for depth
        if shadow == "md":
            self.configure(
                highlightthickness=1,
                highlightbackground=COLORS["border"],
                highlightcolor=COLORS["border"]
            )
        
        # Pack self inside outer frame
        super().pack(padx=2, pady=2, fill="both", expand=True)
        
    def pack(self, **kwargs):
        """Override pack to apply to outer frame"""
        self.outer.pack(**kwargs)
        
    def grid(self, **kwargs):
        """Override grid to apply to outer frame"""
        self.outer.grid(**kwargs)

# ---- Enhanced Header with Gradient Effect ----------------------------------

def header(parent, text):
    """Modern header with better typography and subtle gradient"""
    wrap = tk.Frame(parent, bg=COLORS["dark"], height=60)
    wrap.pack_propagate(False)
    
    # Add subtle bottom border
    border = tk.Frame(wrap, bg=COLORS["primary"], height=3)
    border.pack(side="bottom", fill="x")
    
    lbl = tk.Label(
        wrap, text=text, bg=COLORS["dark"], fg="white",
        font=(FONTS["heading"], 20, "bold"), pady=16
    )
    lbl.pack()
    return wrap

# ---- Modern Button with Rounded Appearance ---------------------------------

class Button(tk.Button):
    """Enhanced button with rounded appearance and smooth hover effects"""
    def __init__(self, parent, text, command=None, variant="primary", size="md", **kw):
        # Determine colors based on variant
        color_map = {
            "primary": (COLORS["primary"], COLORS["primary_hover"]),
            "secondary": (COLORS["secondary"], COLORS["secondary_hover"]),
            "success": (COLORS["success"], COLORS["success_hover"]),
            "warning": (COLORS["warning"], COLORS["warning_hover"]),
            "danger": (COLORS["danger"], COLORS["danger_hover"]),
            "dark": (COLORS["dark"], COLORS["dark_hover"]),
        }
        
        self.default_bg, self.hover_bg = color_map.get(variant, color_map["primary"])
        self.fg = kw.pop("fg", "white")
        
        # Size variations
        size_map = {
            "sm": (10, 10, 6),
            "md": (11, 14, 8),
            "lg": (12, 18, 10)
        }
        font_size, padx, pady = size_map.get(size, size_map["md"])
        
        super().__init__(
            parent, text=text, command=command, cursor="hand2",
            bg=self.default_bg, fg=self.fg, bd=0, relief="flat",
            activebackground=self.hover_bg, activeforeground=self.fg,
            font=(FONTS["base"], font_size, "bold"), 
            padx=padx, pady=pady,
            highlightthickness=0
        )
        
        # Hover effects with visual feedback
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        
    def _on_enter(self, e):
        self.configure(bg=self.hover_bg)
        
    def _on_leave(self, e):
        self.configure(bg=self.default_bg)
        
    def _on_press(self, e):
        # Slight darken effect on press
        self.configure(relief="sunken")
        
    def _on_release(self, e):
        self.configure(relief="flat")

# ---- Icon Button (for compact actions) -------------------------------------

class IconButton(tk.Button):
    """Small circular/square button for icons or single characters"""
    def __init__(self, parent, text, command=None, **kw):
        bg = kw.pop("bg", COLORS["light"])
        hover_bg = kw.pop("hover_bg", COLORS["border"])
        
        super().__init__(
            parent, text=text, command=command, cursor="hand2",
            bg=bg, fg=COLORS["text_primary"], bd=0, relief="flat",
            font=(FONTS["base"], 10, "bold"),
            width=3, height=1,
            highlightthickness=0
        )
        
        self.default_bg = bg
        self.hover_bg = hover_bg
        
        self.bind("<Enter>", lambda e: self.configure(bg=self.hover_bg))
        self.bind("<Leave>", lambda e: self.configure(bg=self.default_bg))

# ---- Enhanced Labels -------------------------------------------------------

def label(parent, text, variant="primary", **kw):
    """Enhanced label with variants"""
    color_map = {
        "primary": COLORS["text_primary"],
        "secondary": COLORS["text_secondary"],
        "white": "white",
        "success": COLORS["success"],
        "danger": COLORS["danger"]
    }
    
    fg = kw.pop("fg", color_map.get(variant, color_map["primary"]))
    font_weight = kw.pop("weight", "normal")
    font_size = kw.pop("size", 11)
    
    return tk.Label(
        parent, text=text, 
        bg=kw.get("bg", parent.cget("bg")),
        fg=fg, 
        font=(FONTS["base"], font_size, font_weight)
    )

def heading(parent, text, level=2, **kw):
    """Heading component with different levels"""
    size_map = {1: 18, 2: 16, 3: 14}
    size = size_map.get(level, 14)
    
    return tk.Label(
        parent, text=text,
        bg=kw.get("bg", parent.cget("bg")),
        fg=kw.get("fg", COLORS["text_primary"]),
        font=(FONTS["heading"], size, "bold")
    )

# ---- Badge Component (for status indicators) -------------------------------

class Badge(tk.Label):
    """Pill-shaped badge for status/tags"""
    def __init__(self, parent, text, variant="primary", **kw):
        color_map = {
            "primary": (COLORS["primary"], "white"),
            "success": (COLORS["success"], "white"),
            "warning": (COLORS["warning"], "white"),
            "danger": (COLORS["danger"], "white"),
            "secondary": (COLORS["light"], COLORS["text_primary"])
        }
        
        bg, fg = color_map.get(variant, color_map["primary"])
        
        super().__init__(
            parent, text=text, bg=bg, fg=fg,
            font=(FONTS["base"], 9, "bold"),
            padx=10, pady=4,
            highlightthickness=0
        )

# ---- Spacer ----------------------------------------------------------------

def spacer(parent, h=8):
    f = tk.Frame(parent, height=h, bg=parent.cget("bg"))
    f.pack_propagate(False)
    return f

# ---- Enhanced Scrollable Frame ---------------------------------------------

class ScrollFrame(tk.Frame):
    """Scrollable frame with modern scrollbar styling"""
    def __init__(self, parent, **kw):
        bg = kw.pop("bg", parent.cget("bg"))
        super().__init__(parent, bg=bg)
        
        # Canvas with subtle border
        canvas = tk.Canvas(
            self, bg=bg, highlightthickness=0, bd=0
        )
        
        # Custom styled scrollbar
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

        # Mouse wheel support
        def _wheel(evt):
            canvas.yview_scroll(int(-1*(evt.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _wheel)

# ---- Input Container (for form-like layouts) -------------------------------

class InputGroup(tk.Frame):
    """Container for label + input with consistent spacing"""
    def __init__(self, parent, label_text, **kw):
        super().__init__(parent, bg=parent.cget("bg"))
        
        # Label
        lbl = label(self, label_text, variant="primary", weight="bold", size=11)
        lbl.pack(anchor="w", pady=(0, 4))
        
        # Container for the input widget
        self.input_container = tk.Frame(self, bg=self.cget("bg"))
        self.input_container.pack(fill="x")

# ---- Divider ---------------------------------------------------------------

def divider(parent):
    """Horizontal divider line"""
    line = tk.Frame(parent, bg=COLORS["border"], height=1)
    line.pack(fill="x", pady=SPACE["md"])
    return line