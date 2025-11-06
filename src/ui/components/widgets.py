import tkinter as tk
from tkinter import ttk
from components.styles import COLORS, FONTS, SPACE, RADIUS

# ---- Modern Card with Enhanced Shadow Effect -----------------------------

class Card(tk.Frame):
    """Enhanced card with dramatic depth effects"""
    def __init__(self, parent, shadow="md", padding=None, glow=False, **kw):
        bg = kw.pop("bg", COLORS["card_bg"])
        
        # Triple-layer shadow for depth using solid dark colors
        shadow3 = tk.Frame(parent, bg=COLORS["bg_dark"])
        shadow2 = tk.Frame(shadow3, bg=COLORS["bg_darker"])
        shadow1 = tk.Frame(shadow2, bg=COLORS["card_bg"])
        
        # Outer frame with optional glow
        if glow:
            self.outer = tk.Frame(
                shadow1, 
                bg=COLORS["card_bg"],
                highlightthickness=2,
                highlightbackground=COLORS["primary"]
            )
        else:
            self.outer = tk.Frame(shadow1, bg=COLORS["bg_darkest"])
        
        super().__init__(self.outer, bg=bg, highlightthickness=0, bd=0)
        
        # Padding
        pad = padding if padding else SPACE["lg"]
        self.configure(padx=pad, pady=pad)
        
        # Add border for definition
        self.configure(
            highlightthickness=1,
            highlightbackground=COLORS["border"],
            highlightcolor=COLORS["border"]
        )
        
        # Pack the layers with smaller offsets
        shadow3.pack(fill="both", expand=True)
        shadow2.pack(fill="both", expand=True, padx=2, pady=2)
        shadow1.pack(fill="both", expand=True, padx=1, pady=1)
        self.outer.pack(fill="both", expand=True, padx=1, pady=1)
        super().pack(fill="both", expand=True, padx=1, pady=1)
        
        # Store references
        self.shadow3 = shadow3
        
    def pack(self, **kwargs):
        """Override pack to apply to outermost shadow frame"""
        self.shadow3.pack(**kwargs)
        
    def grid(self, **kwargs):
        """Override grid to apply to outermost shadow frame"""
        self.shadow3.grid(**kwargs)

# ---- Enhanced Header with Neon Accent -------------------------------------

def header(parent, text):
    """Modern header with neon green accent bar"""
    wrap = tk.Frame(parent, bg=COLORS["bg_darkest"], height=70)
    wrap.pack_propagate(False)
    
    # Neon green accent bar at bottom
    border = tk.Frame(wrap, bg=COLORS["primary"], height=4)
    border.pack(side="bottom", fill="x")
    
    lbl = tk.Label(
        wrap, text=text, bg=COLORS["bg_darkest"], fg=COLORS["text_primary"],
        font=(FONTS["heading"], 22, "bold"), pady=16
    )
    lbl.pack()
    return wrap

# ---- Modern Button with Neon Glow Effect ----------------------------------

class Button(tk.Button):
    """Enhanced button with neon glow and depth effects"""
    def __init__(self, parent, text, command=None, variant="primary", size="md", **kw):
        # Determine colors based on variant
        color_map = {
            "primary": (COLORS["primary"], COLORS["primary_hover"], COLORS["bg_darkest"]),
            "secondary": (COLORS["secondary"], COLORS["secondary_hover"], COLORS["bg_darkest"]),
            "success": (COLORS["success"], COLORS["success_hover"], COLORS["bg_darkest"]),
            "warning": (COLORS["warning"], COLORS["warning_hover"], COLORS["bg_darkest"]),
            "danger": (COLORS["danger"], COLORS["danger_hover"], COLORS["bg_darkest"]),
            "dark": (COLORS["bg_medium"], COLORS["bg_darker"], COLORS["text_primary"]),
        }
        
        self.default_bg, self.hover_bg, self.fg = color_map.get(variant, color_map["primary"])
        
        # Size variations
        size_map = {
            "sm": (10, 12, 7),
            "md": (11, 16, 10),
            "lg": (12, 20, 12)
        }
        font_size, padx, pady = size_map.get(size, size_map["md"])
        
        # Create shadow/glow frame
        self.shadow_frame = tk.Frame(parent, bg=COLORS["bg_darkest"])
        
        super().__init__(
            self.shadow_frame, text=text, command=command, cursor="hand2",
            bg=self.default_bg, fg=self.fg, bd=0, relief="flat",
            activebackground=self.hover_bg, activeforeground=self.fg,
            font=(FONTS["base"], font_size, "bold"), 
            padx=padx, pady=pady,
            highlightthickness=2,
            highlightbackground=COLORS["border"],
            highlightcolor=COLORS["border"]
        )
        
        super().pack(padx=2, pady=2)
        
        # Store variant for glow effect
        self.variant = variant
        
        # Hover effects with glow
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        
    def _on_enter(self, e):
        self.configure(bg=self.hover_bg)
        # Add highlight for primary/success
        if self.variant in ["primary", "success"]:
            self.configure(
                highlightthickness=2,
                highlightbackground=COLORS["primary"]
            )
        
    def _on_leave(self, e):
        self.configure(bg=self.default_bg)
        self.configure(
            highlightthickness=2,
            highlightbackground=COLORS["border"]
        )
        
    def _on_press(self, e):
        self.configure(relief="sunken")
        
    def _on_release(self, e):
        self.configure(relief="flat")
    
    def pack(self, **kwargs):
        """Override pack to apply to shadow frame"""
        self.shadow_frame.pack(**kwargs)
    
    def grid(self, **kwargs):
        """Override grid to apply to shadow frame"""
        self.shadow_frame.grid(**kwargs)

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
    """Enhanced label with variants for dark theme"""
    color_map = {
        "primary": COLORS["text_primary"],
        "secondary": COLORS["text_secondary"],
        "muted": COLORS["text_muted"],
        "white": COLORS["text_primary"],
        "success": COLORS["success"],
        "danger": COLORS["danger"],
        "neon": COLORS["primary"]  # Neon green
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
    """Heading component with different levels and neon option"""
    size_map = {1: 20, 2: 16, 3: 14}
    size = size_map.get(level, 14)
    
    fg = kw.get("fg", COLORS["text_primary"])
    
    return tk.Label(
        parent, text=text,
        bg=kw.get("bg", parent.cget("bg")),
        fg=fg,
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