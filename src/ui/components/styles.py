# src/ui/components/styles.py
from dataclasses import dataclass

COLORS = {
    "bg": "#0f172a",          # slate-900
    "panel": "#111827",       # gray-900
    "muted": "#1f2937",       # gray-800
    "accent": "#6366f1",      # indigo-500
    "accent_hover": "#4f46e5",# indigo-600
    "text": "#e5e7eb",        # gray-200
    "subtext": "#9ca3af",     # gray-400
    "success": "#10b981",     # emerald-500
    "danger": "#ef4444",      # red-500
    "warn": "#f59e0b",        # amber-500
    "border": "#1f2937",
}

@dataclass
class Spacing:
    xs: int = 4
    sm: int = 8
    md: int = 12
    lg: int = 16
    xl: int = 24
    xxl: int = 32

SPACE = Spacing()

FONTS = {
    "h1": ("Segoe UI", 18, "bold"),
    "h2": ("Segoe UI", 14, "bold"),
    "p":  ("Segoe UI", 11, "normal"),
    "btn":("Segoe UI", 11, "bold"),
    "mono":("Consolas", 10, "normal"),
}
