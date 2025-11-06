COLORS = {
    "primary": "#5B5FDE",      # Blue-purple
    "primary_hover": "#4A4EC8",  # Darker on hover
    "secondary": "#7C3AED",    # Purple accent
    "secondary_hover": "#6D2FD4",
    "success": "#10B981",      # Green
    "success_hover": "#0FA373",
    "warning": "#F59E0B",      # Orange
    "warning_hover": "#E08E0A",
    "danger": "#EF4444",       # Red
    "danger_hover": "#DC2626",
    "dark": "#1F2937",         # Dark gray
    "dark_hover": "#111827",
    "light": "#F9FAFB",        # Light gray background
    "white": "#FFFFFF",
    "bg_gradient_start": "#E0E7FF",
    "bg_gradient_end": "#C7D2FE",
    "border": "#E5E7EB",       # Light border
    "text_primary": "#111827",
    "text_secondary": "#6B7280",
    "shadow_light": "#00000010",
    "shadow_medium": "#00000020",
    "card_bg": "#FFFFFF",
}

FONTS = {
    "base": "Segoe UI",
    "heading": "Segoe UI Semibold",
    "mono": "Consolas"
}

SPACE = {
    "xs": 4,
    "sm": 8,
    "md": 12,
    "lg": 16,
    "xl": 24,
    "xxl": 32
}

# Border radius values
RADIUS = {
    "sm": 4,
    "md": 8,
    "lg": 12,
    "xl": 16,
    "full": 999  # For pill shapes
}

# Shadow configurations (we'll simulate with borders/frames)
SHADOWS = {
    "sm": {
        "relief": "flat",
        "bd": 0,
        "highlightthickness": 1,
        "highlightbackground": COLORS["shadow_light"]
    },
    "md": {
        "relief": "flat",
        "bd": 0,
        "highlightthickness": 2,
        "highlightbackground": COLORS["shadow_medium"]
    },
    "lg": {
        "relief": "raised",
        "bd": 2,
        "highlightthickness": 0
    }
}