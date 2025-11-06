COLORS = {
    # Neon green theme
    "primary": "#00FF87",      # Neon green - primary accent
    "primary_hover": "#00E676",  # Brighter on hover
    "primary_glow": "#00FF8740",  # Semi-transparent glow
    
    "secondary": "#00D9FF",    # Cyan accent
    "secondary_hover": "#00B8E6",
    
    "success": "#00FF87",      # Same as primary (neon green)
    "success_hover": "#00E676",
    
    "warning": "#FFD600",      # Bright yellow
    "warning_hover": "#FFC400",
    
    "danger": "#FF3D71",       # Bright red/pink
    "danger_hover": "#FF1744",
    
    # Dark backgrounds
    "bg_darkest": "#0A0E1A",   # Darkest - main background
    "bg_dark": "#141927",      # Dark - secondary background
    "bg_darker": "#1A1F35",    # Card background
    "bg_medium": "#242B42",    # Elevated elements
    
    # Legacy/compatibility
    "dark": "#1A1F35",
    "dark_hover": "#242B42",
    "light": "#0A0E1A",        # Main background (dark)
    "white": "#242B42",        # Card bg (dark)
    
    # Borders and dividers
    "border": "#2A3148",       # Subtle border
    "border_bright": "#3D4968", # Brighter border
    "border_glow": "#00FF8760", # Green glow border
    
    # Text colors
    "text_primary": "#E8EBF7",   # Light text
    "text_secondary": "#9BA3C1", # Muted text
    "text_muted": "#6B7493",     # Very muted
    
    # Shadows and glows
    "shadow_light": "#00000040",
    "shadow_medium": "#00000060",
    "shadow_dark": "#00000080",
    "glow_green": "#00FF8720",
    "glow_cyan": "#00D9FF20",
    
    # Card backgrounds
    "card_bg": "#1A1F35",
    "card_elevated": "#242B42",
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