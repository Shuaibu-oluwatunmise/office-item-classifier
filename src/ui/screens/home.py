import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

from components.widgets import header, Card, Button, label, heading, spacer, InputGroup, divider
from components.styles import COLORS, FONTS, SPACE
from utils.handlers import discover_models

class HomeScreen(tk.Frame):
    def __init__(self, parent, on_go_next):
        super().__init__(parent, bg=COLORS["bg_darkest"])
        self.on_go_next = on_go_next  # (mode, model_path) -> None

        # Header
        header(self, "Office Item Classifier").pack(fill="x")

        # Main content container with max width for better readability
        content = tk.Frame(self, bg=COLORS["bg_darkest"])
        content.pack(fill="both", expand=True, padx=SPACE["xxl"], pady=SPACE["xxl"])
        
        # Welcome section with glow
        welcome_card = Card(content, shadow="md", glow=True)
        welcome_card.pack(fill="x", pady=(0, SPACE["lg"]))
        
        heading(welcome_card.outer, "Welcome!", level=1).pack(anchor="w")
        spacer(welcome_card.outer, 4).pack()
        label(
            welcome_card.outer, 
            "Select your task and model to begin processing media",
            variant="secondary",
            size=12
        ).pack(anchor="w")
        
        # Configuration card
        config_card = Card(content, shadow="md")
        config_card.pack(fill="x", pady=(0, SPACE["lg"]))
        
        heading(config_card.outer, "Configuration", level=2).pack(anchor="w")
        spacer(config_card.outer, SPACE["md"]).pack()
        
        # Task selection with better styling
        task_group = InputGroup(config_card.outer, "Select Task")
        task_group.pack(fill="x", pady=(0, SPACE["lg"]))
        
        self.mode_var = tk.StringVar(value="classification")
        
        # Radio button container with cards
        radio_container = tk.Frame(task_group.input_container, bg=task_group.cget("bg"))
        radio_container.pack(fill="x")
        
        modes = [
            ("📊 Classification", "classification", "Classify images into categories"),
            ("🎯 Detection", "detection", "Detect and locate objects in images")
        ]
        
        for idx, (text, val, desc) in enumerate(modes):
            # Each option in a subtle card with elevated background
            option_frame = tk.Frame(
                radio_container, 
                bg=COLORS["card_elevated"],
                highlightthickness=2,
                highlightbackground=COLORS["border"]
            )
            option_frame.pack(side="left", fill="x", expand=True, padx=(0, SPACE["sm"] if idx == 0 else 0))
            
            rb = tk.Radiobutton(
                option_frame, text=text, value=val, variable=self.mode_var,
                bg=COLORS["card_elevated"], fg=COLORS["text_primary"],
                font=(FONTS["base"], 11, "bold"),
                selectcolor=COLORS["primary"],
                activebackground=COLORS["card_elevated"],
                cursor="hand2"
            )
            rb.pack(anchor="w", padx=SPACE["md"], pady=(SPACE["sm"], 2))
            
            desc_lbl = label(option_frame, desc, variant="secondary", size=9)
            desc_lbl.pack(anchor="w", padx=SPACE["md"], pady=(0, SPACE["sm"]))
        
        divider(config_card.outer).pack(fill="x", pady=SPACE["md"])
        
        # Model selection
        self.models = discover_models()
        
        model_group = InputGroup(config_card.outer, "Select Model")
        model_group.pack(fill="x")
        
        self.model_var = tk.StringVar(value="")
        self.model_paths = {}  # display -> Path
        
        # Styled combobox for dark theme
        style = ttk.Style()
        style.theme_use('clam')
        style.configure(
            'Custom.TCombobox',
            fieldbackground=COLORS["card_elevated"],
            background=COLORS["primary"],
            foreground=COLORS["text_primary"],
            borderwidth=2,
            relief="flat",
            padding=8
        )
        style.map('Custom.TCombobox',
                 fieldbackground=[('readonly', COLORS["card_elevated"])],
                 selectbackground=[('readonly', COLORS["primary"])],
                 selectforeground=[('readonly', COLORS["bg_darkest"])])
        
        self.model_combo = ttk.Combobox(
            model_group.input_container, 
            textvariable=self.model_var, 
            state="readonly",
            style='Custom.TCombobox',
            font=(FONTS["base"], 11),
            height=8
        )
        self.model_combo.pack(fill="x", ipady=4)
        
        spacer(config_card.outer, SPACE["md"]).pack()
        
        # Model info section - CREATE BEFORE _populate_models()
        self.model_info = label(
            config_card.outer, 
            "Select a trained model to continue", 
            variant="secondary",
            size=10
        )
        self.model_info.pack(anchor="w")
        
        # Populate initial based on default mode - AFTER model_info is created
        self._populate_models()
        self.mode_var.trace_add("write", lambda *_: self._populate_models())
        
        spacer(config_card.outer, SPACE["lg"]).pack()
        
        # Action buttons
        action_row = tk.Frame(config_card.outer, bg=config_card.outer.cget("bg"))
        action_row.pack(fill="x")
        
        Button(
            action_row, 
            text="Continue →", 
            command=self._continue,
            variant="primary",
            size="lg"
        ).pack(side="right")

    def _populate_models(self):
        mode = self.mode_var.get()
        items = self.models.get(mode, [])
        display = []
        self.model_paths.clear()

        if not items:
            self.model_combo["values"] = []
            self.model_var.set("")
            self.model_info.config(
                text=f"⚠️ No trained {mode} models found in runs/{mode}",
                fg=COLORS["warning"]
            )
            return

        for run_name, p in items:
            # Show just the run name in dropdown, store full path
            display.append(run_name)
            self.model_paths[run_name] = p

        self.model_combo["values"] = display
        if len(display) == 1:
            self.model_var.set(display[0])
            self.model_info.config(
                text=f"✓ Model loaded: {display[0]}",
                fg=COLORS["primary"]  # Neon green
            )
        else:
            if self.model_var.get() not in display:
                self.model_var.set(display[0])
            self.model_info.config(
                text=f"📦 {len(display)} models available",
                fg=COLORS["text_secondary"]
            )

    def _continue(self):
        mode = self.mode_var.get()
        model_name = self.model_var.get()
        if not model_name:
            messagebox.showerror("No Model Selected", "Please choose a trained model to continue.")
            return
        model_path = self.model_paths[model_name]
        self.on_go_next(mode, model_path)