import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

from components.widgets import header, Card, Button, label, heading, spacer, divider
from components.styles import COLORS, SPACE

class InputSelectScreen(tk.Frame):
    def __init__(self, parent, mode, model_path, on_process, on_back):
        """
        on_process(sources, input_type) -> start processing
        input_type: "file", "folder", or "live"
        """
        super().__init__(parent, bg=COLORS["bg_darkest"])
        self.mode = mode
        self.model_path = model_path
        self.on_process = on_process
        self.on_back = on_back
        self.sources = []  # list[Path]
        self.input_type = None

        title = f"{mode.title()} — Select Input"
        header(self, title).pack(fill="x")

        # Main content
        content = tk.Frame(self, bg=COLORS["bg_darkest"])
        content.pack(fill="both", expand=True, padx=SPACE["xxl"], pady=SPACE["xxl"])

        # Model info banner with glow
        info_card = Card(content, shadow="md", glow=True)
        info_card.pack(fill="x", pady=(0, SPACE["lg"]))
        
        label(info_card.outer, "📦 Using Model:", variant="secondary", size=10).pack(anchor="w")
        spacer(info_card.outer, 2).pack()
        label(
            info_card.outer, 
            self.model_path.parent.parent.name, 
            variant="neon", 
            weight="bold",
            size=13
        ).pack(anchor="w")

        # Input selection card
        input_card = Card(content, shadow="md")
        input_card.pack(fill="x", pady=(0, SPACE["lg"]))

        heading(input_card.outer, "Choose Input Source", level=2).pack(anchor="w")
        spacer(input_card.outer, SPACE["md"]).pack()

        # Selection buttons in a 3-column grid
        btn_grid = tk.Frame(input_card.outer, bg=input_card.outer.cget("bg"))
        btn_grid.pack(fill="x", pady=(0, SPACE["lg"]))

        # File selection option
        file_option = tk.Frame(
            btn_grid, 
            bg=COLORS["card_elevated"],
            highlightthickness=2,
            highlightbackground=COLORS["border"]
        )
        file_option.pack(side="left", fill="both", expand=True, padx=(0, SPACE["sm"]))

        file_icon = label(file_option, "📄", variant="neon", size=32)
        file_icon.pack(pady=(SPACE["lg"], SPACE["sm"]))
        
        label(file_option, "Select Files", variant="primary", weight="bold", size=12).pack(pady=(0, 4))
        label(
            file_option, 
            "Images & Videos",
            variant="secondary",
            size=9
        ).pack(pady=(0, SPACE["md"]))
        
        Button(
            file_option,
            text="Browse Files",
            command=self._choose_files,
            variant="primary",
            size="sm"
        ).pack(pady=(0, SPACE["lg"]))

        # Folder selection option
        folder_option = tk.Frame(
            btn_grid,
            bg=COLORS["card_elevated"],
            highlightthickness=2,
            highlightbackground=COLORS["border"]
        )
        folder_option.pack(side="left", fill="both", expand=True, padx=(0, SPACE["sm"]))

        folder_icon = label(folder_option, "📁", variant="neon", size=32)
        folder_icon.pack(pady=(SPACE["lg"], SPACE["sm"]))
        
        label(folder_option, "Select Folder", variant="primary", weight="bold", size=12).pack(pady=(0, 4))
        label(
            folder_option,
            "Process all media",
            variant="secondary",
            size=9
        ).pack(pady=(0, SPACE["md"]))
        
        Button(
            folder_option,
            text="Browse Folder",
            command=self._choose_folder,
            variant="dark",
            size="sm"
        ).pack(pady=(0, SPACE["lg"]))

        # Live camera option
        live_option = tk.Frame(
            btn_grid,
            bg=COLORS["card_elevated"],
            highlightthickness=2,
            highlightbackground=COLORS["border"]
        )
        live_option.pack(side="left", fill="both", expand=True)

        live_icon = label(live_option, "📹", variant="neon", size=32)
        live_icon.pack(pady=(SPACE["lg"], SPACE["sm"]))
        
        label(live_option, "Live Camera", variant="primary", weight="bold", size=12).pack(pady=(0, 4))
        label(
            live_option,
            "Real-time inference",
            variant="secondary",
            size=9
        ).pack(pady=(0, SPACE["md"]))
        
        Button(
            live_option,
            text="Start Camera",
            command=self._live_camera,
            variant="success",
            size="sm"
        ).pack(pady=(0, SPACE["lg"]))

        divider(input_card.outer).pack(fill="x", pady=SPACE["md"])

        # Selection status with neon highlight
        status_frame = tk.Frame(input_card.outer, bg=input_card.outer.cget("bg"))
        status_frame.pack(fill="x")
        
        label(status_frame, "Selected:", variant="secondary", size=10).pack(side="left")
        spacer(status_frame, SPACE["sm"]).pack(side="left")
        
        self.sel_lbl = label(status_frame, "None", variant="muted", weight="bold", size=11)
        self.sel_lbl.pack(side="left")

        # Action buttons
        action_card = Card(content, shadow="md")
        action_card.pack(fill="x")

        action_row = tk.Frame(action_card.outer, bg=action_card.outer.cget("bg"))
        action_row.pack(fill="x")

        Button(
            action_row,
            text="← Back",
            command=self.on_back,
            variant="dark",
            size="md"
        ).pack(side="left")

        Button(
            action_row,
            text="Start Processing →",
            command=self._process,
            variant="success",
            size="lg"
        ).pack(side="right")

    def _choose_files(self):
        paths = filedialog.askopenfilenames(
            title="Select image(s) and/or video(s)",
            filetypes=[
                ("All Media", "*.jpg *.jpeg *.png *.bmp *.webp *.mp4 *.avi *.mov *.mkv"),
                ("Images", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("Videos", "*.mp4 *.avi *.mov *.mkv"),
                ("All Files", "*.*")
            ]
        )
        if paths:
            self.sources = [Path(p) for p in paths]
            self.input_type = "file"
            count = len(self.sources)
            self.sel_lbl.config(
                text=f"✓ {count} file{'s' if count > 1 else ''} selected",
                fg=COLORS["primary"]
            )

    def _choose_folder(self):
        p = filedialog.askdirectory(title="Select folder containing images/videos")
        if p:
            self.sources = [Path(p)]
            self.input_type = "folder"
            folder_name = Path(p).name
            self.sel_lbl.config(
                text=f"✓ 📁 {folder_name}",
                fg=COLORS["primary"]
            )

    def _live_camera(self):
        """Start live camera inference"""
        self.sources = []  # No file sources for live
        self.input_type = "live"
        self.sel_lbl.config(
            text="✓ Live camera mode",
            fg=COLORS["primary"]
        )
        # Immediately start processing for live mode
        self._process()

    def _process(self):
        if not self.input_type:
            messagebox.showerror(
                "No Input Selected", 
                "Please select files, folder, or live camera before processing."
            )
            return
        self.on_process(self.sources, self.input_type)