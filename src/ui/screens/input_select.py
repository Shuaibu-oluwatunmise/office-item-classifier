import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

from components.widgets import header, Card, Button, label, heading, spacer, divider
from components.styles import COLORS, SPACE

class InputSelectScreen(tk.Frame):
    def __init__(self, parent, mode, model_path, on_process, on_back):
        """
        on_process(sources) -> start processing
        """
        super().__init__(parent, bg=COLORS["light"])
        self.mode = mode
        self.model_path = model_path
        self.on_process = on_process
        self.on_back = on_back
        self.sources = []  # list[Path]

        title = f"{mode.title()} — Select Input"
        header(self, title).pack(fill="x")

        # Main content
        content = tk.Frame(self, bg=COLORS["light"])
        content.pack(fill="both", expand=True, padx=SPACE["xxl"], pady=SPACE["xxl"])

        # Model info banner
        info_card = Card(content, shadow="md")
        info_card.pack(fill="x", pady=(0, SPACE["lg"]))
        
        label(info_card.master, "📦 Using Model:", variant="secondary", size=10).pack(anchor="w")
        spacer(info_card.master, 2).pack()
        label(
            info_card.master, 
            self.model_path.parent.parent.name, 
            variant="primary", 
            weight="bold",
            size=12
        ).pack(anchor="w")

        # Input selection card
        input_card = Card(content, shadow="md")
        input_card.pack(fill="x", pady=(0, SPACE["lg"]))

        heading(input_card.master, "Choose Input Source", level=2).pack(anchor="w")
        spacer(input_card.master, SPACE["md"]).pack()

        # Selection buttons in a grid
        btn_grid = tk.Frame(input_card.master, bg=input_card.master.cget("bg"))
        btn_grid.pack(fill="x", pady=(0, SPACE["lg"]))

        # File selection option
        file_option = tk.Frame(
            btn_grid, 
            bg=COLORS["white"],
            highlightthickness=1,
            highlightbackground=COLORS["border"]
        )
        file_option.pack(side="left", fill="both", expand=True, padx=(0, SPACE["sm"]))

        label(file_option, "📄 Select Files", variant="primary", weight="bold", size=11).pack(pady=(SPACE["md"], 4))
        label(
            file_option, 
            "Choose one or more image files",
            variant="secondary",
            size=9
        ).pack(pady=(0, SPACE["sm"]))
        
        Button(
            file_option,
            text="Browse Files...",
            command=self._choose_files,
            variant="primary",
            size="sm"
        ).pack(pady=(0, SPACE["md"]))

        # Folder selection option
        folder_option = tk.Frame(
            btn_grid,
            bg=COLORS["white"],
            highlightthickness=1,
            highlightbackground=COLORS["border"]
        )
        folder_option.pack(side="left", fill="both", expand=True)

        label(folder_option, "📁 Select Folder", variant="primary", weight="bold", size=11).pack(pady=(SPACE["md"], 4))
        label(
            folder_option,
            "Choose a folder containing images",
            variant="secondary",
            size=9
        ).pack(pady=(0, SPACE["sm"]))
        
        Button(
            folder_option,
            text="Browse Folder...",
            command=self._choose_folder,
            variant="dark",
            size="sm"
        ).pack(pady=(0, SPACE["md"]))

        divider(input_card.master).pack(fill="x", pady=SPACE["md"])

        # Selection status
        status_frame = tk.Frame(input_card.master, bg=input_card.master.cget("bg"))
        status_frame.pack(fill="x")
        
        label(status_frame, "Selected Input:", variant="secondary", size=10).pack(side="left")
        spacer(status_frame, SPACE["sm"]).pack(side="left")
        
        self.sel_lbl = label(status_frame, "None", variant="primary", weight="bold", size=11)
        self.sel_lbl.pack(side="left")

        # Action buttons
        action_card = Card(content, shadow="md")
        action_card.pack(fill="x")

        action_row = tk.Frame(action_card.master, bg=action_card.master.cget("bg"))
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
            title="Select image(s)",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All Files", "*.*")]
        )
        if paths:
            self.sources = [Path(p) for p in paths]
            count = len(self.sources)
            self.sel_lbl.config(
                text=f"{count} file{'s' if count > 1 else ''} selected",
                fg=COLORS["success"]
            )

    def _choose_folder(self):
        p = filedialog.askdirectory(title="Select folder of images")
        if p:
            self.sources = [Path(p)]
            folder_name = Path(p).name
            self.sel_lbl.config(
                text=f"📁 {folder_name}",
                fg=COLORS["success"]
            )

    def _process(self):
        if not self.sources:
            messagebox.showerror(
                "No Input Selected", 
                "Please select files or a folder before processing."
            )
            return
        self.on_process(self.sources)