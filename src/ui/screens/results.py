import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

from PIL import Image, ImageTk

from components.widgets import header, Card, Button, label, heading, spacer, ScrollFrame
from components.styles import COLORS, SPACE
from utils.handlers import list_result_images, copy_results_to

THUMB_W = 260

class ResultsScreen(tk.Frame):
    def __init__(self, parent, mode, output_dir: Path, on_back, on_home):
        super().__init__(parent, bg=COLORS["bg_darkest"])
        self.mode = mode
        self.output_dir = output_dir
        self.on_back = on_back
        self.on_home = on_home
        self._thumb_refs = []  # prevent GC

        header(self, f"{mode.title()} — Results").pack(fill="x")

        # Main content
        content = tk.Frame(self, bg=COLORS["bg_darkest"])
        content.pack(fill="both", expand=True, padx=SPACE["xxl"], pady=SPACE["xxl"])

        # Action bar
        action_bar = Card(content, shadow="md", padding=SPACE["md"])
        action_bar.pack(fill="x", pady=(0, SPACE["lg"]))

        action_row = tk.Frame(action_bar.outer, bg=action_bar.outer.cget("bg"))
        action_row.pack(fill="x")

        # Left side buttons
        left_btns = tk.Frame(action_row, bg=action_row.cget("bg"))
        left_btns.pack(side="left")

        Button(
            left_btns,
            text="← Back",
            command=self.on_back,
            variant="dark",
            size="md"
        ).pack(side="left")

        Button(
            left_btns,
            text="🏠 New Task",
            command=self.on_home,
            variant="primary",
            size="md"
        ).pack(side="left", padx=(SPACE["sm"], 0))

        # Right side buttons
        Button(
            action_row,
            text="💾 Save Results",
            command=self._save,
            variant="success",
            size="md"
        ).pack(side="right")

        # Results container
        results_card = Card(content, shadow="md", padding=SPACE["lg"])
        results_card.pack(fill="both", expand=True)

        # Header
        imgs = list_result_images(self.mode, self.output_dir)
        result_header = tk.Frame(results_card.outer, bg=results_card.outer.cget("bg"))
        result_header.pack(fill="x", pady=(0, SPACE["lg"]))

        heading(result_header, "Processed Images", level=2).pack(side="left")
        label(
            result_header,
            f"{len(imgs)} file{'s' if len(imgs) != 1 else ''}",
            variant="secondary",
            size=11
        ).pack(side="left", padx=(SPACE["sm"], 0))

        if not imgs:
            # Empty state
            empty_container = tk.Frame(results_card.outer, bg=results_card.outer.cget("bg"))
            empty_container.pack(fill="both", expand=True)

            empty_label = label(
                empty_container,
                "📭\n\nNo results found",
                variant="secondary",
                size=14
            )
            empty_label.pack(expand=True)
            return

        # Scrollable grid
        self.grid_container = ScrollFrame(results_card.outer, bg=results_card.outer.cget("bg"))
        self.grid_container.pack(fill="both", expand=True)

        self._populate_grid(imgs)

    def _populate_grid(self, imgs):
        """Create a beautiful grid of thumbnails with hover effects"""
        row = 0
        col = 0
        max_cols = 3

        for p in imgs:
            # Check if it's a video
            is_video = p.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv', '.wmv')
            
            if is_video:
                # For videos, create a placeholder thumbnail with play icon
                thumb_card = tk.Frame(
                    self.grid_container.inner,
                    bg=COLORS["card_elevated"],
                    highlightthickness=1,
                    highlightbackground=COLORS["border"],
                    cursor="hand2",
                    width=THUMB_W,
                    height=int(THUMB_W * 0.75)
                )
                thumb_card.pack_propagate(False)
                thumb_card.grid(row=row, column=col, padx=SPACE["md"], pady=SPACE["md"], sticky="n")
                
                # Video icon
                video_icon = label(thumb_card, "🎬", variant="neon", size=48)
                video_icon.place(relx=0.5, rely=0.4, anchor="center")
                
                video_label = label(thumb_card, "VIDEO", variant="primary", size=12, weight="bold")
                video_label.place(relx=0.5, rely=0.6, anchor="center")
                
                # Filename label
                name_frame = tk.Frame(thumb_card, bg=COLORS["bg_darker"])
                name_frame.pack(side="bottom", fill="x", padx=SPACE["sm"], pady=SPACE["sm"])
                
                display_name = p.name
                if len(display_name) > 25:
                    display_name = display_name[:22] + "..."
                
                name_label = label(name_frame, display_name, variant="secondary", size=9)
                name_label.pack()
                
                # Click to open video
                def open_video(path=p):
                    import os
                    os.startfile(str(path))  # Windows
                
                thumb_card.bind("<Button-1>", lambda e, path=p: open_video(path))
                video_icon.bind("<Button-1>", lambda e, path=p: open_video(path))
                
            else:
                # For images, create normal thumbnail
                try:
                    im = Image.open(p).convert("RGB")
                    ratio = THUMB_W / float(im.width)
                    th = im.resize((THUMB_W, int(im.height * ratio)), Image.Resampling.LANCZOS)
                    tkimg = ImageTk.PhotoImage(th)
                except Exception:
                    continue

                # Thumbnail card with hover effect
                thumb_card = tk.Frame(
                    self.grid_container.inner,
                    bg=COLORS["card_elevated"],
                    highlightthickness=1,
                    highlightbackground=COLORS["border"],
                    cursor="hand2"
                )
                thumb_card.grid(row=row, column=col, padx=SPACE["md"], pady=SPACE["md"], sticky="n")

                # Image container
                img_container = tk.Frame(thumb_card, bg=COLORS["card_elevated"])
                img_container.pack(padx=SPACE["sm"], pady=(SPACE["sm"], 0))

                canvas = tk.Label(img_container, image=tkimg, bg=COLORS["card_elevated"], bd=0)
                canvas.pack()
                self._thumb_refs.append(tkimg)

                # Filename label
                name_frame = tk.Frame(thumb_card, bg=COLORS["bg_darker"])
                name_frame.pack(fill="x", padx=SPACE["sm"], pady=SPACE["sm"])

                # Truncate long names
                display_name = p.name
                if len(display_name) > 30:
                    display_name = display_name[:27] + "..."

                name_label = label(name_frame, display_name, variant="secondary", size=9)
                name_label.pack()

                # Hover effects
                def on_enter(e, card=thumb_card):
                    card.configure(
                        highlightthickness=2,
                        highlightbackground=COLORS["primary"]
                    )

                def on_leave(e, card=thumb_card):
                    card.configure(
                        highlightthickness=1,
                        highlightbackground=COLORS["border"]
                    )

                thumb_card.bind("<Enter>", on_enter)
                thumb_card.bind("<Leave>", on_leave)
                canvas.bind("<Enter>", on_enter)
                canvas.bind("<Leave>", on_leave)

                # Click to open full size
                canvas.bind("<Button-1>", lambda e, path=p: self._open_full(path))
                thumb_card.bind("<Button-1>", lambda e, path=p: self._open_full(path))

            col += 1
            if col >= max_cols:
                col = 0
                row += 1

    def _open_full(self, path: Path):
        """Open full-size image in a modern modal window"""
        modal = tk.Toplevel(self)
        modal.title(path.name)
        modal.configure(bg=COLORS["dark"])
        modal.transient(self)
        modal.grab_set()

        # Header bar
        header_bar = tk.Frame(modal, bg=COLORS["dark"])
        header_bar.pack(fill="x", padx=SPACE["md"], pady=(SPACE["md"], SPACE["sm"]))

        label(
            header_bar,
            path.name,
            variant="white",
            weight="bold",
            size=12
        ).pack(side="left")

        # Close button
        close_btn = Button(
            header_bar,
            text="✕",
            command=modal.destroy,
            variant="danger",
            size="sm"
        )
        close_btn.pack(side="right")

        # Image container with padding
        img_frame = tk.Frame(modal, bg=COLORS["dark"])
        img_frame.pack(padx=SPACE["lg"], pady=(0, SPACE["lg"]))

        try:
            im = Image.open(path).convert("RGB")
            w, h = im.size
            
            # Scale to fit screen nicely
            maxw, maxh = 1200, 900
            scale = min(maxw / w, maxh / h, 1.0)
            
            if scale < 1.0:
                im = im.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)
            
            tkimg = ImageTk.PhotoImage(im)
            
            # Image with subtle border
            img_container = tk.Frame(
                img_frame,
                bg=COLORS["white"],
                highlightthickness=2,
                highlightbackground=COLORS["border"]
            )
            img_container.pack()

            lbl = tk.Label(img_container, image=tkimg, bg=COLORS["white"])
            lbl.image = tkimg  # keep ref
            lbl.pack(padx=2, pady=2)
            
        except Exception as e:
            error_label = label(
                img_frame,
                f"❌ Failed to load image\n{str(e)}",
                variant="white",
                size=11
            )
            error_label.pack(padx=SPACE["xl"], pady=SPACE["xl"])

        # Center the modal
        modal.update_idletasks()
        x = (modal.winfo_screenwidth() // 2) - (modal.winfo_width() // 2)
        y = (modal.winfo_screenheight() // 2) - (modal.winfo_height() // 2)
        modal.geometry(f"+{x}+{y}")

    def _save(self):
        imgs = list_result_images(self.mode, self.output_dir)
        if not imgs:
            messagebox.showinfo("Nothing to Save", "No processed images to save.")
            return
        
        dest = filedialog.askdirectory(title="Choose destination folder")
        if not dest:
            return
        
        try:
            copy_results_to(Path(dest), imgs)
            messagebox.showinfo(
                "✓ Saved Successfully",
                f"Saved {len(imgs)} image{'s' if len(imgs) != 1 else ''} to:\n{dest}"
            )
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save images:\n{str(e)}")