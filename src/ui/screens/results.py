import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

from PIL import Image, ImageTk

from components.widgets import header, Card, Button, label, spacer, ScrollFrame
from components.styles import COLORS, SPACE
from utils.handlers import list_result_images, copy_results_to

THUMB_W = 240

class ResultsScreen(tk.Frame):
    def __init__(self, parent, mode, output_dir: Path, on_back, on_home):
        super().__init__(parent, bg=COLORS["light"])
        self.mode = mode
        self.output_dir = output_dir
        self.on_back = on_back
        self.on_home = on_home
        self._thumb_refs = []  # prevent GC

        header(self, f"{mode.title()} — Results").pack(fill="x")

        topbar = Card(self, bg=COLORS["white"])
        topbar.pack(padx=SPACE["xl"], pady=(SPACE["xl"], SPACE["md"]), fill="x")

        Button(topbar, text="← Back", command=self.on_back, bg=COLORS["dark"]).pack(side="left")
        Button(topbar, text="New Inference", command=self.on_home, bg=COLORS["primary"]).pack(side="left", padx=8)
        Button(topbar, text="Save Results...", command=self._save, bg=COLORS["success"]).pack(side="right")

        body = Card(self, bg=COLORS["white"])
        body.pack(padx=SPACE["xl"], pady=(SPACE["md"], SPACE["xl"]), fill="both", expand=True)

        self.grid_container = ScrollFrame(body, bg=body.cget("bg"))
        self.grid_container.pack(fill="both", expand=True)

        self._populate()

    def _populate(self):
        imgs = list_result_images(self.mode, self.output_dir)
        if not imgs:
            label(self.grid_container.inner, "No images found.", fg=COLORS["dark"]).pack()
            return

        # Build a thumbnail grid
        row = 0
        col = 0
        for p in imgs:
            try:
                im = Image.open(p).convert("RGB")
                ratio = THUMB_W / float(im.width)
                th = im.resize((THUMB_W, int(im.height * ratio)))
                tkimg = ImageTk.PhotoImage(th)
            except Exception:
                continue

            frm = tk.Frame(self.grid_container.inner, bg=self.grid_container.inner.cget("bg"), bd=0)
            frm.grid(row=row, column=col, padx=10, pady=10, sticky="n")

            canvas = tk.Label(frm, image=tkimg, bd=0)
            canvas.pack()
            self._thumb_refs.append(tkimg)

            cap = tk.Label(frm, text=p.name, bg=frm.cget("bg"), fg=COLORS["dark"])
            cap.pack()

            canvas.bind("<Button-1>", lambda e, path=p: self._open_full(path))

            col += 1
            if col >= 3:
                col = 0
                row += 1

    def _open_full(self, path: Path):
        top = tk.Toplevel(self)
        top.title(path.name)
        top.configure(bg=COLORS["dark"])
        try:
            im = Image.open(path).convert("RGB")
            w, h = im.size
            maxw, maxh = 1200, 900
            scale = min(maxw / w, maxh / h, 1.0)
            im = im.resize((int(w*scale), int(h*scale)))
            tkimg = ImageTk.PhotoImage(im)
            lbl = tk.Label(top, image=tkimg, bg=COLORS["dark"])
            lbl.image = tkimg  # keep ref
            lbl.pack(padx=12, pady=12)
        except Exception as e:
            tk.Label(top, text=f"Failed to open: {e}", bg=COLORS["dark"], fg="white").pack(padx=12, pady=12)

    def _save(self):
        imgs = list_result_images(self.mode, self.output_dir)
        if not imgs:
            messagebox.showinfo("Nothing to save", "No images to save.")
            return
        dest = filedialog.askdirectory(title="Choose destination folder")
        if not dest:
            return
        copy_results_to(Path(dest), imgs)
        messagebox.showinfo("Saved", f"Saved {len(imgs)} file(s) to:\n{dest}")
