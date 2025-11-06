import tkinter as tk
from pathlib import Path

from components.styles import COLORS, FONTS
from components.widgets import header
from screens.home import HomeScreen
from screens.input_select import InputSelectScreen
from screens.processing import ProcessingScreen
from screens.live_camera import LiveCameraScreen
from screens.results import ResultsScreen

APP_W, APP_H = 1200, 800

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Office Item Classifier")
        self.configure(bg=COLORS["bg_darkest"])
        self.geometry(f"{APP_W}x{APP_H}")
        self.minsize(1100, 700)

        # Modern navigation bar with neon accents
        self.nav = tk.Frame(self, bg=COLORS["bg_darker"], height=55)
        self.nav.pack(fill="x")
        self.nav.pack_propagate(False)
        
        # Neon green accent bar at bottom of nav
        nav_border = tk.Frame(self, bg=COLORS["primary"], height=3)
        nav_border.pack(fill="x")
        
        self._make_nav()

        # Main container with dark background
        self.container = tk.Frame(self, bg=COLORS["bg_darkest"])
        self.container.pack(fill="both", expand=True)

        self.state = {
            "mode": None,
            "model_path": None,
            "sources": None,
            "input_type": None,
            "output_dir": None
        }

        self._to_home()

    # ---- Navigation Bar ----------------------------------------------------

    def _make_nav(self):
        """Create a modern navigation bar with neon green accents"""
        
        # Left side - branding and main nav
        left_nav = tk.Frame(self.nav, bg=COLORS["bg_darker"])
        left_nav.pack(side="left", fill="y")
        
        # App title/logo with neon accent
        logo_label = tk.Label(
            left_nav,
            text="🤖 Classifier",
            bg=COLORS["bg_darker"],
            fg=COLORS["primary"],  # Neon green
            font=(FONTS["heading"], 13, "bold"),
            padx=18
        )
        logo_label.pack(side="left", pady=12)
        
        # Separator
        sep = tk.Frame(left_nav, bg=COLORS["border_bright"], width=2)
        sep.pack(side="left", fill="y", padx=10, pady=10)
        
        def add_btn(text, cmd, side="left", parent=None):
            target = parent if parent else left_nav
            b = tk.Button(
                target, text=text, command=cmd, 
                bg=COLORS["bg_darker"], fg=COLORS["text_primary"], 
                bd=0, relief="flat", 
                padx=18, pady=12,
                activebackground=COLORS["bg_medium"],
                activeforeground=COLORS["primary"],
                cursor="hand2",
                font=(FONTS["base"], 10, "bold"),
                highlightthickness=0
            )
            b.pack(side=side)
            
            # Hover effect with neon glow
            def on_enter(e):
                if b["state"] != "disabled":
                    b.configure(bg=COLORS["bg_medium"], fg=COLORS["primary"])
            
            def on_leave(e):
                if b["state"] != "disabled":
                    b.configure(bg=COLORS["bg_darker"], fg=COLORS["text_primary"])
            
            b.bind("<Enter>", on_enter)
            b.bind("<Leave>", on_leave)
            
            return b

        self.b_home = add_btn("🏠 Home", self._to_home)
        self.b_input = add_btn("📁 Input", self._back_to_input)
        
        # Right side nav
        right_nav = tk.Frame(self.nav, bg=COLORS["bg_darker"])
        right_nav.pack(side="right", fill="y")
        
        self.b_results = add_btn("📊 Results", self._to_results_direct, side="right", parent=right_nav)

    def _clear(self):
        """Clear current screen"""
        for w in self.container.winfo_children():
            w.destroy()

    def _set_nav_state(self, *, input_enabled=False, results_enabled=False):
        """Enable/disable navigation buttons based on state"""
        self.b_input.configure(
            state=("normal" if input_enabled else "disabled"),
            fg=COLORS["text_primary"] if input_enabled else COLORS["text_muted"]
        )
        self.b_results.configure(
            state=("normal" if results_enabled else "disabled"),
            fg=COLORS["text_primary"] if results_enabled else COLORS["text_muted"]
        )

    # ---- Screen Navigation -------------------------------------------------

    def _to_home(self):
        """Navigate to home screen"""
        self._clear()
        self.state.update({
            "mode": None, 
            "model_path": None, 
            "sources": None,
            "input_type": None
        })
        s = HomeScreen(self.container, on_go_next=self._after_home)
        s.pack(fill="both", expand=True)
        self._set_nav_state(
            input_enabled=False, 
            results_enabled=bool(self.state["output_dir"])
        )

    def _after_home(self, mode, model_path):
        """Called after home screen completes"""
        self.state["mode"] = mode
        self.state["model_path"] = Path(model_path)
        self._to_input()

    def _to_input(self):
        """Navigate to input selection screen"""
        self._clear()
        s = InputSelectScreen(
            self.container,
            mode=self.state["mode"],
            model_path=self.state["model_path"],
            on_process=self._start_processing,
            on_back=self._to_home
        )
        s.pack(fill="both", expand=True)
        self._set_nav_state(
            input_enabled=True, 
            results_enabled=bool(self.state["output_dir"])
        )

    def _back_to_input(self):
        """Navigate back to input screen from nav button"""
        if self.state["mode"] and self.state["model_path"]:
            self._to_input()
        else:
            self._to_home()

    def _start_processing(self, sources, input_type):
        """Start processing with selected sources"""
        self.state["sources"] = sources
        self.state["input_type"] = input_type
        
        self._clear()
        
        if input_type == "live":
            # Live camera mode - different screen
            s = LiveCameraScreen(
                self.container,
                mode=self.state["mode"],
                model_path=self.state["model_path"],
                on_done=self._after_live_camera,
                on_cancel=self._to_input
            )
        else:
            # File/folder processing
            s = ProcessingScreen(
                self.container,
                mode=self.state["mode"],
                model_path=self.state["model_path"],
                sources=self.state["sources"],
                input_type=self.state["input_type"],
                on_done=self._after_processing,
                on_cancel=self._to_input
            )
        
        s.pack(fill="both", expand=True)
        self._set_nav_state(input_enabled=False, results_enabled=False)
    
    def _after_live_camera(self):
        """After live camera ends, return to input"""
        self._to_input()

    def _after_processing(self, out_dir: Path):
        """Called after processing completes"""
        self.state["output_dir"] = out_dir
        self._to_results(out_dir)

    def _to_results(self, out_dir: Path = None):
        """Navigate to results screen"""
        self._clear()
        s = ResultsScreen(
            self.container,
            mode=self.state["mode"],
            output_dir=out_dir or self.state["output_dir"],
            on_back=self._to_input,
            on_home=self._to_home
        )
        s.pack(fill="both", expand=True)
        self._set_nav_state(input_enabled=True, results_enabled=True)

    def _to_results_direct(self):
        """Navigate to results from nav button"""
        if self.state["output_dir"]:
            self._to_results(self.state["output_dir"])

if __name__ == "__main__":
    app = App()
    # Center window on screen
    app.update_idletasks()
    x = (app.winfo_screenwidth() // 2) - (APP_W // 2)
    y = (app.winfo_screenheight() // 2) - (APP_H // 2)
    app.geometry(f"{APP_W}x{APP_H}+{x}+{y}")
    app.mainloop()