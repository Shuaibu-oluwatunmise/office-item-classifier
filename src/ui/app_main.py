import tkinter as tk
from pathlib import Path

from components.styles import COLORS, FONTS
from components.widgets import header
from screens.home import HomeScreen
from screens.input_select import InputSelectScreen
from screens.processing import ProcessingScreen
from screens.results import ResultsScreen

APP_W, APP_H = 1100, 750

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Office Item Classifier")
        self.configure(bg=COLORS["light"])
        self.geometry(f"{APP_W}x{APP_H}")
        self.minsize(1000, 650)

        # Modern navigation bar with better styling
        self.nav = tk.Frame(self, bg=COLORS["dark"], height=50)
        self.nav.pack(fill="x")
        self.nav.pack_propagate(False)
        
        # Add subtle bottom border to nav
        nav_border = tk.Frame(self, bg=COLORS["primary"], height=2)
        nav_border.pack(fill="x")
        
        self._make_nav()

        # Main container with subtle background
        self.container = tk.Frame(self, bg=COLORS["light"])
        self.container.pack(fill="both", expand=True)

        self.state = {
            "mode": None,
            "model_path": None,
            "sources": None,
            "output_dir": None
        }

        self._to_home()

    # ---- Navigation Bar ----------------------------------------------------

    def _make_nav(self):
        """Create a modern navigation bar with better styling"""
        
        # Left side - branding and main nav
        left_nav = tk.Frame(self.nav, bg=COLORS["dark"])
        left_nav.pack(side="left", fill="y")
        
        # App title/logo
        logo_label = tk.Label(
            left_nav,
            text="📊 Classifier",
            bg=COLORS["dark"],
            fg="white",
            font=(FONTS["heading"], 12, "bold"),
            padx=16
        )
        logo_label.pack(side="left", pady=10)
        
        # Separator
        sep = tk.Frame(left_nav, bg=COLORS["border"], width=1)
        sep.pack(side="left", fill="y", padx=8, pady=8)
        
        def add_btn(text, cmd, side="left", parent=None):
            target = parent if parent else left_nav
            b = tk.Button(
                target, text=text, command=cmd, 
                bg=COLORS["dark"], fg="white", 
                bd=0, relief="flat", 
                padx=16, pady=10,
                activebackground=COLORS["primary"], 
                cursor="hand2",
                font=(FONTS["base"], 10, "bold"),
                highlightthickness=0
            )
            b.pack(side=side)
            
            # Hover effect
            def on_enter(e):
                if b["state"] != "disabled":
                    b.configure(bg=COLORS["primary"])
            
            def on_leave(e):
                if b["state"] != "disabled":
                    b.configure(bg=COLORS["dark"])
            
            b.bind("<Enter>", on_enter)
            b.bind("<Leave>", on_leave)
            
            return b

        self.b_home = add_btn("🏠 Home", self._to_home)
        self.b_input = add_btn("📁 Input", self._back_to_input)
        
        # Right side nav
        right_nav = tk.Frame(self.nav, bg=COLORS["dark"])
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
            fg="white" if input_enabled else COLORS["text_secondary"]
        )
        self.b_results.configure(
            state=("normal" if results_enabled else "disabled"),
            fg="white" if results_enabled else COLORS["text_secondary"]
        )

    # ---- Screen Navigation -------------------------------------------------

    def _to_home(self):
        """Navigate to home screen"""
        self._clear()
        self.state.update({
            "mode": None, 
            "model_path": None, 
            "sources": None
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

    def _start_processing(self, sources):
        """Start processing with selected sources"""
        self.state["sources"] = sources
        self._clear()
        s = ProcessingScreen(
            self.container,
            mode=self.state["mode"],
            model_path=self.state["model_path"],
            sources=self.state["sources"],
            on_done=self._after_processing,
            on_cancel=self._to_input
        )
        s.pack(fill="both", expand=True)
        self._set_nav_state(input_enabled=False, results_enabled=False)

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