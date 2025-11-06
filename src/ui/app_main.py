import tkinter as tk
from pathlib import Path

from components.styles import COLORS, FONTS
from components.widgets import header
from screens.home import HomeScreen
from screens.input_select import InputSelectScreen
from screens.processing import ProcessingScreen
from screens.results import ResultsScreen

APP_W, APP_H = 1000, 700

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Office Item Classifier / Detector")
        self.configure(bg=COLORS["light"])
        self.geometry(f"{APP_W}x{APP_H}")
        self.minsize(900, 600)

        # Top nav
        self.nav = tk.Frame(self, bg=COLORS["dark"])
        self.nav.pack(fill="x")
        self._make_nav()

        # Container
        self.container = tk.Frame(self, bg=COLORS["light"])
        self.container.pack(fill="both", expand=True)

        self.state = {
            "mode": None,
            "model_path": None,
            "sources": None,
            "output_dir": None
        }

        self._to_home()

    # ---- Nav ---------------------------------------------------------------

    def _make_nav(self):
        def add_btn(text, cmd, side="left"):
            b = tk.Button(self.nav, text=text, command=cmd, bg=COLORS["dark"],
                          fg="white", bd=0, relief="flat", padx=10, pady=10,
                          activebackground=COLORS["primary"], cursor="hand2",
                          font=(FONTS["base"], 10, "bold"))
            b.pack(side=side, padx=4, pady=2)
            return b

        self.b_home = add_btn("Home", self._to_home)
        self.b_input = add_btn("Input", self._back_to_input)
        self.b_results = add_btn("Results", self._to_results_direct, side="right")

    def _clear(self):
        for w in self.container.winfo_children():
            w.destroy()

    def _set_nav_state(self, *, input_enabled=False, results_enabled=False):
        self.b_input.configure(state=("normal" if input_enabled else "disabled"))
        self.b_results.configure(state=("normal" if results_enabled else "disabled"))

    # ---- Screens -----------------------------------------------------------

    def _to_home(self):
        self._clear()
        self.state.update({"mode": None, "model_path": None, "sources": None, "output_dir": None})
        s = HomeScreen(self.container, on_go_next=self._after_home)
        s.pack(fill="both", expand=True)
        self._set_nav_state(input_enabled=False, results_enabled=bool(self.state["output_dir"]))

    def _after_home(self, mode, model_path):
        self.state["mode"] = mode
        self.state["model_path"] = Path(model_path)
        self._to_input()

    def _to_input(self):
        self._clear()
        s = InputSelectScreen(
            self.container,
            mode=self.state["mode"],
            model_path=self.state["model_path"],
            on_process=self._start_processing,
            on_back=self._to_home
        )
        s.pack(fill="both", expand=True)
        self._set_nav_state(input_enabled=True, results_enabled=bool(self.state["output_dir"]))

    def _back_to_input(self):
        if self.state["mode"] and self.state["model_path"]:
            self._to_input()
        else:
            self._to_home()

    def _start_processing(self, sources):
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
        self.state["output_dir"] = out_dir
        self._to_results(out_dir)

    def _to_results(self, out_dir: Path = None):
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
        if self.state["output_dir"]:
            self._to_results(self.state["output_dir"])

if __name__ == "__main__":
    App().mainloop()
