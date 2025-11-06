# src/ui/app_main.py
import tkinter as tk
from tkinter import ttk
from components.widgets import configure_root
from screens.home import HomeScreen
from screens.input_select import InputSelectScreen
from screens.processing import ProcessingScreen
from screens.results import ResultsScreen

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        configure_root(self)
        self._stack = []

        self.container = ttk.Frame(self, style="TFrame")
        self.container.pack(fill="both", expand=True)

        self._show_home()

    # Screen factory helpers
    def _clear(self):
        for w in self.container.winfo_children():
            w.destroy()

    def _show_home(self):
        self._clear()
        s = HomeScreen(self.container, go_next=self._to_input_select)
        s.pack(fill="both", expand=True)
        self._stack = []

    def _to_input_select(self, mode):
        self._clear()
        s = InputSelectScreen(self.container, mode=mode,
                              go_back=self._show_home,
                              go_process=self._to_processing)
        s.pack(fill="both", expand=True)
        self._stack.append(("home", {}))

    def _to_processing(self, mode, input_path):
        self._clear()
        s = ProcessingScreen(self.container, mode=mode, input_path=input_path,
                             go_back=self._show_home,
                             go_results=self._to_results)
        s.pack(fill="both", expand=True)

    def _to_results(self, mode, output_dir, files):
        self._clear()
        s = ResultsScreen(self.container, mode=mode, output_dir=output_dir, files=files,
                          go_home=self._show_home)
        s.pack(fill="both", expand=True)

if __name__ == "__main__":
    App().mainloop()
