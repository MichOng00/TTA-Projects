#!/usr/bin/python3
import pathlib
import tkinter as tk
import pygubu
from basic_ui import frameUI

cookies = 1

class frame(frameUI):
    def __init__(self, master=None):
        super().__init__(master)

    def on_button_click(self):
        global cookies
        cookies *= 2
        print(f"You have {cookies} cookies!")


if __name__ == "__main__":
    root = tk.Tk()
    app = frame(root)
    app.run()
