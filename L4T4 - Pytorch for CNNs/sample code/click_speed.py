import tkinter as tk
from tkinter import ttk
import time

root = tk.Tk()
root.title("Click speed tester")

previous_time = None
result_var = tk.StringVar()

def store_click():
    global previous_time
    current_time = time.perf_counter()

    if previous_time is None:
        previous_time = current_time
        result_var.set("Click again to measure your click rate")
        return

    elapsed_time = current_time - previous_time
    clicks_per_second = 1 / elapsed_time
    result_var.set(f"{clicks_per_second:.2f} clicks per second")
    previous_time = current_time

ttk.Button(root, text="CLICK", command=store_click).grid(column=0, row=0, padx=10, pady=10)
result_label = ttk.Label(root, textvariable=result_var, font=("Segoe UI", 10, "bold"))
result_label.grid(column=0, row=2, columnspan=3, pady=15)

root.mainloop()