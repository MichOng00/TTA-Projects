import tkinter as tk
from tkinter import ttk

def c_to_f():
    c = float(celsius_var.get())
    f = c * 9/5 + 32
    result.set(f"{f:.2f} Fahrenheit")

root = tk.Tk()
root.title("Temperature converter")
root.resizable(False, False)

celsius_var = tk.StringVar()
result = tk.StringVar(value="Result will appear here")

ttk.Label(root, text="Celsius: ").grid(column=0, row=0, padx=10, pady=5)

c_entry = ttk.Entry(root, width=15, textvariable=celsius_var)
c_entry.grid(column=1, row=0, pady=5)

ttk.Button(root, text="to Fahrenheit", command=c_to_f).grid(column=2, row=0, padx=10)

result_label = ttk.Label(root, textvariable=result, font=("Segoe UI", 10, "bold"))
result_label.grid(column=0, row=2, columnspan=3, pady=15)

root.mainloop()