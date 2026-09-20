import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

def c_to_f():
    c = float(celsius_var.get())
    f = c * 9/5 + 32
    result_var.set(f"{f:.2f} Fahrenheit")

# MAIN WINDOW
root = tk.Tk()
root.title("Temperature converter")

celsius_var = tk.StringVar()
result_var = tk.StringVar()

# WIDGETS
# label
ttk.Label(root, text="Celsius").grid(column=0, row=0, padx=10, pady=5)
result_label = ttk.Label(root, width=15, textvariable=result_var, font=("Segoe UI", 10, "bold"))
result_label.grid(column=0, row=1, columnspan=3, pady=15)

# entry box
c_entry = ttk.Entry(root, width=20, textvariable=celsius_var)
c_entry.grid(column=1, row=0, pady=5)

# button
ttk.Button(root, text="➡️Fahrenheit", command=c_to_f).grid(column=2, row=0, padx=10)

# exercise: do fahrenheit to celsius conversion (another label, entry box, button, function)

root.mainloop()