import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

def c_to_f():
    """Convert Celsius to Fahrenheit."""
    try:
        c = float(celsius_var.get())
        f = c * 9/5 + 32
        result_var.set(f"{f:.2f} °F")
    except ValueError:
        messagebox.showerror("Invalid input", "Please enter a valid number.")

# def f_to_c():
#     """Convert Fahrenheit to Celsius."""
#     try:
#         f = float(fahrenheit_var.get())
#         c = (f - 32) * 5/9
#         result_var.set(f"{c:.2f} °C")
#     except ValueError:
#         messagebox.showerror("Invalid input", "Please enter a valid number.")

# Main window 
root = tk.Tk()
root.title("Temperature Converter")
root.resizable(False, False)  # fixed size

# Variables 
celsius_var     = tk.StringVar()
fahrenheit_var  = tk.StringVar()
result_var      = tk.StringVar(value="Result will appear here")

# Widgets
# Labels and entry fields
ttk.Label(root, text="Celsius (°C):").grid(column=0, row=0, padx=10, pady=5, sticky="E")
c_entry = ttk.Entry(root, width=15, textvariable=celsius_var)
c_entry.grid(column=1, row=0, pady=5)

# ttk.Label(root, text="Fahrenheit (°F):").grid(column=0, row=1, padx=10, pady=5, sticky="E")
# f_entry = ttk.Entry(root, width=15, textvariable=fahrenheit_var)
# f_entry.grid(column=1, row=1, pady=5)

# Buttons
ttk.Button(root, text="→ Fahrenheit", command=c_to_f).grid(column=2, row=0, padx=10)
# ttk.Button(root, text="→ Celsius",   command=f_to_c).grid(column=2, row=1, padx=10)

# Result display
result_label = ttk.Label(root, textvariable=result_var, font=("Segoe UI", 10, "bold"))
result_label.grid(column=0, row=2, columnspan=3, pady=15)

c_entry.focus()

# Start the GUI loop
root.mainloop()