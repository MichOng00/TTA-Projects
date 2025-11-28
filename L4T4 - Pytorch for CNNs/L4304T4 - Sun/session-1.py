import tkinter as tk
from tkinter import ttk 
from tkinter import messagebox

def c_to_f():
    try:
        c = float(celsius_var.get())
        f = c * 9/5 + 32
        result_var.set(f"{f:.2f} Fahrenheit")
    except ValueError:
        messagebox.showerror("Invalid input", "Please enter a valid number")

root = tk.Tk()
root.title("Temperature converter")

celsius_var = tk.StringVar()
result_var = tk.StringVar(value="Result will appear here")

ttk.Label(root, text="Celsius").grid(column=0, row=0, padx=10, pady=5, sticky="E")
c_entry = ttk.Entry(root, width=15, textvariable=celsius_var)
c_entry.grid(column=1, row=0, pady=5)

ttk.Button(root, text="to Fahrenheit", command=c_to_f).grid(column=2, row=0, padx=10)

result_label = ttk.Label(root, textvariable=result_var, font=("Segoe UI", 10, "bold"))
result_label.grid(column=0, row=2, columnspan=3, pady=15)

root.mainloop()

# exercise: add fahrenheit to celsius
# exercise: cookie clicker (separate file)