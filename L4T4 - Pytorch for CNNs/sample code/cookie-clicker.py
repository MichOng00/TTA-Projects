import tkinter as tk
from tkinter import messagebox, ttk

# Initialize the main window
root = tk.Tk()
root.title("Cookie Clicker")

# Game variables
cookies = 0
cookies_per_click = 1
upgrade_cost = 10

# Functions to update the game state
def earn_cookie():
    global cookies
    cookies += cookies_per_click
    cookie_count_var.set(cookies)

def buy_upgrade():
    global cookies, cookies_per_click, upgrade_cost
    if cookies >= upgrade_cost:
        cookies -= upgrade_cost
        cookies_per_click += 1
        upgrade_cost *= 2
        upgrade_button.config(text=f"Upgrade (Cost: {upgrade_cost})")
        cookie_count_var.set(cookies)
    else:
        messagebox.showerror("Not enough cookies!", "Please get more cookies first.")

# Widgets
cookie_count_var = tk.StringVar(value=cookies)
ttk.Label(root, text="Cookies: ", font=("Helvetica", 24)).grid(row=0, column=0, pady=20, padx=20)
ttk.Label(root, textvariable=cookie_count_var, font=("Helvetica", 24)).grid(row=1, column=0, pady=20)
ttk.Button(root, text="Click to Earn Cookie", command=earn_cookie, width=20).grid(row=2, column=0, pady=20)
upgrade_button = ttk.Button(root, text=f"Upgrade (Cost: {upgrade_cost})", command=buy_upgrade, width=20)
upgrade_button.grid(row=3, column=0, pady=20)

# Start the application
root.mainloop()
