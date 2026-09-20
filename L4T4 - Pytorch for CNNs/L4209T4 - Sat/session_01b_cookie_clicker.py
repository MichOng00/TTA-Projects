import tkinter as tk
from tkinter import ttk

root = tk.Tk()
root.title("Cookie clicker")

cookies = 0
cookies_var = tk.IntVar(value=0)
cookies_per_click = 1
upgrade_cost = 10

def earn_cookies():
    global cookies
    cookies += cookies_per_click
    print(cookies)
    cookies_var.set(cookies)

def upgrade():
    global cookies, cookies_per_click, upgrade_cost
    if cookies >= upgrade_cost: # do I have enough cookies?
        cookies -= upgrade_cost
        cookies_per_click *= 2
        upgrade_cost *= 2
        upgrade_button.config(text=f"Upgrade (Cost: {upgrade_cost})")
        cookies_var.set(cookies)

ttk.Label(root, text="Cookies:").grid(row=0, column=0, padx=20, pady=20)
ttk.Label(root, textvariable=cookies_var).grid(row=1, column=0, padx=20, pady=20)
ttk.Button(root, text="CLICK", command=earn_cookies).grid(row=2, column=0, padx=20, pady=20)

# exercise: upgrade button (use multiplier variable)
upgrade_button = ttk.Button(root, text=f"Upgrade (Cost: {upgrade_cost})", command=upgrade)
upgrade_button.grid(row=3, column=0, padx=20, pady=20)

root.mainloop()