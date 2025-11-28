import tkinter as tk
from tkinter import ttk, messagebox

root = tk.Tk()
root.title("Cookie clicker")

cookies = 0
cookie_var = tk.StringVar(value=cookies)

upgrade_cost = 10
cookies_per_click = 1

def earn_cookie():
    global cookies
    # cookies = cookies + 1
    cookies += cookies_per_click
    cookie_var.set(cookies)

def buy_upgrade():
    global cookies_per_click, upgrade_cost, cookies
    if cookies >= upgrade_cost:
        cookies -= upgrade_cost
        cookies_per_click *= 2
        upgrade_cost *= 2
        upgrade_button.config(text=f"Upgrade cost: {upgrade_cost}")
        cookie_var.set(cookies)
        

ttk.Label(root, text="Cookies:", font=("Helvetica", 24)).grid(row=0, column=0, padx=20, pady=20)
ttk.Label(root, textvariable=cookie_var, font=("Helvetica", 24)).grid(row=1, column=0, padx=20, pady=20)

ttk.Button(root, text="Click to earn cookie", command=earn_cookie, width=20).grid(row=2, column=0, padx=20, pady=20)
upgrade_button = ttk.Button(root, text=f"Upgrade cost: {upgrade_cost}", command=buy_upgrade, width=20)
upgrade_button.grid(row=3, column=0)

root.mainloop()