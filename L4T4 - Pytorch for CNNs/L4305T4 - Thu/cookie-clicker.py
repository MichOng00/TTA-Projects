import tkinter as tk
from tkinter import ttk

def earn_cookies():
    global cookies
    cookies += 1
    cookies_var.set(str(cookies))

root = tk.Tk()
root.title("Cookie clicker")

cookies = 0
cookies_var = tk.StringVar(value=str(cookies))

ttk.Label(root, text="cookies:").grid(row=0, column=0, pady=20, padx=20)

ttk.Label(root, textvariable=cookies_var).grid(row=1, column=0, pady=20, padx=20)

ttk.Button(root, text="click", command=earn_cookies).grid(row=2, column=0, pady=20, padx=20)


root.mainloop()