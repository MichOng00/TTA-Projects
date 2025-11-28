import tkinter as tk
from tkinter import ttk

def earn_cookies():
    global cookies
    # cookies = cookies + 500
    cookies += 500
    cookies_var.set(str(cookies))

def cheat():
    global cookies
    code = cheat_entry.get().strip().lower()
    if code == "morecookies":
        cookies += 1000000000000000000000000
        cookies_var.set(str(cookies))
        cheat_entry.delete(0, tk.END)

root = tk.Tk()
root.title("Cookie clicker")

cookies = 0
cookies_var = tk.StringVar(value=str(cookies))

ttk.Label(root, text="cookies:").grid(row=0, column=0, pady=20, padx=20)

ttk.Label(root, textvariable=cookies_var).grid(row=1, column=0, pady=20, padx=20)

ttk.Button(root, text="click here!", command=earn_cookies).grid(row=2, column=0, pady=20, padx=20)

# cheat code
ttk.Label(root, text="enter code").grid(row=3, column=0, pady=20, padx=20)
cheat_entry = ttk.Entry(root)
cheat_entry.grid(row=4, column=0, pady=10, padx=20)
ttk.Button(root, text="Apply", command=cheat).grid(row=5, column=0, pady=20, padx=20)

root.mainloop()