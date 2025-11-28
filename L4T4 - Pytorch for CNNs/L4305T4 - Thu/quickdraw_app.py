# tinyurl.com/4z7jtur9
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw, ImageOps
import torch
import numpy as np
from pathlib import Path
import random

from quickdraw_train import QuickDrawCNN9L

class DrawingApp():
    def __init__(self, root, class_names):
        self.root = root
        self.root.title("Quick Draw")

        # set up frame
        self.mainframe = ttk.Frame(root, padding = "10")
        self.mainframe.grid(row=0, column=0, sticky="nsew")

        # create canvas
        self.canvas = tk.Canvas(self.mainframe, width=280, height=280, bg="gold", relief="solid", bd=2)
        self.canvas.grid(row=0, column=0, columnspan=4)

        # bind canvas to left mouse button
        self.canvas.bind("<B1-Motion>", self.on_paint)

        # create buttons
        self.button_clear = ttk.Button(self.mainframe, text="Clear", command=self.clear_canvas)
        self.button_clear.grid(row=1, column=1, pady=10)

        self.button_predict = ttk.Button(self.mainframe, text="Predict", command=self.show_prediction)
        self.button_predict.grid(row=1, column=0, pady=10)

        self.button_info = ttk.Button(self.mainframe, text="Info", command=self.show_info)
        self.button_info.grid(row=1, column=2, pady=10)

        self.button_start = ttk.Button(self.mainframe, text="Start round", command=self.start_round)
        self.button_start.grid(row=1, column=3, pady=10)

        # create label
        self.label_status = ttk.Label(self.mainframe, text="draw a digit", anchor="w")
        self.label_status.grid(row=2, column=0, columnspan=4, sticky="w")

        self.label_timer = ttk.Label(self.mainframe, text = "", anchor="w")
        self.label_timer.grid(row=4, column=0, columnspan=4, sticky="w")

        # change color
        self.color_label = ttk.Label(self.mainframe, text="Pen Color:")
        self.color_label.grid(row=3, column=0, sticky="w")

        self.color_entry = ttk.Entry(self.mainframe)
        self.color_entry.grid(row=3, column=1)
        self.color_entry.insert(0, "black")

        self.color_entry.bind("<Return>", self.change_color)

        self.image = Image.new("L", (280, 280), color=255) 
        self.draw = ImageDraw.Draw(self.image)

        self.color = "black"
        self.class_names = class_names

        self.model = QuickDrawCNN9L(num_classes=len(self.class_names))
        self.model.load_state_dict(torch.load(Path(root_dir) / "quickdraw_model.pth", map_location="cpu"))
        self.model.eval()

        self.current_target = None
        self.time_left = 0
        self.timer_running = False

    def on_paint(self, event):
        x, y = event.x, event.y
        r = 5 # radius
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill=self.color, outline=self.color)
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill=0)

    def show_info(self):
        info_win = tk.Toplevel()
        info_win.title("Available classes")
        info_text = tk.Text(info_win, width=40, height=20)
        info_text.grid(padx=10, pady=10)
        info_text.insert("end", "\n".join(self.class_names))

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), color=255) 
        self.draw = ImageDraw.Draw(self.image)
        self.label_status.config(text="Draw something!")

    def start_round(self):
        self.clear_canvas()
        self.current_target = random.choice(self.class_names)
        self.label_status.config(text=f"Draw: {self.current_target}")
        self.time_left = 20
        self.timer_running = True
        self.update_timer()
        self.root.after(200, self.check_prediction)

    def update_timer(self):
        if self.time_left > 0 and self.timer_running:
            self.label_timer.config(text=f"Time left: {self.time_left}s")
            self.time_left -= 1
            self.root.after(1000, self.update_timer)
        elif self.timer_running:
            self.label_timer.config(text="Time's up!")
            self.timer_running = False
            self.show_prediction()

    def check_prediction(self):
        if not self.timer_running:
            return
        resized = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        inverted = ImageOps.invert(resized)
        tensor = torch.tensor(np.array(inverted) / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            pred = self.model(tensor)
            probs = torch.softmax(pred, dim=1)
            top2 = torch.topk(probs, 2)
            idx1 = top2.indices[0][0].item()
            idx2 = top2.indices[0][1].item()
            conf1 = top2.values[0][0].item()
            conf2 = top2.values[0][1].item()
            class1 = self.class_names[idx1]
            class2 = self.class_names[idx2]
            self.label_status.config(text=f"Target: {self.current_target}\nPrediction: {class1} ({conf1:.2%})\nSecond guess: {class2} ({conf2:.2%})")
            
            if class1 == self.current_target and conf1 >= 0.6:
                self.label_timer.config(text="I recognised it!")
                self.timer_running = False
                return
        self.root.after(200, self.check_prediction)

    def show_prediction(self):
        resized = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        inverted = ImageOps.invert(resized)
        tensor = torch.tensor(np.array(inverted) / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            pred = self.model(tensor)
            probs = torch.softmax(pred, dim=1)
            top2 = torch.topk(probs, 2)
            idx1 = top2.indices[0][0].item()
            idx2 = top2.indices[0][1].item()
            conf1 = top2.values[0][0].item()
            conf2 = top2.values[0][1].item()
            class1 = self.class_names[idx1]
            class2 = self.class_names[idx2]
            self.label_status.config(text=f"Target: {self.current_target}\nPrediction: {class1} ({conf1:.2%})\nSecond guess: {class2} ({conf2:.2%})")

    def change_color(self, _):
        self.color = self.color_entry.get()

if __name__ == "__main__":
    root = tk.Tk()
    root_dir = "../QuickDraw"
    with open(Path(root_dir) / "class_names.txt", "r") as f:
        class_names = [line.strip() for line in f if line.strip()]

        # class_names = []
        # for line in f:
        #     if line.strip():
        #         class_names.append(line.strip())
    app = DrawingApp(root, class_names)
    root.mainloop()

# tinyurl.com/yy9fj5yx