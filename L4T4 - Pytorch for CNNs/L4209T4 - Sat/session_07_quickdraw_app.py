# note: 
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw, ImageOps
import torch
import numpy as np
from pathlib import Path
from quickdraw_train import QuickDrawCNN9L
import random

class DrawingApp():
    IMAGE_SIZE = 500
    def __init__(self, root, class_names):
        root.title("Draw a digit")

        # Set up frame
        self.mainframe = ttk.Frame(root, padding="10")
        self.mainframe.grid(row=0, column=0, sticky="nsew")

        # Create canvas
        self.canvas = tk.Canvas(self.mainframe, width=self.IMAGE_SIZE, height=self.IMAGE_SIZE, bg="white", relief="solid", bd=2)
        self.canvas.grid(row=0, column=0, columnspan=2)

        # Bind canvas to left mouse button
        self.canvas.bind("<B1-Motion>", self.on_paint)

        # Create buttons
        self.button_clear = ttk.Button(self.mainframe, text="Clear", command=self.clear_canvas)
        self.button_clear.grid(row=1, column=1, pady=10)

        self.button_predict = ttk.Button(self.mainframe, text="Predict", command=self.predict)
        self.button_predict.grid(row=1, column=0, pady=10)

        self.button_start = ttk.Button(self.mainframe, text="Start", command=self.start_round)
        self.button_start.grid(row=1, column=2, pady=10)

        self.status_label = ttk.Label(self.mainframe, text="Draw a digit", anchor="w")
        self.status_label.grid(row=2, column=0, columnspan=2, sticky="w")

        self.color = "black"

        self.color_label = ttk.Label(self.mainframe, text="Pen color:")
        self.color_label.grid(row=3, column=0, sticky="w")

        self.color_entry = ttk.Entry(self.mainframe)
        self.color_entry.grid(row=3, column=1)
        self.color_entry.insert(0, self.color)
        self.color_entry.bind("<Return>", self.change_color)

        self.pen_width = 3

        self.width_label = ttk.Label(self.mainframe, text="Pen width:")
        self.width_label.grid(row=4, column=0, sticky="w")

        self.width_entry = ttk.Entry(self.mainframe)
        self.width_entry.grid(row=4, column=1)
        self.width_entry.insert(0, self.pen_width)
        self.width_entry.bind("<Return>", self.change_width)

        self.image = Image.new("L", (self.IMAGE_SIZE, self.IMAGE_SIZE), color=255)
        self.draw = ImageDraw.Draw(self.image)

        self.class_names = class_names

        # model
        self.model = QuickDrawCNN9L(num_classes = len(self.class_names))
        self.model.load_state_dict(torch.load(Path(root_dir) / "quickdraw_9layer.pth"))
        # self.model.load_state_dict(torch.load("quickdraw_9layer.pth"))
        self.model.eval()
        # tinyurl.com/43y3m746

        root.after(200, self.predict)

        self.current_target = None
        self.time_left = 0
        self.timer_running = False

    def change_color(self, event):
        self.color = self.color_entry.get()

    def change_width(self, event):
        self.pen_width = int(self.width_entry.get())

    def on_paint(self, event):
        x, y = event.x, event.y
        r = self.pen_width # radius of the oval
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill=self.color, outline=self.color)
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill=0) # create ellipse on PIL image

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.IMAGE_SIZE, self.IMAGE_SIZE), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.status_label.config(text="Draw a digit") # remove previous prediction

    def start_round(self):
        self.clear_canvas()
        self.current_target = random.choice(self.class_names)
        self.status_label.config(text = f"Draw: {self.current_target}")
        self.time_left = 20
        self.timer_running = True
        self.update_timer()
        root.after(200, self.predict)

    def update_timer(self):
        if self.time_left > 0 and self.timer_running:
            self.label_timer.config(text=f"Time left: {self.time_left}s")
            self.time_left -= 1
            root.after(1000, self.update_timer) # call itself again after 1 second
        elif self.timer_running:
            self.label_timer.config(text="Time's up!")
            self.timer_running = False

    def predict(self):
        if not self.timer_running:
            return
        resized = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        inverted = ImageOps.invert(resized)
        tensor = torch.tensor(np.array(inverted) / 255, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            pred = self.model(tensor)
            # idx = pred.argmax(dim=1).item() # index (0-9)
            # class_name = self.class_names[idx]
            probs = torch.softmax(pred, dim=1)
            top2 = torch.topk(probs, 2)
            idx1 = top2.indices[0][0].item()
            idx2 = top2.indices[0][1].item()
            conf1 = top2.values[0][0].item()
            conf2 = top2.values[0][1].item()
            class1 = self.class_names[idx1]
            class2 = self.class_names[idx2]

            self.status_label.config(text=f"Target: {self.current_target}\nPrediction: {class1} {conf1:.2%}\n2nd guess: {class2} {conf2:.2%}")

            if self.current_target == class1 and conf1 >= 0.4:
                self.label_timer.config(text="I recognised it!")
                self.timer_running = False
                return

        root.after(200, self.predict)


if __name__ == "__main__":
    root = tk.Tk()
    root_dir = "../QuickDraw"
    with open(Path(root_dir) / "class_names.txt", "r") as f:
        class_names = [line.strip() for line in f if line.strip()]
    app = DrawingApp(root, class_names)
    root.mainloop()

# tinyurl.com/msxkswd