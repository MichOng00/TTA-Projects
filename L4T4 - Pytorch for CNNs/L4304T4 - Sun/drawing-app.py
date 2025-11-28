import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw, ImageOps
import torch
import numpy as np
from digit_train import DigitCNN

class DrawingApp():
    def __init__(self, root):
        root.title("Draw a digit")

        self.mainframe = ttk.Frame(root, padding="10")
        self.mainframe.grid(row=0, column=0, sticky="nsew")

        self.canvas = tk.Canvas(self.mainframe, width=280, height=280, bg="gold", relief="solid", bd=2)
        self.canvas.grid(row=0, column=0, columnspan=2)

        self.canvas.bind("<B1-Motion>", self.on_paint)

        # buttons
        self.button_clear = ttk.Button(self.mainframe, text="Clear", command=self.clear_canvas)
        self.button_clear.grid(row=1, column=1, pady=10)

        self.button_predict = ttk.Button(self.mainframe, text="Predict", command=self.predict)
        self.button_predict.grid(row=1, column=0, pady=10)

        self.label_status = ttk.Label(self.mainframe, text="draw a digit")
        self.label_status.grid(row=2, column=0, columnspan=2)

        # image
        self.image = Image.new("L", (280, 280), color=255)
        self.draw = ImageDraw.Draw(self.image)

        self.model = DigitCNN()
        self.model.load_state_dict(torch.load("digit_model.pth"))
        self.model.eval()

        ##############################################
        # EXERCISE: add an entry box to change the oval color, and a label that says what the box is for
        self.color = "black"

        self.color_label = ttk.Label(self.mainframe, text="Pen color: ")
        self.color_label.grid(row=3, column=0)

        self.color_entry = ttk.Entry(self.mainframe)
        self.color_entry.grid(row=3, column=1)

        self.color_entry.bind("<Return>", self.change_color)

    def change_color(self, _):
        self.color = self.color_entry.get()

    def on_paint(self, event):
        x, y = event.x, event.y
        r = 5

        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill=self.color, outline=self.color)
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill=0)


    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.label_status.config(text="draw a digit")

    def predict(self):
        resized = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        inverted = ImageOps.invert(resized)
        tensor = torch.tensor(np.array(inverted) / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            pred = self.model(tensor)
            digit = pred.argmax(dim=1).item()
            self.label_status.config(text=f"Prediction: {digit}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()

# amartadey.github.io/tkinter-colors/