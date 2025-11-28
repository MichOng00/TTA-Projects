import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw, ImageOps
import torch
import numpy as np
from digit_train import DigitCNN

class DrawingApp:
    def __init__(self, root):
        root.title("Draw a Digit")

        # Set up frame
        self.mainframe = ttk.Frame(root, padding="10")
        self.mainframe.grid(row=0, column=0, sticky="nsew")

        # Create canvas for drawing
        self.canvas = tk.Canvas(self.mainframe, width=280, height=280, bg='gold', relief='solid', bd=2)
        self.canvas.grid(row=0, column=0, columnspan=2)

        # Create buttons and status label
        self.button_predict = ttk.Button(self.mainframe, text="Predict", command=self.predict)
        self.button_predict.grid(row=1, column=0, pady=10)

        self.button_clear = ttk.Button(self.mainframe, text="Clear", command=self.clear_canvas)
        self.button_clear.grid(row=1, column=1, pady=10)

        self.label_status = ttk.Label(self.mainframe, text="Draw a digit", anchor="w")
        self.label_status.grid(row=2, column=0, columnspan=2, sticky="w")

        # Create the image and drawing objects
        self.image = Image.new("L", (280, 280), color=255)  # White background
        self.draw = ImageDraw.Draw(self.image)

        # Load the model
        self.model = DigitCNN()
        self.model.load_state_dict(torch.load("digit_model.pth", map_location='cpu'))
        self.model.eval()

        # Bind the canvas for drawing
        self.canvas.bind("<B1-Motion>", self.on_paint)

    def on_paint(self, event):
        """Handles the painting/drawing on canvas."""
        x, y = event.x, event.y
        r = 5
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='black', outline='black')
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill=0)

    def clear_canvas(self):
        """Clears the canvas and resets the image."""
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), color=255)  # White background
        self.draw = ImageDraw.Draw(self.image)
        self.label_status.config(text="Draw a digit")

    def predict(self):
        """Predict the drawn digit using the model."""
        # Preprocess the image and predict the digit
        resized = self.image.resize((28, 28), Image.LANCZOS)
        inverted = ImageOps.invert(resized)
        tensor = torch.tensor(np.array(inverted) / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Predict with no gradient computation
        with torch.no_grad():
            pred = self.model(tensor)
            digit = pred.argmax(dim=1).item()
            self.label_status.config(text=f"Prediction: {digit}")


if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
