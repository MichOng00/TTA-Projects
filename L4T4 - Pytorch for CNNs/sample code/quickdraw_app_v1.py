# real-time prediction
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw, ImageOps
import torch
import numpy as np
from pathlib import Path

# Import your CNN model
from quickdraw_train import QuickDrawCNN9L

class DrawingApp:
    def __init__(self, root, class_names):
        self.root = root
        self.root.title("QuickDraw Classifier")

        # Set up frame
        self.mainframe = ttk.Frame(root, padding="10")
        self.mainframe.grid(row=0, column=0, sticky="nsew")

        # Create canvas for drawing
        self.canvas = tk.Canvas(self.mainframe, width=280, height=280, bg='gold', relief='solid', bd=2)
        self.canvas.grid(row=0, column=0, columnspan=3)

        # Create buttons and status label
        self.button_predict = ttk.Button(self.mainframe, text="Predict", command=self.predict)
        self.button_predict.grid(row=1, column=0, pady=10)

        self.button_clear = ttk.Button(self.mainframe, text="Clear", command=self.clear_canvas)
        self.button_clear.grid(row=1, column=1, pady=10)

        self.button_info = ttk.Button(self.mainframe, text="Info", command=self.show_info)
        self.button_info.grid(row=1, column=2, pady=10)

        self.label_status = ttk.Label(self.mainframe, text="Draw something!", anchor="w")
        self.label_status.grid(row=2, column=0, columnspan=3, sticky="w")

        # Create the image and drawing objects
        self.image = Image.new("L", (280, 280), color=255)  # White background
        self.draw = ImageDraw.Draw(self.image)

        # Load class names
        self.class_names = class_names

        # Load the model
        self.model = QuickDrawCNN9L(num_classes=len(self.class_names))
        self.model.load_state_dict(torch.load(Path(root_dir) / "quickdraw_model_9layer.pth", map_location='cpu'))
        self.model.eval()

        # Bind the canvas for drawing
        self.canvas.bind("<B1-Motion>", self.on_paint)

        # Start real-time prediction loop
        self.root.after(200, self.predict)

    def show_info(self):
        """Show available classes in a popup window."""
        info_win = tk.Toplevel()
        info_win.title("Available Classes")
        info_text = tk.Text(info_win, width=40, height=20)
        info_text.grid(padx=10, pady=10)
        info_text.insert("end", "\n".join(self.class_names))
        info_text.config(state="disabled")

    def on_paint(self, event):
        x, y = event.x, event.y
        r = 5
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='black', outline='black')
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill=0)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.label_status.config(text="Draw something!")

    def predict(self):
        resized = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        inverted = ImageOps.invert(resized)
        tensor = torch.tensor(np.array(inverted) / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            pred = self.model(tensor)
            idx = pred.argmax(dim=1).item()
            class_name = self.class_names[idx]
            self.label_status.config(text=f"Prediction: {class_name}")
        self.root.after(200, self.predict)

if __name__ == "__main__":
    root = tk.Tk()
    root_dir = "../QuickDraw"
    with open(Path(root_dir) / "class_names.txt", "r") as f:
        class_names = [line.strip() for line in f if line.strip()]
    app = DrawingApp(root, class_names)

    root.mainloop()