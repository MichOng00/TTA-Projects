import tkinter as tk
import pygubu
from PIL import Image, ImageDraw, ImageOps
import torch
import numpy as np
from session_3 import DigitCNN

class DrawingApp:
    def __init__(self, root):
        self.builder = pygubu.Builder()
        self.builder.add_from_file('drawing_ui.ui')
        self.mainframe = self.builder.get_object('frame', root)
        root.title("Draw a Digit waluigi")
        
        self.canvas = self.builder.get_object('canvas')
        self.button_predict = self.builder.get_object('button_predict')
        self.button_clear = self.builder.get_object('button_clear')
        self.label_status = self.builder.get_object('label_status')

        self.image = Image.new("L", (280, 280), color=255)
        self.draw = ImageDraw.Draw(self.image)

        self.model = DigitCNN()
        self.model.load_state_dict(
            torch.load("digit_model.pth", map_location='cpu')
        )
        self.model.eval()

    def on_paint(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='black', outline='black')
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill=0)