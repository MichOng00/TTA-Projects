import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw

class DrawingApp():
    IMAGE_SIZE = 500
    def __init__(self, root):
        root.title("Draw a digit")

        # Set up frame
        self.mainframe = ttk.Frame(root, padding="10")
        self.mainframe.grid(row=0, column=0, sticky="nsew")

        # Create canvas
        self.canvas = tk.Canvas(self.mainframe, width=self.IMAGE_SIZE, height=self.IMAGE_SIZE, bg="forest green", relief="solid", bd=2)
        self.canvas.grid(row=0, column=0, columnspan=2)

        # Bind canvas to left mouse button
        self.canvas.bind("<B1-Motion>", self.on_paint)

        # Create buttons
        self.button_clear = ttk.Button(self.mainframe, text="Clear", command=self.clear_canvas)
        self.button_clear.grid(row=1, column=1, pady=10)

        self.button_predict = ttk.Button(self.mainframe, text="Predict", command=self.predict)
        self.button_predict.grid(row=1, column=0, pady=10)

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

    def change_color(self, event):
        self.color = self.color_entry.get()

    def change_width(self, event):
        self.pen_width = int(self.width_entry.get())

    def on_paint(self, event):
        x, y = event.x, event.y
        r = self.pen_width # radius of the oval
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill=self.color, outline=self.color)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.IMAGE_SIZE, self.IMAGE_SIZE), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.status_label.config(text="Draw a digit") # remove previous prediction

    def predict(self):
        pass


if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()