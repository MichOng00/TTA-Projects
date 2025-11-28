import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw

class DrawingApp():
    def __init__(self, root):
        root.title("Draw a Digit")

        # Set up frame
        self.mainframe = ttk.Frame(root, padding="10")
        self.mainframe.grid(row=0, column=0, sticky="nsew")

        # Create canvas
        self.canvas = tk.Canvas(self.mainframe, width=280, height=280, bg="gold", relief="solid", bd=2)
        self.canvas.grid(row=0, column=0, columnspan=2)

        # Bind canvas to left mouse button
        self.canvas.bind("<B1-Motion>", self.on_paint)

        # Create buttons
        self.button_clear = ttk.Button(self.mainframe, text="Clear", command=self.clear_canvas)
        self.button_clear.grid(row=1, column=1, pady=10)

        self.button_predict = ttk.Button(self.mainframe, text="Predict", command=self.predict)
        self.button_predict.grid(row=1, column=0, pady=10)

        # Create label
        self.label_status = ttk.Label(self.mainframe, text="Draw a digit", anchor="w")
        self.label_status.grid(row=2, column=0, columnspan=2, sticky="w")

        ########################################
        # Default color for drawing
        self.color = "black"
        # Create entry box to change oval color
        self.color_label = ttk.Label(self.mainframe, text="Pen Color:")
        self.color_label.grid(row=3, column=0, sticky="w")

        self.color_entry = ttk.Entry(self.mainframe)
        self.color_entry.grid(row=3, column=1)
        # self.color_entry.insert(0, "black")  # Default color

        # Bind the Enter key to update the pen color
        self.color_entry.bind("<Return>", self.change_color)
        #######################################

        # Initialize image for drawing
        self.image = Image.new("L", (280, 280), color=255) 
        self.draw = ImageDraw.Draw(self.image)

    def change_color(self, event):
        """Change pen color when Enter is pressed."""
        # Get the color from the entry box
        self.color = self.color_entry.get()


    def on_paint(self, event):
        """Handles painting/drawing on canvas."""
        x, y = event.x, event.y
        r = 5  # radius of the oval

        # Draw the oval on the canvas with the current pen color
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill=self.color, outline=self.color)

    def clear_canvas(self):
        """Clears the canvas and resets the image."""
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), color=255) 
        self.draw = ImageDraw.Draw(self.image)
        self.label_status.config(text="Draw a digit")

    def predict(self):
        """Placeholder for prediction functionality."""
        pass

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()

# amartadey.github.io/tkinter-colors/