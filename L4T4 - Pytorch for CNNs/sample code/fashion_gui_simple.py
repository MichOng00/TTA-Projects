import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import torch
import torchvision.transforms as transforms
from fashion_train import FashionCNN, class_names


class FashionApp(ttk.Frame):
    def __init__(self, root):
        super().__init__(root, padding=10)
        self.root = root
        self.root.title("FashionMNIST Predictor")
        self.root.geometry("650x450")

        # Load pretrained model
        self.model = FashionCNN()
        self.model.load_state_dict(torch.load("fashion_model.pth"))
        self.model.eval()

        self.original_image = None
        self.tk_image = None

        self._create_widgets()
        self._layout_widgets()
        self.btn_predict.config(state=tk.DISABLED)

    def _create_widgets(self):
        """Create all widgets."""
        # Left control panel
        self.left_frame = ttk.Frame(self, padding=5)

        self.btn_upload = ttk.Button(self.left_frame, text="Upload Image", command=self.upload_image)
        self.btn_predict = ttk.Button(self.left_frame, text="Predict", command=self.predict_image)

        self.lbl_preview = ttk.Label(self.left_frame, text="Preview (28x28):")
        self.preview = tk.Canvas(self.left_frame, width=84, height=84, bg="#eee")

        self.lbl_results = ttk.Label(self.left_frame, text="Top Predictions:")
        self.results = ttk.Treeview(self.left_frame, columns=("Class", "Conf"), show="headings", height=4)
        self.results.heading("Class", text="Class")
        self.results.heading("Conf", text="Confidence")
        self.results.column("Class", width=100)
        self.results.column("Conf", width=80, anchor="e")

        # Right image display area
        self.canvas = tk.Canvas(self, bg="lightgray")
        self.label = ttk.Label(self.canvas, text="Upload an image")

    def _layout_widgets(self):
        """Use grid layout for a modern look."""
        # Grid for main frame
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # Left control panel
        self.left_frame.grid(row=0, column=0, sticky="ns")
        self.left_frame.columnconfigure(0, weight=1)

        self.btn_upload.grid(row=0, column=0, sticky="ew", pady=2)
        self.btn_predict.grid(row=1, column=0, sticky="ew", pady=2)

        self.lbl_preview.grid(row=2, column=0, pady=(10, 2))
        self.preview.grid(row=3, column=0, pady=2)

        self.lbl_results.grid(row=4, column=0, pady=(10, 2))
        self.results.grid(row=5, column=0, sticky="nsew", pady=2)
        self.left_frame.rowconfigure(5, weight=1)

        # Right display area
        self.canvas.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.label.place(relx=0.5, rely=0.5, anchor="center")

    ##################################################
    def upload_image(self):
        """Open file dialog and load an image."""
        path = filedialog.askopenfilename(title="Select an image")
        if not path:
            return
        try:
            self.original_image = Image.open(path).convert("RGB")
            self.show_image()
            self.show_preview()
            self.btn_predict.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image:\n{e}")

    def show_image(self):
        """Display the uploaded image on the right canvas."""
        self.canvas.delete("all")
        self.label.place_forget()

        # Resize image to fit the canvas while maintaining aspect ratio
        img = self.original_image.copy()
        w, h = self.canvas.winfo_width() - 4, self.canvas.winfo_height() - 4
        img.thumbnail((max(1, w), max(1, h)))
        self.tk_image = ImageTk.PhotoImage(img)

        # Center image in canvas
        self.canvas.create_image(self.canvas.winfo_width() // 2,
                                 self.canvas.winfo_height() // 2,
                                 image=self.tk_image, anchor="center")

    def get_processed_image(self):
        """Convert to grayscale, resize to 28x28, and invert colors."""
        if not self.original_image:
            return None
        img = ImageOps.grayscale(self.original_image).resize((28, 28))
        return ImageOps.invert(img)

    def show_preview(self):
        """Display the 28x28 processed image in the preview box."""
        img = self.get_processed_image()
        self.preview.delete("all")
        if img:
            small = img.resize((84, 84))  # Scaled-up for visibility
            self.preview_tk = ImageTk.PhotoImage(small)
            self.preview.create_image(42, 42, image=self.preview_tk)

    def predict_image(self):
        """Run model prediction and display top 3 predictions."""
        img = self.get_processed_image()
        if not img:
            return

        tensor = transforms.ToTensor()(img).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0]
            top_p, top_c = torch.topk(probs, 3)

        # Clear old results
        for i in self.results.get_children():
            self.results.delete(i)

        # Insert top 3 predictions
        for p, c in zip(top_p, top_c):
            self.results.insert("", "end",
                                values=(class_names[c.item()], f"{p.item() * 100:.1f}%"))
    ##################################################


if __name__ == "__main__":
    root = tk.Tk()
    app = FashionApp(root)
    app.grid(sticky="nsew")  # Fill entire window
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    root.mainloop()
