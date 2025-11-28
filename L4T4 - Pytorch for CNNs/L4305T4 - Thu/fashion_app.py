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

        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FashionCNN().to(self.device)
        self.model.load_state_dict(torch.load("fashion_model.pth", map_location=self.device))
        self.model.eval()

        self.original_image = None
        self.tk_image = None

        self._create_widgets()
        self.update_ui()

    def _create_widgets(self):
        left = ttk.Frame(self)
        left.pack(side="left", fill="y", padx=5, pady=5)

        self.btn_upload = ttk.Button(left, text="Upload Image", command=self.upload_image)
        self.btn_upload.pack(fill="x", pady=2)

        self.btn_predict = ttk.Button(left, text="Predict", command=self.predict_image, state=tk.DISABLED)
        self.btn_predict.pack(fill="x", pady=2)

        ttk.Label(left, text="Preview (28x28):").pack(pady=(10, 2))
        self.preview = tk.Canvas(left, width=84, height=84, bg="#eee")
        self.preview.pack()

        ttk.Label(left, text="Top Predictions:").pack(pady=(10, 2))
        self.results = ttk.Treeview(left, columns=("Class", "Conf"), show="headings", height=4)
        self.results.heading("Class", text="Class")
        self.results.heading("Conf", text="Confidence")
        self.results.column("Class", width=100)
        self.results.column("Conf", width=80, anchor="e")
        self.results.pack(fill="both", expand=True)

        # Image display
        self.canvas = tk.Canvas(self, bg="lightgray")
        self.canvas.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        self.label = ttk.Label(self.canvas, text="Upload an image")
        self.label.place(relx=0.5, rely=0.5, anchor="center")

    def upload_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
        if not path: return
        try:
            self.original_image = Image.open(path).convert("RGB")
            self.show_image()
            self.show_preview()
            self.update_ui()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_image(self):
        self.canvas.delete("all")
        self.label.place_forget()
        img = self.original_image.copy()
        img.thumbnail((self.canvas.winfo_width() - 4, self.canvas.winfo_height() - 4))
        self.tk_image = ImageTk.PhotoImage(img)
        self.canvas.create_image(self.canvas.winfo_width()//2, self.canvas.winfo_height()//2,
                                 image=self.tk_image, anchor="center")

    def get_processed_image(self):
        if not self.original_image: return None
        img = ImageOps.grayscale(self.original_image).resize((28, 28))
        return ImageOps.invert(img)

    def show_preview(self):
        img = self.get_processed_image()
        self.preview.delete("all")
        if img:
            small = img.resize((84, 84))
            self.preview_tk = ImageTk.PhotoImage(small)
            self.preview.create_image(42, 42, image=self.preview_tk)

    def predict_image(self):
        img = self.get_processed_image()
        if not img: return
        tensor = transforms.ToTensor()(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0]
            top_p, top_c = torch.topk(probs, 3)

        for i in self.results.get_children():
            self.results.delete(i)
        for p, c in zip(top_p, top_c):
            self.results.insert("", "end", values=(class_names[c.item()], f"{p.item()*100:.1f}%"))

    def update_ui(self):
        self.btn_predict.config(state=tk.NORMAL if self.original_image else tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    app = FashionApp(root)
    app.pack(fill="both", expand=True)
    root.mainloop()
