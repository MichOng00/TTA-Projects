import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import torch
import torchvision.transforms as transforms
from tkinterdnd2 import DND_FILES, TkinterDnD  # Added for drag-and-drop

# Import model + class names from training script
from fashion_train import FashionCNN, class_names


class FashionApp(ttk.Frame):
    def __init__(self, root):
        super().__init__(root, padding="15")
        self.root = root
        # Use TkinterDnD.Tk instead of tk.Tk to enable drag-and-drop
        self.root.title("FashionMNIST Predictor")
        self.root.geometry("750x550")
        self.root.minsize(700, 500)

        # --- Style Configuration (NEW) ---
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"))

        # --- Model Loading ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FashionCNN().to(self.device)
        self.model.load_state_dict(torch.load("../fashion_model.pth", map_location=self.device))
        self.model.eval()

        # --- Initialize UI Variables ---
        self.original_image = None
        self.tk_image = None
        self.crop_rect = None
        self.crop_coords = None

        # --- Create UI Layout (NEW Structure) ---
        self._create_widgets()
        self.update_ui_state() # Initial state update

        # --- Drag-and-Drop Setup (NEW) ---
        self.canvas.drop_target_register(DND_FILES)
        self.canvas.dnd_bind('<<Drop>>', self.handle_drop)

    def _create_widgets(self):
        """Create and arrange all UI components."""
        # Main layout frames
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.columnconfigure(1, weight=3) # Give more weight to the image column
        main_frame.columnconfigure(0, weight=2)
        main_frame.rowconfigure(0, weight=1)

        controls_frame = self._create_controls_frame(main_frame)
        controls_frame.grid(row=0, column=0, padx=(0, 10), pady=5, sticky="nsew")

        image_frame = self._create_image_frame(main_frame)
        image_frame.grid(row=0, column=1, pady=5, sticky="nsew")


    def _create_controls_frame(self, parent):
        """Create the left-side panel with controls and results."""
        frame = ttk.Frame(parent, padding=10)
        frame.rowconfigure(5, weight=1) # Allow results to expand

        # --- Control Widgets ---
        ttk.Label(frame, text="Controls", style="Header.TLabel").grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky="w")
        self.btn_upload = ttk.Button(frame, text="Upload Image", command=self.upload_image)
        self.btn_upload.grid(row=1, column=0, columnspan=2, pady=4, sticky="ew")

        self.btn_clear_crop = ttk.Button(frame, text="Clear Crop", command=self.clear_crop, state=tk.DISABLED)
        self.btn_clear_crop.grid(row=2, column=0, pady=4, sticky="ew")

        self.btn_predict = ttk.Button(frame, text="Predict", command=self.predict_image, state=tk.DISABLED)
        self.btn_predict.grid(row=2, column=1, pady=4, sticky="ew")

        # --- Processed Image Preview (NEW) ---
        ttk.Label(frame, text="Processed Preview (28x28)", font=("Segoe UI", 10, "italic")).grid(row=3, column=0, columnspan=2, pady=(15, 5), sticky="w")
        self.preview_canvas = tk.Canvas(frame, width=84, height=84, bg="#f0f0f0", relief="sunken", bd=1)
        self.preview_canvas.grid(row=4, column=0, columnspan=2, pady=5)
        self.preview_label = ttk.Label(self.preview_canvas, text="N/A", font=("Segoe UI", 9))
        self.preview_label.place(relx=0.5, rely=0.5, anchor='center')

        # --- Results Display (NEW Treeview) ---
        ttk.Label(frame, text="Top Predictions", style="Header.TLabel").grid(row=5, column=0, columnspan=2, pady=(20, 10), sticky="w")
        self.results_tree = ttk.Treeview(frame, columns=("Class", "Confidence"), show="headings", height=3)
        self.results_tree.heading("Class", text="Class")
        self.results_tree.heading("Confidence", text="Confidence")
        self.results_tree.column("Class", width=120)
        self.results_tree.column("Confidence", width=80, anchor="e")
        self.results_tree.grid(row=6, column=0, columnspan=2, sticky="nsew")

        return frame

    def _create_image_frame(self, parent):
        """Create the right-side panel for image display."""
        frame = ttk.Frame(parent, padding=10)
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)

        ttk.Label(frame, text="Image", style="Header.TLabel").grid(row=0, column=0, pady=(0, 10), sticky="w")
        self.canvas = tk.Canvas(frame, bg="lightgray", relief="sunken", bd=1)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas_label = ttk.Label(self.canvas, text="Drag & Drop or Upload an Image", font=("Segoe UI", 12))
        self.canvas_label.place(relx=0.5, rely=0.5, anchor='center')

        # --- Crop Bindings ---
        self.canvas.bind("<ButtonPress-1>", self.start_crop)
        self.canvas.bind("<B1-Motion>", self.draw_crop)

        return frame

    def handle_drop(self, event):
        """Handle file drop event."""
        self.load_image(event.data)

    def upload_image(self):
        """Open file dialog to select an image."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if file_path:
            self.load_image(file_path)

    def load_image(self, file_path):
        """Load an image, display it, and reset state."""
        try:
            # Clean up path if it's wrapped in braces (from dnd)
            clean_path = file_path.strip().replace("{", "").replace("}", "")
            self.original_image = Image.open(clean_path).convert("RGB")
            self.display_image(self.original_image)
            self.clear_crop()
            self.update_ui_state()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image: {e}")

    def display_image(self, img):
        """Fit image to canvas and display it."""
        self.canvas.delete("all")
        self.canvas_label.place_forget() # Hide placeholder text
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if w < 2 or h < 2: # Canvas not ready yet, reschedule
            self.root.after(50, lambda: self.display_image(img))
            return

        img_resized = img.copy()
        img_resized.thumbnail((w - 4, h - 4)) # -4 for border offset
        self.tk_image = ImageTk.PhotoImage(img_resized)

        # Center the image on the canvas
        x_offset = (w - self.tk_image.width()) // 2
        y_offset = (h - self.tk_image.height()) // 2
        self.canvas.create_image(x_offset, y_offset, image=self.tk_image, anchor='nw')

    def start_crop(self, event):
        """Begin drawing the crop rectangle."""
        if not self.original_image: return
        if self.crop_rect: self.canvas.delete(self.crop_rect)

        self.start_x, self.start_y = event.x, event.y
        self.crop_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline="red", width=2
        )

    def draw_crop(self, event):
        """Update crop rectangle as mouse is dragged."""
        if not self.crop_rect: return
        self.canvas.coords(self.crop_rect, self.start_x, self.start_y, event.x, event.y)
        self.update_processed_preview() # Live preview

    def clear_crop(self):
        """Reset the crop selection."""
        if self.crop_rect:
            self.canvas.delete(self.crop_rect)
        self.crop_rect = None
        self.crop_coords = None
        self.update_processed_preview()
        self.update_ui_state()

    def get_image_for_prediction(self):
        """Get the full or cropped image, ready for the model."""
        if not self.original_image: return None
        img_to_process = self.original_image

        if self.crop_rect:
            # Get canvas image and its offsets to calculate crop on original
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()
            img_w, img_h = self.tk_image.width(), self.tk_image.height()
            x_offset = (canvas_w - img_w) // 2
            y_offset = (canvas_h - img_h) // 2

            # Get crop coords relative to the displayed image
            c_x1, c_y1, c_x2, c_y2 = self.canvas.coords(self.crop_rect)
            box_x1 = max(0, c_x1 - x_offset)
            box_y1 = max(0, c_y1 - y_offset)
            box_x2 = min(img_w, c_x2 - x_offset)
            box_y2 = min(img_h, c_y2 - y_offset)

            if box_x1 < box_x2 and box_y1 < box_y2:
                # Scale these coordinates to the original image size
                scale_x = self.original_image.width / img_w
                scale_y = self.original_image.height / img_h
                orig_x1 = int(box_x1 * scale_x)
                orig_y1 = int(box_y1 * scale_y)
                orig_x2 = int(box_x2 * scale_x)
                orig_y2 = int(box_y2 * scale_y)
                img_to_process = self.original_image.crop((orig_x1, orig_y1, orig_x2, orig_y2))

        # --- Process image for the model ---
        img = ImageOps.grayscale(img_to_process).resize((28, 28), Image.Resampling.LANCZOS)
        img = ImageOps.invert(img)
        return img

    def update_processed_preview(self):
        """Update the 28x28 preview canvas."""
        processed_img = self.get_image_for_prediction()
        self.preview_canvas.delete("all")

        if processed_img:
            self.preview_label.place_forget()
            # Resize for better visibility on the preview canvas
            preview_img_resized = processed_img.resize((84, 84), Image.Resampling.NEAREST)
            self.preview_tk_img = ImageTk.PhotoImage(preview_img_resized)
            self.preview_canvas.create_image(42, 42, image=self.preview_tk_img)
        else:
            self.preview_canvas.create_rectangle(0, 0, 84, 84, fill="#f0f0f0", outline="")
            self.preview_label.place(relx=0.5, rely=0.5, anchor='center')

    def predict_image(self):
        """Process the image and get a prediction from the model."""
        img = self.get_image_for_prediction()
        if img is None:
            messagebox.showerror("Error", "No image loaded!")
            return

        transform = transforms.ToTensor()
        tensor = transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            # Get top 3 predictions
            top_probs, top_preds = torch.topk(probs, 3)

        # Clear previous results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        # Display new results
        for i in range(top_preds.size(0)):
            pred_class = class_names[top_preds[i].item()]
            confidence = top_probs[i].item() * 100
            self.results_tree.insert("", "end", values=(pred_class, f"{confidence:.2f}%"))

    def update_ui_state(self):
        """Enable or disable buttons based on the application state."""
        image_loaded = self.original_image is not None
        crop_active = self.crop_rect is not None

        self.btn_predict.config(state=tk.NORMAL if image_loaded else tk.DISABLED)
        self.btn_clear_crop.config(state=tk.NORMAL if crop_active else tk.DISABLED)


if __name__ == "__main__":
    # Use the TkinterDnD wrapper for the root window
    root = TkinterDnD.Tk()
    app = FashionApp(root)
    root.pack_propagate(False)
    app.pack(fill="both", expand=True)
    root.mainloop()