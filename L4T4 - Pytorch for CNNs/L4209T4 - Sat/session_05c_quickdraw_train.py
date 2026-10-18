# Just to download data first.
# Choose 10 classes to 1) fit the previous neural network structure (also 10 classes)
# and 2) limit space usage on computer / network training time.
# Also limit images per class for space and time (there are >100k images for each class!)
from quickdraw_utils import QuickDrawDataset

if __name__ == "__main__":
    # Root is one level above current folder, because I share the same folder across multiple classes. 
    # Maybe easier to just use "QuickDraw" especially for student.
    root = "../QuickDraw" 
    class_limit = 10
    max_items_per_class = 2000

    dataset = QuickDrawDataset(root, max_items_per_class, class_limit, 
                               class_names = ["apple", "The_Eiffel_Tower", "cat", "smiley_face", "sun", "toothbrush", "pizza", "hedgehog", "lighthouse", "ice_cream"])