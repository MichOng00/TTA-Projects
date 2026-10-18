from quickdraw_utils import QuickDrawDataset

if __name__ == "__main__":
    root = "../QuickDraw"
    class_limit = 10
    max_items_per_class = 2000

    dataset = QuickDrawDataset(root, max_items_per_class, class_limit, 
                               class_names = ["apple", "The_Eiffel_Tower", "cat", "smiley_face", "sun", "toothbrush", "pizza", "hedgehog", "lighthouse", "ice_cream"])