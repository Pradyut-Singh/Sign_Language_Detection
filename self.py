import os

DATA_DIR = 'SavedImages'
IMG_SIZE = 300
BATCH_SIZE = 32
EPOCHS = 10
CLASSES = ['A', 'B', 'C']


def load_data(data_dir, classes):
    images = []
    labels = []
    print(f"Loading images from {data_dir}...")

    # Iterate through class directories
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Error: Directory {class_dir} does not exist.")
            continue
        else:
            print(class_dir)


load_data(DATA_DIR, CLASSES)
