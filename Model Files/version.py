import os
from PIL import Image

dataset_path = "archive (1)/Oral cancer Dataset 2.0/OC Dataset kaggle new"

def check_images(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                img = Image.open(file_path)
                img.verify()  # Verify if the file is an image
            except (IOError, SyntaxError) as e:
                print(f"Problematic file: {file_path}")

check_images(dataset_path)
