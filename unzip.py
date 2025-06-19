import zipfile
import os
import zipfile
import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
zip_file_path = 'archive.zip'

# Replace with the desired directory to extract the contents
destination_directory = 'unzipped_data'

# Create the destination directory if it doesn't exist
os.makedirs(destination_directory, exist_ok=True)

# Unzip the file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(destination_directory)

print(f"File unzipped to: {destination_directory}")

def load_image_dataset(image_size=(48, 48), color_mode='grayscale'):

    # Replace with the desired directory to extract the contents
    destination_directory = '/unzipped_data/train'

    class_names = sorted([
        d for d in os.listdir(destination_directory)
        if os.path.isdir(os.path.join(destination_directory, d))
    ])


    # 2. Prepare containers
    images = []
    labels = []

    # 3. Iterate over each class folder
    for class_name in class_names:
        class_dir = os.path.join(destination_directory, class_name)
        print(f"Processing class directory: {class_dir}") # Debugging print
        for fname in os.listdir(class_dir):
            fpath = os.path.join(class_dir, fname)
            # Only process files with common image extensions
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                continue

            # 4. Read image
            if color_mode == 'grayscale':
                img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(fpath, cv2.IMREAD_COLOR)

            if img is None:
                # skip unreadable/corrupt images
                # print(f"Could not read image: {fpath}") # Debugging print
                continue

            # 5. Resize to target size
            img = cv2.resize(img, image_size)

            # 6. Normalize pixel values to [0,1]
            img = img.astype('float32') / 255.0

            # 7. Add channel dimension if needed
            if color_mode == 'grayscale':
                img = np.expand_dims(img, axis=-1)  # (h, w) -> (h, w, 1)

            images.append(img)
            labels.append(class_name)

    # 8. Convert lists to arrays
    X = np.array(images, dtype='float32')
    y_text = np.array(labels)

    if y_text.size == 0:
         raise ValueError("No labels were loaded. Please check your data directory and image files.")


    # 9. Encode text labels to integers, then to one-hot
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y_text)
    y = to_categorical(integer_encoded)

    return X, y, label_encoder
# Assuming the zip file was unzipped to '/unzipped_data' in the previous step
x,y,label=load_image_dataset()