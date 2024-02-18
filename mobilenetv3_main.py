import cv2
import os
import numpy as np
import pandas as pd
import csv
import glob
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input


def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))  # Resize to 224x224 for MobileNetV3
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    frame = np.expand_dims(frame, axis=0)
    frame = preprocess_input(frame)  # Preprocess for MobileNetV3
    return frame

def extract_and_aggregate_features(model, image_paths):
    features = []
    for img_path in image_paths:
        frame = cv2.imread(img_path)
        processed_frame = preprocess_frame(frame)
        feature = model.predict(processed_frame)
        features.append(feature)
    aggregated_features = np.mean(np.array(features), axis=0)
    return aggregated_features

base_dir = './dataset/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/train/'

# Load the CSV file and split the data
csv_file_path = './dataset/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv'
data = pd.read_csv(csv_file_path, sep='|')
split_data = data
data_to_save = []
image_sequence_paths_temp = split_data['video'].tolist()
sequence_glosses = split_data['orth'].tolist()
text_translations = split_data['translation'].tolist()
prefix = "./dataset/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/train/"
image_sequence_paths = [path.replace('/1', '') for path in image_sequence_paths_temp]

base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(2560, activation='relu')(x)  # Adjust the number of neurons if needed
custom_model = Model(inputs=base_model.input, outputs=x)

for i, img_path in enumerate(image_sequence_paths[:3]):
    full_path = os.path.join(base_dir, img_path)

    images = glob.glob(full_path)
    if images:
        aggregated_features = extract_and_aggregate_features(custom_model, images)
        flattened_features = aggregated_features.flatten()
        print(f"Length of feature vector for image {i}: {len(flattened_features)}")  # Check the length
        features_string = ' '.join([f'{num:.17g}' for num in flattened_features])  # Convert array to string
        data_to_save.append({
            "features": features_string,  # Store features as a string
            "gloss": sequence_glosses[i],
            "translation": text_translations[i]
        })
    else:
        print(f"No images found in {full_path}.")


df = pd.DataFrame(data_to_save)

# Save to CSV
csv_file = 'train_aggregated_features.csv'
df.to_csv(csv_file, index=False, float_format='%.17g')
print(f"Data saved to {csv_file}")

cv2.destroyAllWindows()
