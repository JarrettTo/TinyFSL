import cv2
import os
import numpy as np
import pandas as pd
import csv
import glob
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Input
from keras.layers import Dense
from keras.applications.mobilenet import preprocess_input


from keras.layers import Add, Activation, Concatenate, Dropout
from keras.layers import Flatten, MaxPooling2D
import keras.backend as K
from inception_v4 import create_model



def preprocess_frame(frame):
    frame = cv2.resize(frame, (299, 299))  # Resize to 299x299 for InceptionV4
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    frame = np.expand_dims(frame, axis=0)
    frame = preprocess_input(frame)  # Preprocess for InceptionV4
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
    




base_dir = './dataset/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/test/'

# Load the CSV file and split the data
csv_file_path = './dataset/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.test.corpus.csv'  # Update with the actual path
data = pd.read_csv(csv_file_path, sep='|')  # Use the correct separator
split_data = data
data_to_save = []
# Extracting the relevant columns
image_sequence_paths_temp = split_data['video'].tolist()
sequence_glosses = split_data['orth'].tolist()
text_translations = split_data['translation'].tolist()
prefix = "./dataset/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/test/"
image_sequence_paths = [path.replace('/1', '') for path in image_sequence_paths_temp]

base_model = create_model(weights='imagenet', include_top=False)
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(2560, activation='relu')(x)
custom_model = Model(inputs=base_model.input, outputs=x)


for i,img_path in enumerate(image_sequence_paths[:5]):
    full_path = os.path.join(base_dir, img_path)


    images = glob.glob(full_path)
    if images:
        aggregated_features = extract_and_aggregate_features(custom_model, images)
        data_to_save.append({
            "features": aggregated_features.flatten(),  # Flatten the features if they are multi-dimensional
            "gloss": sequence_glosses[i],
            "translation": text_translations[i]
        })
        
    else:
        print(f"No images found in {images}.")

df = pd.DataFrame(data_to_save)

# Save to CSV
csv_file = 'aggregated_features.csv'
df.to_csv(csv_file, index=False)
print(f"Data saved to {csv_file}")

    
cv2.destroyAllWindows()