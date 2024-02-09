import gzip
import pickle
import torch

def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        print("HI  :", loaded_object)

        # Save the loaded object to a text file
        with open('./notes/check_mobilenet.txt', 'w') as txt_file:
            txt_file.write(str(loaded_object))
        
        return loaded_object

load_dataset_file('./mobilenet/phoenix.mobilenet.dev.gz')
