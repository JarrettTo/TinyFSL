import gzip
import pickle
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import squeezenet1_1
from PIL import Image

# Load the pre-trained SqueezeNet model
model = squeezenet1_1(pretrained=True)

# Modify the classifier of SqueezeNet to match EfficientNet-B7 output
model.classifier[1] = nn.Conv2d(512, 1000, kernel_size=(1,1))
model.num_classes = 1000
model.eval()  # Set the model to inference mode

# Resize and normalize the tensor for SqueezeNet
def preprocess_tensor_data(tensor_data):
    # Reshape the tensor to create a single-channel image
    tensor_data = tensor_data.unsqueeze(0)  # Shape becomes [1, 181, 1024]

    # Convert to PIL Image for resizing
    image = transforms.ToPILImage()(tensor_data)

    # Resize image
    image = transforms.Resize((224, 224))(image)

    # Convert back to tensor
    tensor = transforms.ToTensor()(image)

    # Replicate the single channel to create a 3-channel image
    tensor = tensor.repeat(3, 1, 1)

    # Normalize
    tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor)

    return tensor


def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)

        # Processing each item in the dataset
        processed_data = []
        for item in loaded_object:
            if 'sign' in item:
                tensor_data = item['sign']
                preprocessed_data = preprocess_tensor_data(tensor_data)

                with torch.no_grad():
                    output = model(preprocessed_data.unsqueeze(0))  # Add batch dimension

                # Creating a dictionary with the original data and the model output
                processed_item = item.copy()
                processed_item['squeezenet_output'] = output.cpu().numpy().tolist()  # Convert output to list

                processed_data.append(processed_item)

        # Saving the processed data to a file
        with open('squeezenet.txt', 'w') as output_file:
            output_file.write(str(processed_data))

        return processed_data

# Example usage
load_dataset_file('phoenix14t.pami0.test')