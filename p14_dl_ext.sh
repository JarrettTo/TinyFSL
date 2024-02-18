URL="https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014-T.v3.tar.gz"

# Define the directory where the dataset will be stored
DATASET_DIR="./dataset"

# Create the dataset directory if it does not exist
mkdir -p $DATASET_DIR

# Navigate to the dataset directory
cd $DATASET_DIR

# Download the file using wget
echo "Downloading the dataset..."
wget $URL -O phoenix-2014-T.v3.tar.gz

# Extract the downloaded file
echo "Extracting the dataset..."
tar -xzf phoenix-2014-T.v3.tar.gz

# Remove the compressed file to save space
rm phoenix-2014-T.v3.tar.gz

echo "Download and extraction complete."