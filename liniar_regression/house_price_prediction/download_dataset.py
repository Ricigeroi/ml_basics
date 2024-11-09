import kagglehub

# Download latest version
path = kagglehub.dataset_download("aravinii/house-price-prediction-treated-dataset")

print("Path to dataset files:", path)