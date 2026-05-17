import kagglehub

# Download latest version
path = kagglehub.dataset_download("mirzayasirabdullah07/customer-support-tickets-dataset-200k-records")

print("Path to dataset files:", path)