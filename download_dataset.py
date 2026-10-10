import os
import kagglehub

kagglehub.login() # Login with your kaggle API token

# Path to data files
target_dir = os.path.abspath(os.path.join(os.getcwd(), 'data', 'raw'))

# Check if the main dataset file already exists locally
expected_file = os.path.join(target_dir, "application_train.csv")

if os.path.exists(expected_file):
    print(f"Dataset already exists at: {target_dir}. Skipping download.")
else:
    # Download latest version
    path = kagglehub.competition_download('home-credit-default-risk', 
                                        output_dir=target_dir, 
                                        force_download=True)

    print("Path to data files:", path)