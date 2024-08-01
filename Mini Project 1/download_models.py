import gdown
import os
import zipfile

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Download the zip file containing the models
url = 'https://drive.google.com/uc?id=1_aDScOvBeBLCn_iv0oxSO8X1ySQpSbIS'
output = 'modelNweight.zip'
gdown.download(url, output, quiet=False)

# Unzip the downloaded file
with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall('models')

# Remove the zip file
os.remove(output)

print("Models downloaded and extracted successfully.")