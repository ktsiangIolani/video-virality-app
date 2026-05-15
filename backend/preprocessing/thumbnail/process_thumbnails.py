# This file assumes that the dataset used is already cleaned using the image.py file

import sys
from pathlib import Path
import requests
from PIL import Image, ImageOps
from io import BytesIO
import numpy as np
import pandas as pd
import h5py
from google.cloud import storage

# 1. Setup paths
root_dir = Path(__file__).resolve().parent.parent.parent
csv_path = root_dir / 'USvideos.csv'
h5_path = root_dir / 'processed_images.h5'

sys.path.insert(0, str(root_dir)) 

def preprocess_thumbnail_pad(url, target_size=(224, 224), pad_color=(0, 0, 0)): 
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status() 
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img = ImageOps.pad(img, target_size, method=Image.Resampling.BICUBIC, color=pad_color)
        
        # CHANGED: No longer dividing by 255.0. Saving as tiny 8-bit integers!
        img_array = np.array(img).astype(np.uint8)
        return img_array
    except Exception as e:
        return None

# 2. Load your ALREADY CLEANED dataset
print(f"Loading dataset from: {csv_path}")
data = pd.read_csv(csv_path)
num_images = len(data)

print(f"Starting optimized pipeline for {num_images} images...")

# 3. Open the HDF5 Vault directly on the hard drive
with h5py.File(h5_path, 'w') as hf:
    
    # CHANGED: dtype is now np.uint8 to save massive amounts of hard drive space and RAM!
    images_db = hf.create_dataset("images", shape=(num_images, 224, 224, 3), dtype=np.uint8)
    
    # 4. Process and stream directly to disk
    for i, (index, row) in enumerate(data.iterrows()):
        url = row['thumbnail_link'] 
        img_array = preprocess_thumbnail_pad(url)
        
        if img_array is not None:
            images_db[i] = img_array
        else:
            # CHANGED: Fallback blank images are also saved as uint8
            images_db[i] = np.zeros((224, 224, 3), dtype=np.uint8)
            
        # Print a progress update every 500 images
        if i % 500 == 0 and i > 0:
            print(f"Saved {i} / {num_images} images to disk...")

gcs_client = storage.Client()
gcs_client.bucket('video-virality').blob('processed_images.h5').upload_from_filename(str(h5_path))
print(f"\n--- SUCCESS: Uploaded processed_images.h5 to GCS bucket video-virality ---")