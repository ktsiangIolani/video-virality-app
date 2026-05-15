import os
import sys
import io
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import re
from google.cloud import storage
from PIL import Image, ImageOps

# Import your protected preprocessing functions!
from preprocessing.text.tags import get_tags, get_tag_features
from preprocessing.text.title import capsRatio, infoDensity
from preprocessing.text.semantics import semantic_vectors
from preprocessing.text.sentiment_tokens import analyze_sentiment, count_tokens

# ==========================================
# 1. LOAD AI TOOLS IN GLOBAL MEMORY
# ==========================================
# We load these outside the function so they only load ONCE when the server starts.
# This makes predictions lightning fast!
print("Loading Models and Tools into memory...")
try:
    model = tf.keras.models.load_model('virality_model.keras')
    gcs_bucket = storage.Client().bucket('video-virality')
    scaler = joblib.load(io.BytesIO(gcs_bucket.blob('scaler.pkl').download_as_bytes()))
    mlb = joblib.load('tags_binarizer.pkl')
    print("✅ All tools loaded successfully!")
except Exception as e:
    print(f"⚠️ Warning: Could not load tools. Make sure train.py has finished running. Error: {e}")


# ==========================================
# 2. IMAGE PREPROCESSING (From your partner)
# ==========================================
def format_user_thumbnail(uploaded_image_file, target_size=(224, 224), pad_color=(0, 0, 0)): 
    """Formats a user's uploaded thumbnail to perfectly match EfficientNetB0."""
    try:
        img = Image.open(uploaded_image_file).convert('RGB')
        img = ImageOps.pad(img, target_size, method=Image.Resampling.BICUBIC, color=pad_color)
        img_array = np.array(img).astype(np.uint8) # Same 8-bit format as training
        return np.expand_dims(img_array, axis=0) # Shape: (1, 224, 224, 3)
    except Exception as e:
        print(f"Error processing image: {e}")
        # Fallback to a black image if upload is corrupted
        return np.zeros((1, 224, 224, 3), dtype=np.uint8)


# ==========================================
# 3. THE MAIN PREDICTION PIPELINE
# ==========================================
def predict_virality(title, description, tags_string, image_path):
    """
    Takes raw frontend inputs, processes them through the 5 branches, 
    and returns predicted views and likes.
    """
    
    # --- GROUP 1: IMAGE ---
    x1_image = format_user_thumbnail(image_path)
    
    # --- GROUP 2: SEMANTICS ---
    x2_semantics = semantic_vectors(description)
    x2_semantics = np.expand_dims(x2_semantics, axis=0) # Shape: (1, 384)
    
    # --- PREPARE INTERMEDIATE TEXT FEATURES ---
    tags_list = get_tags(tags_string)
    main_tag, overlap_ratio = get_tag_features({'tags_list': tags_list, 'title': title})
    sentiment = analyze_sentiment(description)
    token_count = len(count_tokens(description))
    
    # --- GROUP 3: TABULAR (SCALED) ---
    # Must perfectly match the order from train.py!
    tabular_features = [
        len(title) if title else 0,                                 # title_cLength
        1 if re.search(r'\d', title) else 0,                        # title_hasNumber
        capsRatio(title),                                           # title_capsRatio
        title.count('!') if title else 0,                           # title_exCount
        1 if title and title.strip() and title.strip()[-1]=='?' else 0, # title_endInQ
        infoDensity(title),                                         # title_infoDensity
        len(tags_list),                                             # tags_count
        overlap_ratio,                                              # tags_title_overlapRatio
        token_count,                                                # description_tokenCounts
        sentiment['Negative'],                                      # Negative
        sentiment['Neutral'],                                       # Neutral
        sentiment['Positive']                                       # Positive
    ]
    
    # Convert to 2D array and scale it
    x3_tabular = np.array([tabular_features])
    x3_scaled = scaler.transform(x3_tabular).astype('float32') # Shape: (1, 12)
    
    # --- GROUP 4: BINARIZED TAGS ---
    # Filter out tags that the model has never seen before
    valid_tags = [tag for tag in tags_list if tag in mlb.classes_]
    x4_tags = mlb.transform([valid_tags]).astype('int32') # Shape: (1, 500)
    
    # --- GROUP 5: MAIN TAG ---
    x5_main_tag = tf.constant([[main_tag]], dtype=tf.string) # Shape: (1, 1) string array
    
    # --- BUILD INPUT DICTIONARY ---
    model_inputs = {
        'Group1_Input': x1_image,
        'Group2_Input': x2_semantics,
        'Group3_Input': x3_scaled,
        'Group4_Input': x4_tags,
        'Group5_Input': x5_main_tag
    }
    
    # --- PREDICT ---
    prediction = model.predict(model_inputs, verbose=0)
    
    # CRITICAL: Reverse the np.log1p() used during training!
    # np.expm1 does (e^x - 1) which gets our real numbers back.
    real_numbers = np.expm1(prediction[0])
    
    predicted_views = int(real_numbers[0])
    predicted_likes = int(real_numbers[1])
    
    return {
        "views": predicted_views,
        "likes": predicted_likes
    }


# ==========================================
# 4. TEST BLOCK (For terminal testing)
# ==========================================
if __name__ == "__main__":
    print("\n--- Testing Prediction Pipeline ---")
    
    # You can put a temporary test image in your backend folder
    test_image = "test_thumbnail.jpg" 
    
    # Create a dummy blank image if you don't have one right now
    if not os.path.exists(test_image):
        Image.new('RGB', (1280, 720), color = 'red').save(test_image)
        print(f"(Created a temporary red {test_image} for testing)")

    test_title = "I coded an AI to predict YouTube views! (IT WORKED?)"
    test_desc = "In this video, I build a deep learning multimodal neural network using Keras and Python."
    test_tags = "python|machine learning|artificial intelligence|coding|keras"
    
    try:
        results = predict_virality(test_title, test_desc, test_tags, test_image)
        print("\n🎉 PREDICTION SUCCESSFUL! 🎉")
        print(f"Title: {test_title}")
        print(f"Predicted Views: {results['views']:,}")
        print(f"Predicted Likes: {results['likes']:,}")
    except Exception as e:
        print(f"\n❌ Prediction Failed. Error: {e}")