import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import joblib
import h5py

data = pd.read_csv('processedUSvideos.csv')

# =============================
# REMOVE LATER
import ssl
import os

# This tells Python to ignore SSL certificate verification
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and 
    getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

# ==============================

# ==========================================
# 1. UPDATED: The Index-Aware Generator
# ==========================================
class MultimodalGenerator(tf.keras.utils.Sequence):
    def __init__(self, h5_path, indices, x2_data, x3_data, x4_data, x5_data, y_data, batch_size=32):
        self.h5_path = h5_path
        self.indices = indices  # This tells the generator WHICH rows to pick
        self.x2_data = x2_data
        self.x3_data = x3_data
        self.x4_data = x4_data
        self.x5_data = x5_data
        self.y_data = y_data
        self.batch_size = batch_size
        self.hf = h5py.File(self.h5_path, 'r')
        self.num_samples = len(self.indices)

    def __len__(self):
        return int(np.ceil(self.num_samples / float(self.batch_size)))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, self.num_samples)
        
        # Get the specific batch of "global IDs" for this step
        batch_indices = self.indices[start:end]

        # Sorted and undoed indices
        sort_idx = np.argsort(batch_indices)
        rev_idx = np.argsort(sort_idx)

        sorted_indices = batch_indices[sort_idx]
        batch_x1_sorted = self.hf['images'][sorted_indices.tolist()]

        # Use those IDs to pull the CORRECT images from the HDF5
        # We use list-based indexing for the H5 file to ensure alignment
        batch_x1 = batch_x1_sorted[rev_idx]

        # Nuclear option for x5
        raw_x5 = self.x5_data[start:end]
        batch_x5 = tf.constant(raw_x5, dtype=tf.string)
        
        # The metadata is already sliced, so we use standard local indexing
        batch_x2 = self.x2_data[start:end]
        batch_x3 = self.x3_data[start:end]
        batch_x4 = self.x4_data[start:end]
        batch_x5 = tf.reshape(batch_x5, [-1,1]) # Force the strings into a fixed-length Unicode numpy array
        batch_y  = self.y_data[start:end]

        return {
            'Group1_Input': batch_x1, 
            'Group2_Input': batch_x2, 
            'Group3_Input': batch_x3, 
            'Group4_Input': batch_x4, 
            'Group5_Input': batch_x5
        }, batch_y

        
# ==========================================
# 2. Setting up tf inputs
# ==========================================
# x1 -> image
x1_input = layers.Input(shape=(224, 224, 3), name='Group1_Input')

# Add Data Augmentation natively in the graph (has the model learn more robust features of thumbnails)
x1_aug = layers.RandomFlip("horizontal")(x1_input)
x1_aug = layers.RandomRotation(0.1)(x1_aug)
x1_aug = layers.RandomZoom(0.1)(x1_aug)

base_model = EfficientNetB0(weights='imagenet', include_top=False)
base_model.trainable = True

# !! Freeze all layers except for the final block
for layer in base_model.layers[:-20]: 
    layer.trainable = False

x1 = base_model(x1_input)
x1 = layers.GlobalAveragePooling2D()(x1)
x1 = layers.Dense(128, activation='relu', name='Group1_Final')(x1)

# x2 -> semantics
x2_input = layers.Input(shape=(384,), name='Group2_Input')
x2 = layers.Dense(128, activation='relu')(x2_input)
x2 = layers.Dropout(0.2)(x2)
x2 = layers.Dense(64, activation='relu', name='Group2_Final')(x2)

# x3 -> 1d features
x3_input = layers.Input(shape=(12,), name='Group3_Input')
x3 = layers.Dense(64, activation='relu')(x3_input)
x3 = layers.BatchNormalization()(x3)
x3 = layers.Dropout(0.2)(x3)
x3 = layers.Dense(32, activation='relu', name='Group3_Final')(x3)
x3 = layers.BatchNormalization()(x3)

# x4 -> binarizer
x4_input = layers.Input(shape=(500,), name='Group4_Input')
x4 = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l1(0.01), name='Group4_Final')(x4_input)

# x5 -> main_tag
x5_vocab = data['main_tag'].unique() 
x5_input = layers.Input(shape=(1,), dtype='string', name='Group5_Input')
x5_encoded = layers.StringLookup(vocabulary=x5_vocab)(x5_input) 
x5_embedding = layers.Embedding(input_dim=len(x5_vocab)+1, output_dim=4)(x5_encoded) 
x5_flat = layers.Flatten()(x5_embedding) 

# Merging nodes
merged = layers.Concatenate()([x1, x2, x3, x4, x5_flat])
x = layers.Dense(32, activation='relu')(merged)
x = layers.Dropout(0.2)(x) # Added Dropout
output = layers.Dense(2, activation='linear', name='Results')(x)

model = models.Model(inputs=[x1_input, x2_input, x3_input, x4_input, x5_input], outputs=output)
model.compile(
    optimizer=Adam(learning_rate=1e-4), 
    loss=tf.keras.losses.Huber(delta=1.0) # changed from MSE to account for outliers
)

# ==========================================
# 3. Fitting data into the model
# ==========================================

# Create a master list of all row indices

y = np.log1p(data[['views', 'likes']].values).astype('float32')

all_indices = np.arange(len(y))

# Shuffles indicies to counteract any bias
np.random.seed(42)
np.random.shuffle(all_indices)

# Manually split indices at 80%
split_idx = int(0.8 * len(y))
train_idx = all_indices[:split_idx]
val_idx = all_indices[split_idx:]

# Prepare NumPy/Pandas Data in RAM
x_group2 = np.load('description_semantics.npy')

data['log_exclamation'] = np.log1p(data['title_exCount'].values).astype('int32')
x3_df = data[['title_cLength', 'title_hasNumber', 'title_capsRatio', 'log_exclamation', 'title_endInQ', 'title_infoDensity', 'tags_count', 'tags_title_overlapRatio', 'description_tokenCounts', 'Negative', 'Neutral', 'Positive']]
x3_df = x3_df.fillna(0).astype('float32')
scaler = StandardScaler()
scaler.fit(x3_df.values[:split_idx])
x_group3 = scaler.transform(x3_df.values).astype('float32')

joblib.dump(scaler, 'scaler.pkl')
print("Saved scaler.pkl successfully!")

x_group4 = np.load('top_tags_binarized.npy').astype('int32')
x_group5 = data['main_tag'].fillna('none').astype(str).values

# Create Training Generator 
train_generator = MultimodalGenerator(
    h5_path='processed_images.h5',
    indices=train_idx, # Pass the first 80% of IDs
    x2_data=x_group2[:split_idx],
    x3_data=x_group3[:split_idx],
    x4_data=x_group4[:split_idx],
    x5_data=x_group5[:split_idx],
    y_data=y[:split_idx],
    batch_size=32
)

# Create Validation Generator
val_generator = MultimodalGenerator(
    h5_path='processed_images.h5',
    indices=val_idx, # Pass the last 20% of IDs
    x2_data=x_group2[split_idx:],
    x3_data=x_group3[split_idx:],
    x4_data=x_group4[split_idx:],
    x5_data=x_group5[split_idx:],
    y_data=y[split_idx:],
    batch_size=32
)

# Define the "Stop" rule
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=10,             # Wait 10 epochs to see if it improves again
    restore_best_weights=True # CRITICAL: Rolls the model back to its best version
)

# Setup the logger
# 'append=True' is great because if your training crashes and you restart, 
# it won't delete your previous progress.
csv_logger = CSVLogger('training_history.csv', append=False, separator=',')

reduce_lr = ReduceLROnPlateau( # learning rate reducer function
    monitor='val_loss', 
    factor=0.2,       # Reduce learning rate to 20% of its current value
    patience=4,       # Do this after 4 epochs of no improvement
    min_lr=1e-6,      # Don't go below this rate
    verbose=1
)

# Train using the generators! 
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=100,
    callbacks=[csv_logger, early_stopping, reduce_lr] # The model will now "watch itself"
)

model.save('virality_model.keras')
print("✅ Model successfully saved to 'virality_model.keras'!")


# !!!!!!!! DELETE THE FOLLOWING LATER (THIS IS JUST FOR A TEST RUN OF THE MODEL) !!!!!!!! 

# ==========================================
# 4. DRY RUN EXECUTION (Test 10 samples)
# ==========================================

# Create a tiny subset of indices for testing
# test_indices = all_indices[:10] 

# try:
#     print("\n--- Initializing Dry Run Generator ---")
#     dry_run_gen = MultimodalGenerator(
#         h5_path='processed_images.h5',
#         indices=test_indices,
#         x2_data=x_group2[:10],
#         x3_data=x_group3[:10],
#         x4_data=x_group4[:10],
#         x5_data=x_group5[:10],
#         y_data=y[:10],
#         batch_size=2
#     )

#     print("\n--- X-RAY DIAGNOSTIC ---")
#     sample_inputs, sample_labels = dry_run_gen[0]
    
#     for key, val in sample_inputs.items():
#         print(f"{key} -> Shape: {val.shape} | Dtype: {val.dtype}")
    
#     print(f"Labels (y) -> Shape: {sample_labels.shape} | Dtype: {sample_labels.dtype}")
#     print("------------------------\n")

#     print("--- Starting Model Fit (1 Epoch, 5 Batches) ---")
#     # We run for 1 epoch just to see the progress bar move
#     model.fit(dry_run_gen, epochs=1)
    
#     print("\n✅ DRY RUN SUCCESSFUL!")
#     print("The pipeline is aligned. You are ready for full training.")

# except Exception as e:
#     print("\n❌ DRY RUN FAILED")
#     print(f"Error Type: {type(e).__name__}")
#     print(f"Error Message: {e}")