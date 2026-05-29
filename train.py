import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Input,
    concatenate
)

# =========================================================
# LOAD TABULAR DATA
# =========================================================

print("\nLoading tabular data...\n")

columns = ["bedrooms", "bathrooms", "area", "zipcode", "price"]

df = pd.read_csv(
    "data/HousesInfo.txt",
    sep=" ",
    header=None,
    names=columns
)

print(df.head())

# =========================================================
# LOAD IMAGES
# =========================================================

dataset_path = "data/Houses Dataset"

images = []

# use small limit first
LIMIT = 200

print("\nLoading images...\n")

for i in range(LIMIT):

    image_path = os.path.join(
        dataset_path,
        f"{i+1}_frontal.jpg"
    )

    # check image exists
    if not os.path.exists(image_path):
        print(f"Missing image: {image_path}")
        continue

    # read image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Failed to load: {image_path}")
        continue

    # resize image
    image = cv2.resize(image, (128, 128))

    # normalize image
    image = image / 255.0

    images.append(image)

    # progress
    if i % 20 == 0:
        print(f"Processed {i} images")

# convert to numpy array
X_images = np.array(images)

print("\nImages loaded successfully!")
print("Image dataset shape:", X_images.shape)

# =========================================================
# TABULAR FEATURES
# =========================================================

X_tabular = df[["bedrooms", "bathrooms", "area"]]

y = df["price"]

# use same limit
X_tabular = X_tabular[:LIMIT]

y = y[:LIMIT]

# normalize tabular data
scaler = MinMaxScaler()

X_tabular = scaler.fit_transform(X_tabular)

print("\nTabular data normalized!")

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

(
    trainImages,
    testImages,
    trainTabular,
    testTabular,
    trainY,
    testY
) = train_test_split(
    X_images,
    X_tabular,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain-test split completed!")

print("Train Images Shape:", trainImages.shape)
print("Test Images Shape:", testImages.shape)

# =========================================================
# CNN MODEL
# =========================================================

# =========================================================
# CNN MODEL
# =========================================================

print("\nBuilding CNN model...\n")

# image input
image_input = Input(shape=(128, 128, 3))

# conv layer 1
y = Conv2D(
    32,
    (3, 3),
    activation='relu'
)(image_input)

y = MaxPooling2D(pool_size=(2, 2))(y)

# conv layer 2
y = Conv2D(
    64,
    (3, 3),
    activation='relu'
)(y)

y = MaxPooling2D(pool_size=(2, 2))(y)

# flatten
y = Flatten()(y)

# dense layer
y = Dense(128, activation='relu')(y)
# =========================================================
# TABULAR MODEL
# =========================================================

print("\nBuilding tabular model...\n")

tabular_input = Input(shape=(3,))

x = Dense(64, activation='relu')(tabular_input)

x = Dense(32, activation='relu')(x)

# =========================================================
# FEATURE FUSION
# =========================================================

print("\nCombining image + tabular features...\n")

combined = concatenate([y, x])

z = Dense(64, activation='relu')(combined)

z = Dense(32, activation='relu')(z)

# regression output
z = Dense(1, activation='linear')(z)

# =========================================================
# FINAL MULTIMODAL MODEL
# =========================================================

model = Model(
    inputs=[image_input, tabular_input],
    outputs=z
)

# compile model
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

print("\nMODEL SUMMARY:\n")

model.summary()

# =========================================================
# TRAIN MODEL
# =========================================================

print("\nTraining model...\n")

history = model.fit(
    [trainImages, trainTabular],
    trainY,
    validation_data=(
        [testImages, testTabular],
        testY
    ),
    epochs=10,
    batch_size=8,
    verbose=1
)

# =========================================================
# EVALUATE MODEL
# =========================================================

print("\nEvaluating model...\n")

predictions = model.predict(
    [testImages, testTabular]
)

# MAE
mae = mean_absolute_error(
    testY,
    predictions
)

# RMSE
rmse = np.sqrt(
    mean_squared_error(
        testY,
        predictions
    )
)

print("\n===================================")
print("FINAL RESULTS")
print("===================================")

print(f"MAE  : {mae}")

print(f"RMSE : {rmse}")

# =========================================================
# SAVE MODEL
# =========================================================

model.save("models/housing_multimodal_model.h5")

print("\nModel saved successfully!")

# =========================================================
# PLOT LOSS GRAPH
# =========================================================

plt.figure(figsize=(8,5))

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title("Training vs Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend(["Train Loss", "Validation Loss"])

plt.show()

# =========================================================
# PLOT MAE GRAPH
# =========================================================

plt.figure(figsize=(8,5))

plt.plot(history.history['mae'])

plt.plot(history.history['val_mae'])

plt.title("Training vs Validation MAE")

plt.xlabel("Epoch")

plt.ylabel("MAE")

plt.legend(["Train MAE", "Validation MAE"])

plt.show()

print("\nProject completed successfully!")