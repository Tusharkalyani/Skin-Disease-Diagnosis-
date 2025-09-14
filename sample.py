'''import cv2
import numpy as np

img = cv2.imread('image2.jpg')
img_resize = cv2.resize(img,(250,250))
cv2.imshow('without nor',img_resize)
img_nor =img_resize/255.0
cv2.imshow('with nor',img_nor)
img_dim = np.expand_dims(img_nor,axis=0)
print(img_dim)
'''

'''import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # Support Vector Machine for classification
from sklearn.metrics import accuracy_score
from joblib import dump, load  # For saving/loading models

# Paths
data_dir = '/Users/ktanvee/Downloads/skin-disease-datasaet/train_set'
classes = os.listdir(data_dir)

# Parameters
img_height, img_width = 64, 64  # Resize images to this size

# Load dataset
X = []
y = []

for label, class_name in enumerate(classes):
    class_dir = os.path.join(data_dir, class_name)
    if os.path.isdir(class_dir):  # Check if class_dir is a directory
        for img_name in os.listdir(class_dir):
            if img_name == '.DS_Store':
                continue  # Skip .DS_Store files
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (img_width, img_height))  # Resize to fixed size
            img = img.flatten()  # Flatten to a 1D vector
            X.append(img)
            y.append(label)

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = SVC(kernel='linear', probability=True)  # Support Vector Classifier
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Save the model
dump(model, 'skin_disease_model.joblib')
print("Model saved as skin_disease_model.joblib")
'''

'''import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
import joblib

# Load your dataset
data_dir = '/Users/ktanvee/Downloads/skin-disease-datasaet/train_set'

# Load dataset
X = []
y = []

img_height, img_width = 64, 64  # Resize dimensions
classes = os.listdir(data_dir)

for label, class_name in enumerate(classes):
    class_dir = os.path.join(data_dir, class_name)
    if os.path.isdir(class_dir):
        for img_name in os.listdir(class_dir):
            if img_name == '.DS_Store':
                continue
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (img_width, img_height))  # Resize
            img = img.flatten()  # Flatten to 1D
            X.append(img)
            y.append(label)

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
    ('scaler', StandardScaler()),  # Normalize features
    ('pca', PCA(n_components=50)),  # Dimensionality reduction
    ('classifier', RandomForestClassifier(random_state=42))  # Random Forest
])

# Define hyperparameters for tuning
param_grid = {
    'classifier__n_estimators': [100, 300, 500],
    'classifier__max_depth': [10, 15, 20],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# Grid Search with Cross-Validation
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Predict and evaluate
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)

# Save the model
joblib.dump(best_model, 'skin_disease_rf_model_optimized.joblib')
print("Optimized model saved to skin_disease_rf_model_optimized.joblib")
'''
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import joblib

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2, 
                                   height_shift_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    '/Users/ktanvee/Downloads/skin-disease-datasaet/train_set',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)
test_generator = test_datagen.flow_from_directory(
    '/Users/ktanvee/Downloads/skin-disease-datasaet/test_set',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, validation_data=test_generator, epochs=10)

# Print final accuracy and loss
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
train_loss = history.history['loss'][-1]
val_loss = history.history['val_loss'][-1]

print(f"Final Training Accuracy: {train_acc * 100:.2f}%")
print(f"Final Validation Accuracy: {val_acc * 100:.2f}%")
print(f"Final Training Loss: {train_loss:.4f}")
print(f"Final Validation Loss: {val_loss:.4f}")

# Save TensorFlow model in .h5 format
model.save('skin_disease_model.h5')
print("Model saved as 'skin_disease_model.h5'")

# Wrap the model save in joblib
class KerasModelWrapper:
    def __init__(self, model_path):
        self.model_path = model_path

    def save(self, filename):
        joblib.dump(self.model_path, filename)

    @staticmethod
    def load(filename):
        model_path = joblib.load(filename)
        return load_model(model_path)

# Save model as a joblib-compatible object
keras_wrapper = KerasModelWrapper('skin_disease_model.h5')
keras_wrapper.save('skin_disease_ten_model.joblib')
print("Optimized TensorFlow model saved as 'skin_disease_ten_model.joblib'")
