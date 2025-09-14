'''import cv2
import numpy as np
import os
import sklearn.model_selection
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def load_and_preprocess_images(data_dir, image_size=(64, 64)):
    images = []
    labels = []

    for class_folder in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_folder)
        if os.path.isdir(class_path):
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)
                try: 
                    img = cv2.imread(image_path)
                    if img is None:
                        continue

                    img = cv2.resize(img, image_size)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    images.append(img.flatten())
                    labels.append(class_folder)
                except Exception as e:
                    pass

    return np.array(images), np.array(labels)

data_directory = "/Users/ktanvee/Downloads/skin-disease-datasaet/train_set"
image_size = (64, 64)

images, labels = load_and_preprocess_images(data_directory, image_size)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    images, labels, test_size=0.2, random_state=42
)

model = SVC(kernel='linear', C=1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

def predict_disease(image_path, model, image_size):
    img = cv2.imread(image_path)
    if img is None:
        return "Error: Could not load image"

    img = cv2.resize(img, image_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.flatten().reshape(1, -1)
    prediction = model.predict(img)
    return prediction[0]

# Update the path with absolute path
new_image_path = "/Users/ktanvee/Downloads/skin-disease-datasaet/test_set/BA- cellulitis/BA- cellulitis (1).webp"
predicted_disease = predict_disease(new_image_path, model, image_size)
print(f"Predicted Disease: {predicted_disease}")

import pickle
model_filename = "skin_disease_model.pkl"
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

print(f"Model saved to {model_filename}")
'''
import cv2
import numpy as np
import os
import sklearn.model_selection
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def load_and_preprocess_images(data_dir, image_size=(64, 64)):
    images = []
    labels = []

    for class_folder in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_folder)
        if os.path.isdir(class_path):
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)
                try: 
                    img = cv2.imread(image_path)
                    if img is None:
                        continue

                    img = cv2.resize(img, image_size)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    images.append(img.flatten())
                    labels.append(class_folder)
                except Exception as e:
                    pass

    return np.array(images), np.array(labels)

data_directory = "/Users/ktanvee/Downloads/skin-disease-datasaet/train_set"
image_size = (64, 64)

images, labels = load_and_preprocess_images(data_directory, image_size)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    images, labels, test_size=0.2, random_state=42
)


models = {
    'SVC (Linear)': SVC(kernel='linear', C=1),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, max_depth=3,random_state=42),
    'Naive Bayes': GaussianNB()
}


for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy}\n")
