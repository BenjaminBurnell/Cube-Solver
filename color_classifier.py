# color_classifier.py
import os
import pickle
from sklearn.neighbors import KNeighborsClassifier
import joblib
import numpy as np

# NOTE: I was originally going to use just straight python and not store the training data in a .pkl file but then I asked chatGPT how I could make the script run faster and more efficent thats the first thing it told me to do then I found I should try and take 6 frames then combined them to get a closer avg_hsv to reference the model

class CubeColorClassifier:
    def __init__(self, model_path="cube_color_model.pkl", data_path="cube_training_data.pkl"):
        self.model_path = model_path
        self.data_path = data_path
        self.model = KNeighborsClassifier(n_neighbors=3)
        
        """
        Remeber you can create your own if you have a different cube than me I have red, green, 
        blue, white, yellow, and orange. but you may have different shades so its best if you 
        create your own training model.
        """
        # Load existing training data if there is any
        if os.path.exists(self.data_path):
            with open(self.data_path, "rb") as f:
                data = pickle.load(f)
                self.samples = data["samples"]
                self.labels = data["labels"]
            if len(self.samples) > 0:
                self.model.fit(self.samples, self.labels)
        else:
            self.samples = []
            self.labels = []

    def add_training_sample(self, hsv_avg, color_label):
        # Add new HSV sample with color label
        hsv_avg = np.array(hsv_avg).reshape(1, -1)
        self.samples.append(hsv_avg.flatten())
        self.labels.append(color_label)
        self.model.fit(self.samples, self.labels)
        self.save_model()
        self.save_data()

    def predict(self, hsv_avg):
        # Predict color label from HSV array
        hsv_avg = np.array(hsv_avg).reshape(1, -1)
        return self.model.predict(hsv_avg)[0]

    def save_model(self):
        joblib.dump(self.model, self.model_path)

    def save_data(self):
        with open(self.data_path, "wb") as f:
            pickle.dump({"samples": self.samples, "labels": self.labels}, f)

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)

    def load_data(self):
        if os.path.exists(self.data_path):
            with open(self.data_path, "rb") as f:
                data = pickle.load(f)
                self.samples = data["samples"]
                self.labels = data["labels"]
