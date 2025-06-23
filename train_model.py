import face_recognition
import os
import pickle
import numpy as np
from tqdm import tqdm

def train_model():
    known_encodings = []
    known_names = []
    
    print("Starting model training...")
    
    for emp_dir in tqdm(os.listdir("dataset")):
        dir_path = os.path.join("dataset", emp_dir)
        if not os.path.isdir(dir_path):
            continue
            
        name = emp_dir.split('_')[1]
        
        for img_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_name)
            image = face_recognition.load_image_file(img_path)
            
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(name)
    
    # Save trained model
    with open("trained_model/model.pkl", "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)
    
    print(f"✅ Model trained successfully with {len(known_names)} samples")
    print(f"Total employees registered: {len(set(known_names))}")

if __name__ == "__main__":
    train_model()
