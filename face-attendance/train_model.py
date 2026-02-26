import cv2
import os
import numpy as np
import pickle

def train_model():
    faces_dir = "dataset/faces"
    if not os.path.exists(faces_dir):
        return False

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    
    faces = []
    ids = []
    names = {}
    id_count = 0

    subjects = os.listdir(faces_dir)
    if not subjects:
        return False

    for username in subjects:
        user_dir = os.path.join(faces_dir, username)
        if not os.path.isdir(user_dir):
            continue
        
        names[id_count] = username
        
        for image_name in os.listdir(user_dir):
            image_path = os.path.join(user_dir, image_name)
            img = cv2.imread(image_path)
            if img is None: continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(detected_faces) == 0:
                detected_faces = profile_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in detected_faces:
                face_roi = gray[y:y+h, x:x+w]
                if face_roi.size > 0:
                    # Standardize face size and normalize lighting
                    face_roi = cv2.resize(face_roi, (200, 200))
                    face_roi = cv2.equalizeHist(face_roi)
                    faces.append(face_roi)
                    ids.append(id_count)
        
        id_count += 1

    if faces:
        print(f"Training on {len(faces)} face samples...")
        recognizer.train(faces, np.array(ids))
        recognizer.save("trainer.yml")
        with open("names.pickle", "wb") as f:
            pickle.dump(names, f)
        print("Model trained and saved successfully.")
        return True
    
    print("No faces found in any images. Training failed.")
    return False

if __name__ == "__main__":
    train_model()
