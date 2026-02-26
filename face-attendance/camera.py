import cv2
import os

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        # Using built-in Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        if not success:
            return None
        
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def capture_image(self, username, count):
        dataset_path = f"dataset/faces/{username}"
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        
        success, image = self.video.read()
        if success:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                file_path = f"{dataset_path}/{username}_{count}.jpg"
                cv2.imwrite(file_path, image)
                return True
        return False
