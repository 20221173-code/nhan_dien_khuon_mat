import cv2
import os
import numpy as np
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceDetector:
    def __init__(self):
        # Load cascade classifier
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.trained_model_path = 'trained_model/trained_face_recognizer.yml'
        self.model_loaded = False
        
        # Load model nếu tồn tại
        if os.path.exists(self.trained_model_path):
            try:
                self.face_recognizer.read(self.trained_model_path)
                self.model_loaded = True
                logger.info("✓ Model đã được load!")
            except Exception as e:
                logger.error(f"✗ Lỗi load model: {e}")
    
    def detect_faces(self, image):
        """Phát hiện khuôn mặt trong ảnh"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces
    
    def draw_faces(self, image, faces):
        """Vẽ hình chữ nhật xung quanh khuôn mặt"""
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return image
    
    def extract_face(self, image, face_coords):
        """Cắt khuôn mặt từ ảnh"""
        x, y, w, h = face_coords
        face = image[y:y+h, x:x+w]
        return face
    
    def prepare_training_data(self, dataset_path='dataset'):
        """Chuẩn bị dữ liệu training từ thư mục dataset"""
        faces = []
        labels = []
        label_dict = {}
        current_label = 0
        
        try:
            for person_name in os.listdir(dataset_path):
                person_path = os.path.join(dataset_path, person_name)
                
                if not os.path.isdir(person_path):
                    continue
                
                label_dict[current_label] = person_name
                
                for image_name in os.listdir(person_path):
                    image_path = os.path.join(person_path, image_name)
                    
                    # Đọc ảnh
                    image = cv2.imread(image_path)
                    if image is None:
                        continue
                    
                    # Phát hiện khuôn mặt
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    detected_faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                    
                    for (x, y, w, h) in detected_faces:
                        face_roi = gray[y:y+h, x:x+w]
                        face_roi = cv2.resize(face_roi, (200, 200))
                        
                        faces.append(face_roi)
                        labels.append(current_label)
                        logger.info(f"✓ Đã load: {image_path}")
                
                current_label += 1
            
            logger.info(f"✓ Chuẩn bị dữ liệu xong! {len(faces)} ảnh, {len(set(labels))} người")
            return faces, labels, label_dict
        
        except Exception as e:
            logger.error(f"✗ Lỗi chuẩn bị dữ liệu: {e}")
            return [], [], {}
    
    def train_model(self, dataset_path='dataset'):
        """Training mô hình nhận diện khuôn mặt"""
        logger.info("🚀 Bắt đầu training...")
        
        faces, labels, label_dict = self.prepare_training_data(dataset_path)
        
        if len(faces) == 0:
            logger.error("✗ Không có dữ liệu training!")
            return False
        
        try:
            # Training
            self.face_recognizer.train(faces, np.array(labels))
            
            # Tạo thư mục nếu chưa tồn tại
            os.makedirs('trained_model', exist_ok=True)
            
            # Lưu model
            self.face_recognizer.save(self.trained_model_path)
            self.model_loaded = True
            
            logger.info("✓ Training hoàn tất! Model đã được lưu!")
            return True, label_dict
        
        except Exception as e:
            logger.error(f"✗ Lỗi training: {e}")
            return False, {}
    
    def recognize_faces(self, image, confidence_threshold=70):
        """Nhận diện khuôn mặt trong ảnh"""
        if not self.model_loaded:
            logger.error("✗ Model chưa được load!")
            return []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        results = []
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))
            
            label, confidence = self.face_recognizer.predict(face_roi)
            
            if confidence < confidence_threshold:
                results.append({
                    'label': label,
                    'confidence': confidence,
                    'coords': (x, y, w, h)
                })
        
        return results