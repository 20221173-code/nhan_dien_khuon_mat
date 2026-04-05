import cv2
import os
import sys
from face_detector import FaceDetector
from utils import Utils
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceTrainer:
    def __init__(self):
        self.detector = FaceDetector()
        self.dataset_path = 'dataset'
        self.person_id = None
        self.image_count = 0
        self.cap = None
    
    def capture_images_from_camera(self, person_name, num_images=30):
        """Chụp ảnh khuôn mặt từ camera"""
        logger.info(f"🎥 Chụp {num_images} ảnh cho '{person_name}'...")
        
        # Tạo thư mục cho người
        Utils.create_person_folder(person_name, self.dataset_path)
        
        self.cap = cv2.VideoCapture(0)
        self.image_count = 0
        
        if not self.cap.isOpened():
            logger.error("✗ Không thể mở camera!")
            return False
        
        while self.image_count < num_images:
            ret, frame = self.cap.read()
            
            if not ret:
                logger.error("✗ Không thể đọc frame từ camera!")
                break
            
            # Phát hiện khuôn mặt
            faces = self.detector.detect_faces(frame)
            
            # Vẽ hình chữ nhật xung quanh khuôn mặt
            frame = self.detector.draw_faces(frame, faces)
            
            # Nếu có khuôn mặt được phát hiện
            if len(faces) > 0:
                # Chụp ảnh khuôn mặt lớn nhất
                face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = face
                face_roi = frame[y:y+h, x:x+w]
                
                # Lưu ảnh
                Utils.save_face_image(face_roi, person_name, self.image_count, self.dataset_path)
                self.image_count += 1
                
                # Hiển thị thông tin
                cv2.putText(frame, f"Captured: {self.image_count}/{num_images}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No face detected!", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Capturing Face Images', frame)
            
            # Nhấn 'q' để thoát
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        logger.info(f"✓ Chụp xong {self.image_count} ảnh cho '{person_name}'!")
        return True
    
    def train_all_faces(self):
        """Training mô hình với tất cả ảnh"""
        logger.info("🚀 Bắt đầu training mô hình...")
        
        success, label_dict = self.detector.train_model(self.dataset_path)
        
        if success:
            logger.info("✓ Training hoàn tất!")
            logger.info(f"Danh sách người: {label_dict}")
            return True, label_dict
        else:
            logger.error("✗ Training thất bại!")
            return False, {}
    
    def get_training_info(self):
        """Lấy thông tin training"""
        persons = Utils.get_all_persons(self.dataset_path)
        total_images = sum(p['image_count'] for p in persons)
        
        logger.info("=" * 50)
        logger.info("THÔNG TIN TRAINING")
        logger.info("=" * 50)
        for person in persons:
            logger.info(f"{person['name']}: {person['image_count']} ảnh")
        logger.info(f"Tổng cộng: {total_images} ảnh, {len(persons)} người")
        logger.info("=" * 50)
        
        return persons, total_images

def main():
    trainer = FaceTrainer()
    
    while True:
        print("\n" + "="*50)
        print("CHƯƠNG TRÌNH TRAINING NHẬN DIỆN KHUÔN MẶT")
        print("="*50)
        print("1. Chụp ảnh khuôn mặt mới")
        print("2. Xem thông tin training")
        print("3. Training mô hình")
        print("4. Thoát")
        print("="*50)
        
        choice = input("Chọn tùy chọn (1-4): ").strip()
        
        if choice == '1':
            person_name = input("Nhập tên người: ").strip()
            if not person_name:
                print("✗ Tên không hợp lệ!")
                continue
            
            try:
                num_images = int(input("Số ảnh cần chụp (mặc định 30): ") or "30")
            except ValueError:
                num_images = 30
            
            trainer.capture_images_from_camera(person_name, num_images)
        
        elif choice == '2':
            trainer.get_training_info()
        
        elif choice == '3':
            success, label_dict = trainer.train_all_faces()
            if success:
                print("✓ Model đã được lưu thành công!")
        
        elif choice == '4':
            print("Tạm biệt!")
            break
        
        else:
            print("✗ Tùy chọn không hợp lệ!")

if __name__ == "__main__":
    main()