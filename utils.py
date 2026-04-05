import cv2
import os
import shutil
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Utils:
    @staticmethod
    def create_person_folder(person_name, dataset_path='dataset'):
        """Tạo thư mục cho người mới"""
        person_path = os.path.join(dataset_path, person_name)
        os.makedirs(person_path, exist_ok=True)
        logger.info(f"✓ Tạo thư mục cho '{person_name}'")
        return person_path
    
    @staticmethod
    def save_face_image(image, person_name, image_count, dataset_path='dataset'):
        """Lưu ảnh khuôn mặt vào thư mục người"""
        person_path = Utils.create_person_folder(person_name, dataset_path)
        filename = f"{person_name}_{image_count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(person_path, filename)
        
        cv2.imwrite(filepath, image)
        logger.info(f"✓ Lưu ảnh: {filepath}")
        return filepath
    
    @staticmethod
    def get_next_person_id():
        """Lấy ID người tiếp theo"""
        dataset_path = 'dataset'
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path, exist_ok=True)
            return 1
        
        persons = os.listdir(dataset_path)
        if not persons:
            return 1
        
        # Lấy ID cao nhất
        max_id = 0
        for person_folder in persons:
            try:
                # Tìm ID từ tên folder (nếu có)
                parts = person_folder.split('_')
                if parts[0].isdigit():
                    person_id = int(parts[0])
                    max_id = max(max_id, person_id)
            except:
                pass
        
        return max_id + 1
    
    @staticmethod
    def count_images_per_person(person_name, dataset_path='dataset'):
        """Đếm số ảnh của một người"""
        person_path = os.path.join(dataset_path, person_name)
        if os.path.exists(person_path):
            return len([f for f in os.listdir(person_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
        return 0
    
    @staticmethod
    def delete_person_folder(person_name, dataset_path='dataset'):
        """Xóa thư mục người"""
        person_path = os.path.join(dataset_path, person_name)
        if os.path.exists(person_path):
            shutil.rmtree(person_path)
            logger.info(f"✓ Xóa thư mục '{person_name}'")
            return True
        return False
    
    @staticmethod
    def get_all_persons(dataset_path='dataset'):
        """Lấy danh sách tất cả người"""
        if not os.path.exists(dataset_path):
            return []
        
        persons = []
        for person_folder in os.listdir(dataset_path):
            person_path = os.path.join(dataset_path, person_folder)
            if os.path.isdir(person_path):
                image_count = len([f for f in os.listdir(person_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
                persons.append({
                    'name': person_folder,
                    'image_count': image_count
                })
        
        return persons
    
    @staticmethod
    def resize_image(image, width=200, height=200):
        """Resize ảnh"""
        return cv2.resize(image, (width, height))
    
    @staticmethod
    def get_timestamp():
        """Lấy timestamp hiện tại"""
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    @staticmethod
    def log_message(message, log_type='INFO'):
        """Ghi log message"""
        timestamp = Utils.get_timestamp()
        log_message = f"[{timestamp}] [{log_type}] {message}"
        
        # Tạo thư mục logs nếu chưa tồn tại
        os.makedirs('logs', exist_ok=True)
        
        # Ghi vào file log
        with open('logs/app.log', 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
        
        logger.info(log_message)
        return log_message
