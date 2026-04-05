import mysql.connector
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

class DatabaseManager:
    def __init__(self):
        self.host = os.getenv('DB_HOST', 'localhost')
        self.user = os.getenv('DB_USER', 'root')
        self.password = os.getenv('DB_PASSWORD', '')
        self.database = os.getenv('DB_NAME', 'face_recognition_db')
        self.port = os.getenv('DB_PORT', '3306')
        self.connection = None
        self.cursor = None
    
    def connect(self):
        """Kết nối tới MySQL"""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                port=self.port
            )
            self.cursor = self.connection.cursor(dictionary=True)
            print("✓ Kết nối MySQL thành công!")
            return True
        except mysql.connector.Error as err:
            print(f"✗ Lỗi kết nối MySQL: {err}")
            return False
    
    def create_tables(self):
        """Tạo bảng cơ sở dữ liệu nếu chưa tồn tại"""
        try:
            # Bảng người dùng/khuôn mặt
            create_persons_table = """
            CREATE TABLE IF NOT EXISTS persons (
                id INT AUTO_INCREMENT PRIMARY KEY,
                person_id INT UNIQUE NOT NULL,
                name VARCHAR(255) NOT NULL,
                age INT,
                gender VARCHAR(50),
                email VARCHAR(255),
                phone VARCHAR(20),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
            """
            
            # Bảng ghi nhận nhận diện
            create_recognition_log_table = """
            CREATE TABLE IF NOT EXISTS recognition_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                person_id INT NOT NULL,
                recognized_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                confidence FLOAT,
                location VARCHAR(255),
                FOREIGN KEY (person_id) REFERENCES persons(person_id)
            )
            """
            
            self.cursor.execute(create_persons_table)
            self.cursor.execute(create_recognition_log_table)
            self.connection.commit()
            print("✓ Tạo bảng thành công!")
            return True
        except mysql.connector.Error as err:
            print(f"✗ Lỗi tạo bảng: {err}")
            return False
    
    def add_person(self, person_id, name, age=None, gender=None, email=None, phone=None):
        """Thêm người vào database"""
        try:
            query = """
            INSERT INTO persons (person_id, name, age, gender, email, phone)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            self.cursor.execute(query, (person_id, name, age, gender, email, phone))
            self.connection.commit()
            print(f"✓ Thêm người '{name}' thành công!")
            return True
        except mysql.connector.Error as err:
            print(f"✗ Lỗi thêm người: {err}")
            return False
    
    def get_person(self, person_id):
        """Lấy thông tin người từ person_id"""
        try:
            query = "SELECT * FROM persons WHERE person_id = %s"
            self.cursor.execute(query, (person_id,))
            result = self.cursor.fetchone()
            return result
        except mysql.connector.Error as err:
            print(f"✗ Lỗi lấy người: {err}")
            return None
    
    def get_all_persons(self):
        """Lấy tất cả người trong database"""
        try:
            query = "SELECT * FROM persons"
            self.cursor.execute(query)
            results = self.cursor.fetchall()
            return results
        except mysql.connector.Error as err:
            print(f"✗ Lỗi lấy danh sách người: {err}")
            return None
    
    def log_recognition(self, person_id, confidence, location=None):
        """Ghi nhận lần nhận diện khuôn mặt"""
        try:
            query = """
            INSERT INTO recognition_logs (person_id, confidence, location)
            VALUES (%s, %s, %s)
            """
            self.cursor.execute(query, (person_id, confidence, location))
            self.connection.commit()
            print(f"✓ Ghi nhận nhận diện thành công!")
            return True
        except mysql.connector.Error as err:
            print(f"✗ Lỗi ghi nhận: {err}")
            return False
    
    def get_recognition_logs(self, person_id=None, limit=100):
        """Lấy lịch sử nhận diện"""
        try:
            if person_id:
                query = """
                SELECT * FROM recognition_logs 
                WHERE person_id = %s 
                ORDER BY recognized_at DESC 
                LIMIT %s
                """
                self.cursor.execute(query, (person_id, limit))
            else:
                query = """
                SELECT * FROM recognition_logs 
                ORDER BY recognized_at DESC 
                LIMIT %s
                """
                self.cursor.execute(query, (limit,))
            
            results = self.cursor.fetchall()
            return results
        except mysql.connector.Error as err:
            print(f"✗ Lỗi lấy lịch sử: {err}")
            return None
    
    def update_person(self, person_id, **kwargs):
        """Cập nhật thông tin người"""
        try:
            updates = []
            values = []
            for key, value in kwargs.items():
                if key in ['name', 'age', 'gender', 'email', 'phone']:
                    updates.append(f"{key} = %s")
                    values.append(value)
            
            if not updates:
                return False
            
            values.append(person_id)
            query = f"UPDATE persons SET {', '.join(updates)} WHERE person_id = %s"
            self.cursor.execute(query, values)
            self.connection.commit()
            print(f"✓ Cập nhật thông tin thành công!")
            return True
        except mysql.connector.Error as err:
            print(f"✗ Lỗi cập nhật: {err}")
            return False
    
    def delete_person(self, person_id):
        """Xóa người khỏi database"""
        try:
            # Xóa logs trước
            query_logs = "DELETE FROM recognition_logs WHERE person_id = %s"
            self.cursor.execute(query_logs, (person_id,))
            
            # Xóa người
            query = "DELETE FROM persons WHERE person_id = %s"
            self.cursor.execute(query, (person_id,))
            self.connection.commit()
            print(f"✓ Xóa người thành công!")
            return True
        except mysql.connector.Error as err:
            print(f"✗ Lỗi xóa người: {err}")
            return False
    
    def close(self):
        """Đóng kết nối"""
        if self.connection:
            self.connection.close()
            print("✓ Đóng kết nối MySQL!")
