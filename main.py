import sys
import cv2
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QComboBox, QSpinBox, QTableWidget, 
                             QTableWidgetItem, QFileDialog, QMessageBox, QTabWidget, QTextEdit)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer
from face_detector import FaceDetector
from database import DatabaseManager
from utils import Utils
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CameraThread(QThread):
    frame_signal = pyqtSignal(object)
    faces_signal = pyqtSignal(list)
    
    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        self.cap = None
        self.running = False
    
    def run(self):
        self.cap = cv2.VideoCapture(0)
        self.running = True
        
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Phát hiện khuôn mặt
                faces = self.detector.detect_faces(frame)
                frame = self.detector.draw_faces(frame, faces)
                
                self.frame_signal.emit(frame)
                self.faces_signal.emit(faces)
            
            cv2.waitKey(1)
    
    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

class FaceRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.detector = FaceDetector()
        self.db = DatabaseManager()
        self.camera_thread = None
        self.label_dict = {}
        
        # Kết nối database
        if not self.db.connect():
            QMessageBox.critical(self, "Lỗi", "Không thể kết nối MySQL!")
            sys.exit(1)
        
        self.db.create_tables()
        
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("Ứng dụng Nhận diện Khuôn mặt")
        self.setGeometry(100, 100, 1200, 800)
        
        # Tab widget
        tabs = QTabWidget()
        
        # Tab 1: Camera & Recognition
        tab1 = self.create_recognition_tab()
        tabs.addTab(tab1, "Nhận diện Khuôn mặt")
        
        # Tab 2: Database Management
        tab2 = self.create_database_tab()
        tabs.addTab(tab2, "Quản lý Dữ liệu")
        
        # Tab 3: Recognition History
        tab3 = self.create_history_tab()
        tabs.addTab(tab3, "Lịch sử Nhận diện")
        
        self.setCentralWidget(tabs)
    
    def create_recognition_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Camera display
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("border: 2px solid black;")
        layout.addWidget(self.camera_label)
        
        # Info display
        self.info_label = QLabel("Sẵn sàng...")
        self.info_label.setFont(QFont("Arial", 12))
        layout.addWidget(self.info_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.start_camera_btn = QPushButton("Bắt đầu Camera")
        self.start_camera_btn.clicked.connect(self.start_camera)
        button_layout.addWidget(self.start_camera_btn)
        
        self.stop_camera_btn = QPushButton("Dừng Camera")
        self.stop_camera_btn.clicked.connect(self.stop_camera)
        self.stop_camera_btn.setEnabled(False)
        button_layout.addWidget(self.stop_camera_btn)
        
        layout.addLayout(button_layout)
        
        widget.setLayout(layout)
        return widget
    
    def create_database_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Table widget
        self.persons_table = QTableWidget()
        self.persons_table.setColumnCount(6)
        self.persons_table.setHorizontalHeaderLabels(["ID", "Tên", "Tuổi", "Giới tính", "Email", "Điện thoại"])
        self.persons_table.setColumnWidth(0, 50)
        self.persons_table.setColumnWidth(1, 150)
        layout.addWidget(self.persons_table)
        
        # Button layout
        btn_layout = QHBoxLayout()
        
        refresh_btn = QPushButton("Làm mới")
        refresh_btn.clicked.connect(self.refresh_persons_table)
        btn_layout.addWidget(refresh_btn)
        
        delete_btn = QPushButton("Xóa")
        delete_btn.clicked.connect(self.delete_person)
        btn_layout.addWidget(delete_btn)
        
        layout.addLayout(btn_layout)
        
        # Add person layout
        add_layout = QVBoxLayout()
        add_layout.addWidget(QLabel("Thêm người mới:"))
        
        form_layout = QHBoxLayout()
        
        self.name_input = QLabel()
        form_layout.addWidget(QLabel("Tên:"))
        form_layout.addWidget(self.name_input)
        
        form_layout.addWidget(QLabel("ID:"))
        self.person_id_input = QSpinBox()
        self.person_id_input.setValue(Utils.get_next_person_id())
        form_layout.addWidget(self.person_id_input)
        
        add_btn = QPushButton("Thêm")
        add_btn.clicked.connect(self.add_person_to_db)
        form_layout.addWidget(add_btn)
        
        add_layout.addLayout(form_layout)
        layout.addLayout(add_layout)
        
        widget.setLayout(layout)
        return widget
    
    def create_history_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        # History table
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(4)
        self.history_table.setHorizontalHeaderLabels(["Người", "Thời gian", "Độ tin cậy", "Địa điểm"])
        layout.addWidget(self.history_table)
        
        # Button layout
        btn_layout = QHBoxLayout()
        
        refresh_history_btn = QPushButton("Làm mới")
        refresh_history_btn.clicked.connect(self.refresh_history_table)
        btn_layout.addWidget(refresh_history_btn)
        
        layout.addLayout(btn_layout)
        
        widget.setLayout(layout)
        return widget
    
    def start_camera(self):
        self.camera_thread = CameraThread(self.detector)
        self.camera_thread.frame_signal.connect(self.update_camera_frame)
        self.camera_thread.faces_signal.connect(self.process_faces)
        self.camera_thread.start()
        
        self.start_camera_btn.setEnabled(False)
        self.stop_camera_btn.setEnabled(True)
        self.info_label.setText("Camera đang chạy...")
    
    def stop_camera(self):
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread.wait()
        
        self.start_camera_btn.setEnabled(True)
        self.stop_camera_btn.setEnabled(False)
        self.info_label.setText("Camera đã dừng")
    
    def update_camera_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = rgb_frame.shape
        bytes_per_line = 3 * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaledToWidth(640)
        self.camera_label.setPixmap(scaled_pixmap)
    
    def process_faces(self, faces):
        if len(faces) > 0:
            self.info_label.setText(f"Phát hiện {len(faces)} khuôn mặt")
    
    def refresh_persons_table(self):
        persons = self.db.get_all_persons()
        self.persons_table.setRowCount(len(persons) if persons else 0)
        
        if persons:
            for row, person in enumerate(persons):
                self.persons_table.setItem(row, 0, QTableWidgetItem(str(person['person_id'])))
                self.persons_table.setItem(row, 1, QTableWidgetItem(person['name']))
                self.persons_table.setItem(row, 2, QTableWidgetItem(str(person['age'] or '')))
                self.persons_table.setItem(row, 3, QTableWidgetItem(person['gender'] or ''))
                self.persons_table.setItem(row, 4, QTableWidgetItem(person['email'] or ''))
                self.persons_table.setItem(row, 5, QTableWidgetItem(person['phone'] or ''))
    
    def add_person_to_db(self):
        person_id = self.person_id_input.value()
        name = self.name_input.text()
        
        if not name:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng nhập tên!")
            return
        
        if self.db.add_person(person_id, name):
            QMessageBox.information(self, "Thành công", f"Đã thêm '{name}'!")
            self.refresh_persons_table()
            self.name_input.setText("")
            self.person_id_input.setValue(Utils.get_next_person_id())
    
    def delete_person(self):
        current_row = self.persons_table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn người cần xóa!")
            return
        
        person_id = int(self.persons_table.item(current_row, 0).text())
        person_name = self.persons_table.item(current_row, 1).text()
        
        reply = QMessageBox.question(self, "Xác nhận", f"Xóa '{person_name}'?", 
                                     QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            if self.db.delete_person(person_id):
                QMessageBox.information(self, "Thành công", "Đã xóa!")
                self.refresh_persons_table()
    
    def refresh_history_table(self):
        logs = self.db.get_recognition_logs(limit=100)
        self.history_table.setRowCount(len(logs) if logs else 0)
        
        if logs:
            for row, log in enumerate(logs):
                person = self.db.get_person(log['person_id'])
                person_name = person['name'] if person else "Không rõ"
                
                self.history_table.setItem(row, 0, QTableWidgetItem(person_name))
                self.history_table.setItem(row, 1, QTableWidgetItem(str(log['recognized_at'])))
                self.history_table.setItem(row, 2, QTableWidgetItem(f"{log['confidence']:.2f}%"))
                self.history_table.setItem(row, 3, QTableWidgetItem(log['location'] or ''))
    
    def closeEvent(self, event):
        self.stop_camera()
        self.db.close()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()