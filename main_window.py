import base64
import pickle
import subprocess
import sys
import os
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QComboBox, QTextEdit, QPushButton, QMessageBox,
                             QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

from PyQt5.QtCore import QCoreApplication

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtGui import QFont

def enable_high_dpi_support():
    if sys.platform == 'win32':  # 在 Windows 上启用 DPI 缩放
        QGuiApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    elif sys.platform == 'darwin':  # 在 macOS 上启用 Retina 显示支持
        QGuiApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

def adjust_scaling():
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)  # 在高 DPI 显示器上启用更高的像素图

def set_font():
    font = QFont()
    font.setFamily('Arial')  # 设置字体
    font.setPointSize(12)    # 设置字号
    QApplication.setFont(font)


class QueryApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Large Language Model for Tabular-related Tasks")
        self.setGeometry(200, 200, 1500, 800)  # 增加窗口大小以适应预览

        # central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # main layout
        main_layout = QVBoxLayout(central_widget)

        # task selection part
        task_layout = QHBoxLayout()
        task_layout.addWidget(QLabel("Choose Tabular-related Task:"))

        self.task_combo = QComboBox()
        self.task_combo.addItems(["Column Type Annotation", "Hybrid Question Answering"])
        task_layout.addWidget(self.task_combo)

        self.task_combo.currentIndexChanged.connect(self.update_task_settings)


        main_layout.addLayout(task_layout)

        # Table selection part
        table_layout = QHBoxLayout()
        table_layout.addWidget(QLabel("Select Target Table:"))

        self.table_path_label = QLabel("No file selected")
        table_layout.addWidget(self.table_path_label)

        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.browse_file)
        table_layout.addWidget(self.browse_btn)

        main_layout.addLayout(table_layout)

        # Table preview
        self.preview_label = QLabel("Table Preview:")
        main_layout.addWidget(self.preview_label)

        self.table_preview = QTableWidget()
        self.table_preview.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table_preview.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_preview.setFixedHeight(200)
        main_layout.addWidget(self.table_preview)

        # query part
        main_layout.addWidget(QLabel("Query:"))
        query_layout = QHBoxLayout()

        self.query_input = QTextEdit()
        # self.query_input.setPlaceholderText("Input your query here...")
        self.query_input.setMaximumHeight(400)  # 设置最大高度为 80
        query_layout.addWidget(self.query_input)

        # Add a QLabel to show the image
        self.image_label = QLabel()
        # self.pixmap = QPixmap("./instruction.png")  # 使用你实际的图片路径
        # self.image_label.setPixmap(self.pixmap.scaled(500, 500, Qt.KeepAspectRatio))  # 调整图片大小
        self.update_image_and_font()
        query_layout.addWidget(self.image_label)

        main_layout.addLayout(query_layout)

        self.update_task_settings()  # 初始调用以设置默认的提示文本


        # button layout
        button_layout = QHBoxLayout()

        self.execute_btn = QPushButton("Execute the query")
        self.execute_btn.clicked.connect(self.execute_query)
        button_layout.addWidget(self.execute_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_inputs)
        button_layout.addWidget(self.clear_btn)

        main_layout.addLayout(button_layout)

        # result output part
        main_layout.addWidget(QLabel("Result for this query:"))

        self.result_output = QTextEdit()
        self.result_output.setReadOnly(True)
        main_layout.addWidget(self.result_output)

        # status bar
        self.statusBar().showMessage("Ready")
        
        # Store the loaded dataframe
        self.df = None

    def update_task_settings(self):
        """Update the query input font and image based on the selected task."""
        self.update_image_and_font()
    def update_image_and_font(self):
        """Update the image and font based on the selected task."""
        task = self.task_combo.currentText()

        if task == "Column Type Annotation":
            # Change font to a more formal style for this task
            # font = QFont("Times New Roman", 12)
            # self.query_input.setFont(font)
            self.query_input.setPlaceholderText("e.g. What are the correct column types for column Player")


            # Load a different image for this task
            self.pixmap = QPixmap("instruction.png")
            self.image_label.setPixmap(self.pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        elif task == "Hybrid Question Answering":
            # Change font to a more modern style for this task
            # font = QFont("Arial", 12)
            # self.query_input.setFont(font)
            self.query_input.setPlaceholderText("e.g. How many dollars are the basic research in 2016 and 2018")


            # Load a different image for this task
            self.pixmap = QPixmap("qa.png")
            self.image_label.setPixmap(self.pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # Optionally, you can add a custom alignment for the image:
        self.image_label.setAlignment(Qt.AlignCenter)

    def browse_file(self):
        """Open file dialog to select CSV file and display preview"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", "", 
            "CSV Files (*.csv);;All Files (*)", 
            options=options
        )
        
        if file_path:
            self.table_path_label.setText(os.path.basename(file_path))
            self.statusBar().showMessage(f"Loaded: {file_path}")
            
            try:
                # Read CSV file
                self.file_path = file_path
                self.df = pd.read_csv(file_path)
                
                # Display preview
                self.show_preview(self.df)
                
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load file:\n{str(e)}")
                self.table_path_label.setText("Error loading file")

    def show_preview(self, df):
        self.table_preview.clear()
        
        # 设置表格行列数
        preview_rows = min(5, len(df))
        self.table_preview.setRowCount(preview_rows)
        self.table_preview.setColumnCount(len(df.columns))
        
        # 设置表头
        self.table_preview.setHorizontalHeaderLabels(df.columns)
        
        # 填充表格数据
        for i in range(preview_rows):
            for j, col in enumerate(df.columns):
                item = QTableWidgetItem(str(df.iloc[i, j]))
                self.table_preview.setItem(i, j, item)
        
        # 调整列宽
        self.table_preview.resizeColumnsToContents()
        
        # 让所有列宽自适应整个表格
        self.table_preview.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # 可选：限制第一列的最大宽度，避免其占据过多空间
        first_column_width = self.table_preview.columnWidth(0)
        self.table_preview.setColumnWidth(0, min(first_column_width, 250))  # 设置最大宽度为 250


    def execute_query(self):
        """Execute the query by calling the external process"""
        if self.df is None:
            QMessageBox.warning(self, "Warning", "Please select a CSV file first!")
            return
            
        query = self.query_input.toPlainText().strip()
        if not query:
            QMessageBox.warning(self, "Warning", "Query cannot be empty!")
            return
            
        self.statusBar().showMessage("Executing query...")
        QApplication.processEvents()  # 更新UI
        
        try:
            task = self.task_combo.currentText()
            query = self.query_input.toPlainText().strip()
            
            table_bytes = pickle.dumps(self.df)
            table_str = base64.b64encode(table_bytes).decode('utf-8')
            
            # 调用外部脚本
            process = subprocess.Popen(
                [sys.executable, "./process_table_query.py", task, table_str, query],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 实时获取输出
            output, error = process.communicate()
            
            if error:
                self.result_output.append(f"Error: {error.strip()}")
            
            if output:
                self.result_output.setPlainText(output.strip())
            
            return_code = process.returncode
            if return_code != 0:
                self.result_output.append(f"\nQuery execution failed with return code: {return_code}")
            
            self.statusBar().showMessage("Query completed", 3000)
            
        except Exception as e:
            self.result_output.append(f"Error occurred: {str(e)}")
            self.statusBar().showMessage("Query failed", 3000)

    def clear_inputs(self):
        """Clear all inputs and results"""
        self.query_input.clear()
        self.result_output.clear()
        self.statusBar().showMessage("Inputs cleared", 2000)



def apply_modern_bright_style(app):
    modern_bright_style = """
    QMainWindow {
        background-color: #F5F5F5;  /* Light gray background */
    }

    QLabel {
        color: #333333;  /* Dark gray text */
        font-family: 'Arial', sans-serif;
        font-size: 14px;
    }

    QComboBox, QTextEdit, QPushButton {
        background-color: #FFFFFF;
        border: 1px solid #CCCCCC;  /* Light border */
        color: #333333;  /* Dark text */
        font-family: 'Arial', sans-serif;
        font-size: 14px;
    }

    QPushButton {
        border-radius: 5px;
        padding: 8px 16px;
        background-color: #007BFF;  /* Blue button */
        color: white;
        font-weight: bold;
    }

    QPushButton:hover {
        background-color: #0056b3;  /* Darker blue on hover */
        color: white;
    }

    QTextEdit {
        background-color: #FFFFFF;
        color: #333333;
        border: 1px solid #CCCCCC;
        font-family: 'Arial', sans-serif;
        font-size: 14px;
        padding: 10px;
    }

    QTableWidget {
        background-color: #FFFFFF;
        color: #333333;
        border: 1px solid #CCCCCC;
    }

    QTableWidget::item {
        padding: 8px;
        border: 1px solid #DDDDDD;
    }

    QTableWidget::item:selected {
        background-color: #007BFF;
        color: white;
    }

    QStatusBar {
        background-color: #F5F5F5;
        color: #333333;
    }

    QScrollBar {
        background-color: #F5F5F5;
        border: 1px solid #CCCCCC;
    }
    """
    
    app.setStyleSheet(modern_bright_style)


if __name__ == "__main__":
    adjust_scaling()
    set_font()  # 设置字体

    enable_high_dpi_support()  # 启用高 DPI 支持
    app = QApplication(sys.argv)
    apply_modern_bright_style(app) 
    window = QueryApp()
    window.show()
    sys.exit(app.exec_())