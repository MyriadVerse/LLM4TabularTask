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

class QueryApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Large Language Model for Tabular-related Tasks")
        self.setGeometry(200, 200, 800, 800)  # 增加窗口大小以适应预览

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
        self.table_preview.setMaximumHeight(200)
        main_layout.addWidget(self.table_preview)

        # query part
        main_layout.addWidget(QLabel("Query:"))

        self.query_input = QTextEdit()
        self.query_input.setPlaceholderText("Input your query here...")
        main_layout.addWidget(self.query_input)

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
        """Display a preview of the dataframe"""
        self.table_preview.clear()
        
        # Set row and column count
        preview_rows = min(5, len(df))
        self.table_preview.setRowCount(preview_rows)
        self.table_preview.setColumnCount(len(df.columns))
        
        # Set headers
        self.table_preview.setHorizontalHeaderLabels(df.columns)
        
        # Fill table with data
        for i in range(preview_rows):
            for j, col in enumerate(df.columns):
                item = QTableWidgetItem(str(df.iloc[i, j]))
                self.table_preview.setItem(i, j, item)
        
        # Resize columns to fit content
        self.table_preview.resizeColumnsToContents()

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
                [sys.executable, "/home/yongkang/work/Course/LLM4TabularTask/process_table_query.py", task, table_str, query],
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QueryApp()
    window.show()
    sys.exit(app.exec_())