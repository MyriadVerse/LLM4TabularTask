import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QComboBox, QTextEdit, QPushButton, QMessageBox)
from PyQt5.QtCore import Qt
import subprocess

class QueryApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("数据查询工具 - PyQt5版")
        self.setGeometry(100, 100, 800, 600)
        
        # 中心窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 表格选择部分
        table_layout = QHBoxLayout()
        table_layout.addWidget(QLabel("选择表格:"))
        
        self.table_combo = QComboBox()
        self.table_combo.addItems(["用户表", "订单表", "产品表", "日志表"])
        table_layout.addWidget(self.table_combo)
        
        main_layout.addLayout(table_layout)
        
        # 查询输入部分
        main_layout.addWidget(QLabel("输入查询:"))
        
        self.query_input = QTextEdit()
        self.query_input.setPlaceholderText("在此输入您的查询语句...")
        main_layout.addWidget(self.query_input)
        
        # 按钮部分
        button_layout = QHBoxLayout()
        
        self.execute_btn = QPushButton("执行查询")
        self.execute_btn.clicked.connect(self.execute_query)
        button_layout.addWidget(self.execute_btn)
        
        self.clear_btn = QPushButton("清除")
        self.clear_btn.clicked.connect(self.clear_inputs)
        button_layout.addWidget(self.clear_btn)
        
        main_layout.addLayout(button_layout)
        
        # 结果展示部分
        main_layout.addWidget(QLabel("查询结果:"))
        
        self.result_output = QTextEdit()
        self.result_output.setReadOnly(True)
        main_layout.addWidget(self.result_output)
        
        # 状态栏
        self.statusBar().showMessage("准备就绪")
    
    def execute_query(self):
        table = self.table_combo.currentText()
        query = self.query_input.toPlainText().strip()
        
        if not query:
            QMessageBox.critical(self, "错误", "查询不能为空!")
            return
        
        self.statusBar().showMessage("正在执行查询...")
        QApplication.processEvents()  # 更新UI
        
        try:
            # 调用外部脚本
            process = subprocess.Popen(
                [sys.executable, "process_query.py", table, query],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 实时获取输出
            while True:
                output = process.stdout.readline()
                error = process.stderr.readline()
                
                if output == '' and error == '' and process.poll() is not None:
                    break
                
                if output:
                    self.result_output.append(output.strip())
                if error:
                    self.result_output.append(f"错误: {error.strip()}")
            
            return_code = process.poll()
            if return_code != 0:
                self.result_output.append(f"\n查询执行失败，返回码: {return_code}")
            
            self.statusBar().showMessage("查询完成", 3000)
            
        except Exception as e:
            self.result_output.append(f"发生错误: {str(e)}")
            self.statusBar().showMessage("查询出错", 3000)
    
    def clear_inputs(self):
        self.query_input.clear()
        self.result_output.clear()
        self.statusBar().showMessage("已清除输入", 2000)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 设置应用程序样式 (可选)
    app.setStyle('Fusion')
    
    window = QueryApp()
    window.show()
    
    sys.exit(app.exec_())