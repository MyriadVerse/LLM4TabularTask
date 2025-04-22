import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QTextEdit,
    QPushButton,
    QMessageBox,
)
from PyQt5.QtCore import Qt
import subprocess


class QueryApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Large Language Model for Tabular-related Tasks")
        self.setGeometry(200, 200, 1600, 1200)

        # central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # main layout
        main_layout = QVBoxLayout(central_widget)

        # table selection part
        table_layout = QHBoxLayout()
        table_layout.addWidget(QLabel("Choose target table:"))

        self.table_combo = QComboBox()
        self.table_combo.addItems(["1", "2", "3", "4"])
        table_layout.addWidget(self.table_combo)

        main_layout.addLayout(table_layout)

        # query input part
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
        main_layout.addWidget(QLabel("result for this query:"))

        self.result_output = QTextEdit()
        self.result_output.setReadOnly(True)
        main_layout.addWidget(self.result_output)

        # status bar
        self.statusBar().showMessage("Ready")

    def execute_query(self):
        table = self.table_combo.currentText()
        query = self.query_input.toPlainText().strip()

        if not query:
            QMessageBox.critical(self, "Wrong", "Query can not be empry!")
            return

        self.statusBar().showMessage("executing query...")
        QApplication.processEvents()  # update the status bar

        try:
            # 调用外部脚本
            process = subprocess.Popen(
                [sys.executable, "process_query.py", table, query],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # 实时获取输出
            while True:
                output = process.stdout.readline()
                error = process.stderr.readline()

                if output == "" and error == "" and process.poll() is not None:
                    break

                if output:
                    self.result_output.append(output.strip())
                if error:
                    self.result_output.append(f"Error: {error.strip()}")

            return_code = process.poll()
            if return_code != 0:
                self.result_output.append(f"\nError, return code: {return_code}")

            self.statusBar().showMessage("Done", 3000)

        except Exception as e:
            self.result_output.append(f"Error: {str(e)}")
            self.statusBar().showMessage("Error", 3000)

    def clear_inputs(self):
        self.query_input.clear()
        self.result_output.clear()
        self.statusBar().showMessage("Clear", 2000)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    #  Set the application style(optional)
    app.setStyle("Fusion")

    window = QueryApp()
    window.show()

    sys.exit(app.exec_())
