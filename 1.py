import sys
import os
import pandas as pd
import subprocess
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QTabWidget, QVBoxLayout, QWidget,
    QPushButton, QStatusBar, QHBoxLayout, QLabel
)
from PyQt5.QtCore import QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT


class PlotCanvas(FigureCanvas):
    def __init__(self, data=None, title='', parent=None):
        self.fig, self.ax = Figure(figsize=(5, 4), dpi=100), None
        super().__init__(self.fig)
        self.setParent(parent)
        self.plot(data, title)

    def plot(self, data, title):
        self.ax = self.fig.add_subplot(111)
        self.ax.clear()
        if data is not None:
            self.ax.plot(data)
        self.ax.set_title(title)
        self.draw()


class Worker(QThread):
    update_status = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, folder_path, parent=None):
        super().__init__(parent)
        self.folder_path = folder_path

    def run(self):
        self.update_status.emit("Status: Running analysis...")
        command = ["python", "vibration_analysis.py", "-i", self.folder_path]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            self.update_status.emit("Status: Analysis completed successfully.")
        except subprocess.CalledProcessError as e:
            self.update_status.emit(f"Status: Error: {e.stderr}")
        self.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSV Plot Viewer")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        button_layout = QHBoxLayout()

        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        self.upload_button = QPushButton("Upload Folder")
        self.submit_button = QPushButton("Submit")
        self.submit_button.setEnabled(False)
        self.status_label = QLabel("Status: Waiting for folder upload...")

        button_layout.addWidget(self.upload_button)
        button_layout.addWidget(self.submit_button)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.status_label)
        central_widget.setLayout(main_layout)

        self.upload_button.clicked.connect(self.upload_folder)
        self.submit_button.clicked.connect(self.submit_folder)
        self.initialize_tabs()
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

    def initialize_tabs(self):
        plots = {"Sensor1 Fundamental Frequency": None,
                 "Sensor2 Fundamental Frequency": None,
                 "Sensor3 Fundamental Frequency": None,
                 "Sensor4 Fundamental Frequency": None}

        self.tab_widget.clear()
        for title, data in plots.items():
            tab, layout = QWidget(), QVBoxLayout()
            canvas = PlotCanvas(data, title)
            layout.addWidget(NavigationToolbar2QT(canvas, self))
            layout.addWidget(canvas)
            tab.setLayout(layout)
            self.tab_widget.addTab(tab, title)

        self.add_combined_plot_tab()

    def add_combined_plot_tab(self):
        combined_tab, layout = QWidget(), QVBoxLayout()
        self.combined_canvas = PlotCanvas(None, "Combined Sensor Plot")
        layout.addWidget(NavigationToolbar2QT(self.combined_canvas, self))
        layout.addWidget(self.combined_canvas)
        combined_tab.setLayout(layout)
        self.tab_widget.addTab(combined_tab, "Combined Plot")

    def plot_combined_data(self, df):
        if hasattr(self, 'combined_canvas'):
            ax = self.combined_canvas.ax
            ax.clear()
            ax.plot(df['s1_f0'], label='Sensor1', color='blue')
            ax.plot(df['s2_f0'], label='Sensor2', color='green')
            ax.plot(df['s3_f0'], label='Sensor3', color='red')
            ax.plot(df['s4_f0'], label='Sensor4', color='purple')
            ax.set_title("Combined Sensor Plot")
            ax.legend()
            self.combined_canvas.draw()

    def upload_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.status_label.setText(f"Status: Folder '{folder_path}' selected.")
            self.submit_button.setEnabled(True)
            self.selected_folder = folder_path

    def submit_folder(self):
        if not hasattr(self, 'selected_folder') or not self.selected_folder:
            self.status_label.setText("Status: No folder selected.")
            return

        self.status_label.setText("Status: Processing CSV file...")
        csv_files = [f for f in os.listdir(self.selected_folder) if f.endswith('.csv')]
        if not csv_files:
            self.status_label.setText("Status: No CSV files found.")
            return

        self.submit_button.setEnabled(False)
        self.worker = Worker(self.selected_folder)
        self.worker.update_status.connect(self.update_status)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()
import sys
import os
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QTabWidget, QVBoxLayout, QWidget,
    QPushButton, QStatusBar, QHBoxLayout, QLabel
)
from PyQt5.QtCore import QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

class Worker(QThread):
    update_status = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path

    def run(self):
        csv_files = [f for f in os.listdir(self.folder_path) if f.endswith('.csv')]
        if not csv_files:
            self.update_status.emit("No CSV files found.")
            self.finished.emit()
            return
        
        for csv_file in csv_files:
            file_path = os.path.join(self.folder_path, csv_file)
            try:
                df = pd.read_csv(file_path)
                print(f"Processing: {csv_file}")
                self.update_status.emit(f"Processed: {csv_file}")
            except Exception as e:
                self.update_status.emit(f"Error processing {csv_file}: {str(e)}")
        
        self.update_status.emit("Processing completed.")
        self.finished.emit()

class PlotCanvas(FigureCanvas):
    def __init__(self, data=None, title='', parent=None):
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.plot(data, title)

    def plot(self, data, title):
        self.ax.clear()
        if data is not None and not data.empty:
            self.ax.plot(data, marker='o', linestyle='-', label=title)  # Added label here
            self.ax.set_title(title)
            self.ax.set_xlabel("Index")
            self.ax.set_ylabel("Value")
            self.ax.legend()  # Added legend to show the label
        else:
            self.ax.text(0.5, 0.5, "No Data Available", ha='center', va='center', fontsize=12)
        self.draw()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSV Plot Viewer")
        self.setGeometry(100, 100, 800, 600)
        
        self.selected_folder = None  
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        button_layout = QHBoxLayout()
        
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        self.upload_button = QPushButton("Upload Folder")
        self.submit_button = QPushButton("Submit")
        self.submit_button.setEnabled(False)
        self.status_label = QLabel("Status: Waiting for folder upload...")

        button_layout.addWidget(self.upload_button)
        button_layout.addWidget(self.submit_button)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.status_label)
        central_widget.setLayout(main_layout)
        
        self.upload_button.clicked.connect(self.upload_folder)
        self.submit_button.clicked.connect(self.submit_folder)
        
        self.initialize_tabs()
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

    def initialize_tabs(self):
        plots = {
            "Sensor1 Fundamental Frequency": None,
            "Sensor2 Fundamental Frequency": None,
            "Sensor3 Fundamental Frequency": None,
            "Sensor4 Fundamental Frequency": None,   
        }
        
        self.tab_widget.clear()
        for title, data in plots.items():
            tab = QWidget()
            layout = QVBoxLayout()
            canvas = PlotCanvas(data, title)
            layout.addWidget(NavigationToolbar2QT(canvas, self))
            layout.addWidget(canvas)
            tab.setLayout(layout)
            self.tab_widget.addTab(tab, title)

    def add_combined_plot_tab(self):
        combined_tab, layout = QWidget(), QVBoxLayout()
        self.combined_canvas = PlotCanvas(None, "Combined Sensor Plot")
        layout.addWidget(NavigationToolbar2QT(self.combined_canvas, self))
        layout.addWidget(self.combined_canvas)
        combined_tab.setLayout(layout)
        self.tab_widget.addTab(combined_tab, "Combined Plot")

    def plot_combined_data(self, df):
        if hasattr(self, 'combined_canvas'):
            ax = self.combined_canvas.ax
            ax.clear()
            ax.plot(df['s1_f0'], label='Sensor1', color='blue')
            ax.plot(df['s2_f0'], label='Sensor2', color='green')
            ax.plot(df['s3_f0'], label='Sensor3', color='red')
            ax.plot(df['s4_f0'], label='Sensor4', color='purple')
            ax.set_title("Combined Sensor Plot")
            ax.legend()
            self.combined_canvas.draw()

    def upload_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.status_label.setText(f"Status: Folder '{folder_path}' selected.")
            self.submit_button.setEnabled(True)
            self.selected_folder = folder_path

    def submit_folder(self):
        if not self.selected_folder:
            self.status_label.setText("Status: No Folder Selected.")
            return
        self.status_label.setText("Status: Processing CSV files...")
        csv_files = [f for f in os.listdir(self.selected_folder) if f.endswith('.csv')]
        if not csv_files:
            self.status_label.setText("Status: No CSV Files Found.")
            return
        
        self.submit_button.setEnabled(False)
        self.worker = Worker(self.selected_folder)
        self.worker.update_status.connect(self.update_status)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def update_status(self, message):
        self.status_label.setText(f"Status: {message}")

    def on_worker_finished(self):
        self.submit_button.setEnabled(True)
        csv_filename = os.path.join(self.selected_folder, 'results', 'F0_Analysis.csv')
        if not os.path.exists(csv_filename):
            self.status_label.setText("Status: CSV file not found.")
            return
        self.process_csv(csv_filename)

    def process_csv(self, csv_filename):
        columns = ['filename', 's1_f0', 's2_f0', 's3_f0', 's4_f0', 's1_M', 's2_M', 's3_M', 's4_M']
        try:
            df = pd.read_csv(csv_filename, names=columns, skiprows=1, delimiter=',')
        except Exception as e:
            self.status_label.setText(f"Error reading CSV: {str(e)}")
            return
        
        if df.empty:
            self.status_label.setText("Status: No data in CSV file.")
            return

        print("DataFrame loaded successfully:")
        print(df.head())  # Added to show the data structure for debugging
        
        if 's1_f0' not in df.columns or 's2_f0' not in df.columns or 's3_f0' not in df.columns or 's4_f0' not in df.columns:
            self.status_label.setText("Status: Missing expected columns.")
            print("Missing expected columns in the CSV file.")
            return

        plots = {
            "Sensor1 Fundamental Frequency": df['s1_f0'],
            "Sensor2 Fundamental Frequency": df['s2_f0'],
            "Sensor3 Fundamental Frequency": df['s3_f0'],
            "Sensor4 Fundamental Frequency": df['s4_f0']
        }

        self.tab_widget.clear()
        for title, data in plots.items():
            tab, layout = QWidget(), QVBoxLayout()
            canvas = PlotCanvas(data, title)
            layout.addWidget(NavigationToolbar2QT(canvas, self))
            layout.addWidget(canvas)
            tab.setLayout(layout)
            self.tab_widget.addTab(tab, title)

        self.add_combined_plot_tab()
        self.plot_combined_data(df)
        
        self.status_label.setText("Status: Processing completed.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

    def update_status(self, message):
        self.status_label.setText(message)

    def on_worker_finished(self):
        self.submit_button.setEnabled(True)
        csv_filename = os.path.join(self.selected_folder, 'results', 'F0_Analysis.csv')
        if not os.path.exists(csv_filename):
            self.status_label.setText("Status: CSV file not found.")
            return
        self.process_csv(csv_filename)

    def process_csv(self, csv_filename):
        columns = ['filename', 's1_f0', 's2_f0', 's3_f0', 's4_f0', 's1_M', 's2_M', 's3_M', 's4_M']
        df = pd.read_csv(csv_filename, names=columns, skiprows=1, delimiter=',')

        plots = {"Sensor1 Fundamental Frequency": df['s1_f0'],
                 "Sensor2 Fundamental Frequency": df['s2_f0'],
                 "Sensor3 Fundamental Frequency": df['s3_f0'],
                 "Sensor4 Fundamental Frequency": df['s4_f0']}

        self.tab_widget.clear()
        for title, data in plots.items():
            tab, layout = QWidget(), QVBoxLayout()
            canvas = PlotCanvas(data, title)
            layout.addWidget(NavigationToolbar2QT(canvas, self))
            layout.addWidget(canvas)
            tab.setLayout(layout)
            self.tab_widget.addTab(tab, title)

        self.add_combined_plot_tab()
        self.plot_combined_data(df)
        self.status_label.setText("Status: Processing completed.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
