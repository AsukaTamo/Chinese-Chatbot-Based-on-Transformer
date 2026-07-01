"""
loading_window.py
========================================
启动加载窗口

负责：
1. 显示 Logo
2. 显示加载状态
3. 初始化 ChatBot
4. 加载完成后进入 MainWindow
"""

import os

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QProgressBar,
)

from config import Config
from inference import ChatBot


# ==========================================
# 后台加载模型
# ==========================================

class LoaderThread(QThread):

    loaded = Signal(object)

    failed = Signal(str)

    progress = Signal(str)

    def run(self):

        try:

            self.progress.emit("正在读取配置...")

            config = Config()

            if (
                config.device == "cuda"
                and not __import__("torch").cuda.is_available()
            ):
                config.device = "cpu"

            self.progress.emit("正在加载模型...")

            model_path = os.path.join(
                config.model_save_dir,
                "best_model.pt"
            )

            chatbot = ChatBot(
                model_path,
                config
            )

            self.progress.emit("初始化完成")

            self.loaded.emit(chatbot)

        except Exception as e:

            self.failed.emit(str(e))


# ==========================================
# Loading Window
# ==========================================

class LoadingWindow(QWidget):

    chatbot_loaded = Signal(object)

    def __init__(self):

        super().__init__()

        self.setWindowTitle("Transformer ChatBot")

        self.setFixedSize(500, 320)

        self.init_ui()

        self.start_loading()

    # ---------------------------------------

    def init_ui(self):

        layout = QVBoxLayout(self)

        layout.setContentsMargins(40, 40, 40, 40)

        layout.setSpacing(25)

        # Logo

        self.logo = QLabel("🤖")

        self.logo.setAlignment(Qt.AlignCenter)

        self.logo.setStyleSheet("""
            font-size:64px;
        """)

        layout.addWidget(self.logo)

        # 标题

        self.title = QLabel(
            "Transformer 中文聊天机器人"
        )

        self.title.setAlignment(Qt.AlignCenter)

        self.title.setStyleSheet("""
            font-size:22px;
            font-weight:bold;
        """)

        layout.addWidget(self.title)

        # 状态

        self.status = QLabel("准备启动...")

        self.status.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.status)

        # Progress

        self.progress = QProgressBar()

        self.progress.setRange(0, 0)

        layout.addWidget(self.progress)

    # ---------------------------------------

    def start_loading(self):

        self.worker = LoaderThread()

        self.worker.progress.connect(
            self.status.setText
        )

        self.worker.failed.connect(
            self.load_failed
        )

        self.worker.loaded.connect(
            self.load_finished
        )

        self.worker.start()

    # ---------------------------------------

    def load_finished(self, chatbot):

        self.chatbot_loaded.emit(chatbot)

        self.close()

    # ---------------------------------------

    def load_failed(self, msg):

        self.progress.setRange(0, 1)

        self.progress.setValue(1)

        self.status.setText("加载失败")

        print(msg)