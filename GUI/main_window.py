"""
main_window.py
=========================================
主窗口（V2）

职责：
1. 创建所有组件
2. 创建 Worker
3. 创建 Controller
4. 完成布局

除此之外不负责任何业务逻辑
"""

from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSizePolicy,
)

from GUI.header import Header
from GUI.sidebar import Sidebar
from GUI.chat_view import ChatView
from GUI.input_panel import InputPanel
from GUI.setting_panel import SettingPanel

from GUI.controller import ChatController
from GUI.worker import ChatWorker


class MainWindow(QMainWindow):

    def __init__(self, chatbot):
        super().__init__()

        self.chatbot = chatbot

        self.worker = None
        self.controller = None

        self.init_ui()

        self.init_worker()

        self.init_controller()

    # =========================================================

    def init_ui(self):

        self.setWindowTitle("Transformer 中文聊天机器人")

        self.resize(1500, 900)

        self.setMinimumSize(1280, 800)

        central = QWidget()

        self.setCentralWidget(central)

        root = QVBoxLayout(central)

        root.setContentsMargins(0, 0, 0, 0)

        root.setSpacing(0)

        # =====================================
        # Header
        # =====================================

        self.header = Header()

        root.addWidget(self.header)

        # =====================================
        # Body
        # =====================================

        body = QWidget()

        body_layout = QHBoxLayout(body)

        body_layout.setContentsMargins(0, 0, 0, 0)

        body_layout.setSpacing(0)

        root.addWidget(body)

        # -------------------------------------
        # Sidebar
        # -------------------------------------

        self.sidebar = Sidebar()

        self.sidebar.setFixedWidth(260)

        body_layout.addWidget(self.sidebar)

        # -------------------------------------
        # Center
        # -------------------------------------

        center = QWidget()

        center_layout = QVBoxLayout(center)

        center_layout.setContentsMargins(0, 0, 0, 0)

        center_layout.setSpacing(0)

        body_layout.addWidget(center)

        center.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )

        self.chat_view = ChatView()

        center_layout.addWidget(
            self.chat_view,
            1
        )

        self.input_panel = InputPanel()

        center_layout.addWidget(
            self.input_panel
        )

        # -------------------------------------
        # Setting
        # -------------------------------------

        self.setting_panel = SettingPanel()

        self.setting_panel.setFixedWidth(300)

        body_layout.addWidget(
            self.setting_panel
        )

    # =========================================================

    def init_worker(self):

        self.worker = ChatWorker()

        self.worker.set_chatbot(
            self.chatbot
        )

    # =========================================================

    def init_controller(self):

        self.controller = ChatController(

            header=self.header,

            sidebar=self.sidebar,

            chat_view=self.chat_view,

            input_panel=self.input_panel,

            setting_panel=self.setting_panel,

            worker=self.worker

        )