"""
sidebar.py
=====================================
左侧聊天记录栏
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QFrame,
)


class Sidebar(QWidget):

    # 新建聊天
    new_chat_clicked = Signal()

    # 切换聊天
    conversation_changed = Signal(int)

    def __init__(self):
        super().__init__()

        self.setObjectName("Sidebar")

        self.init_ui()

    # ----------------------------------------------------

    def init_ui(self):

        layout = QVBoxLayout(self)

        layout.setContentsMargins(15, 15, 15, 15)

        layout.setSpacing(12)

        # =====================================
        # Logo
        # =====================================

        title = QLabel("🤖 Transformer Chat")

        title.setObjectName("SidebarTitle")

        layout.addWidget(title)

        # 分割线

        line = QFrame()

        line.setFrameShape(QFrame.HLine)

        line.setObjectName("SidebarLine")

        layout.addWidget(line)

        # =====================================
        # 新建聊天
        # =====================================

        self.new_chat_btn = QPushButton("+  新建聊天")

        self.new_chat_btn.setObjectName("NewChatButton")

        layout.addWidget(self.new_chat_btn)

        # =====================================
        # 聊天记录
        # =====================================

        self.history = QListWidget()

        self.history.setObjectName("HistoryList")

        layout.addWidget(self.history)

        # =====================================
        # 底部
        # =====================================

        version = QLabel("Transformer v1.0")

        version.setAlignment(Qt.AlignCenter)

        version.setObjectName("VersionLabel")

        layout.addWidget(version)

        # =====================================
        # 信号
        # =====================================

        self.new_chat_btn.clicked.connect(
            self.new_chat_clicked
        )

        self.history.currentRowChanged.connect(
            self.conversation_changed
        )

        self.history.currentRowChanged.connect(
            self.conversation_changed.emit
        )

    # ----------------------------------------------------

    def add_conversation(self, title):

        item = QListWidgetItem(title)

        self.history.addItem(item)

    # ----------------------------------------------------

    def remove_current(self):

        row = self.history.currentRow()

        if row >= 0:

            self.history.takeItem(row)

    # ----------------------------------------------------

    def rename_current(self, title):

        item = self.history.currentItem()

        if item:

            item.setText(title)

    # ----------------------------------------------------

    def current_index(self):

        return self.history.currentRow()

    # ----------------------------------------------------

    def current_title(self):

        item = self.history.currentItem()

        if item:

            return item.text()

        return ""
    
    def select(self, index):
        self.history.setCurrentRow(index)

    def update_current_title(self, title):

        row = self.history.currentRow()

        if row < 0:
            return

        self.history.item(row).setText(title)