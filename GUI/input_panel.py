"""
input_panel.py
========================================
聊天输入面板
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QTextEdit,
    QPushButton,
)


class MessageEdit(QTextEdit):
    """
    自定义输入框

    Enter：发送
    Shift + Enter：换行
    """

    sendPressed = Signal()

    def __init__(self):
        super().__init__()

        self.setPlaceholderText("输入消息，Enter发送，Shift+Enter换行...")

        self.setMinimumHeight(56)

        self.setMaximumHeight(150)

        self.setAcceptRichText(False)

    # ------------------------------------------

    def keyPressEvent(self, event):

        # Enter

        if (
            event.key() in (Qt.Key_Return, Qt.Key_Enter)
            and not (event.modifiers() & Qt.ShiftModifier)
        ):

            self.sendPressed.emit()

            return

        super().keyPressEvent(event)


class InputPanel(QWidget):

    """
    底部输入区域
    """

    sendClicked = Signal(str)

    def __init__(self):
        super().__init__()

        self.setObjectName("InputPanel")

        self.init_ui()

    # ==========================================

    def init_ui(self):

        layout = QHBoxLayout(self)

        layout.setContentsMargins(18, 15, 18, 15)

        layout.setSpacing(12)

        # --------------------------------------
        # 输入框
        # --------------------------------------

        self.editor = MessageEdit()

        layout.addWidget(self.editor)

        # --------------------------------------
        # 发送按钮
        # --------------------------------------

        self.send_btn = QPushButton("➤")

        self.send_btn.setObjectName("SendButton")

        self.send_btn.setFixedSize(48, 48)

        layout.addWidget(self.send_btn)

        # --------------------------------------
        # 信号
        # --------------------------------------

        self.send_btn.clicked.connect(
            self.send_message
        )

        self.editor.sendPressed.connect(
            self.send_message
        )

    # ==========================================

    def send_message(self):

        text = self.editor.toPlainText().strip()

        if text == "":
            return

        self.sendClicked.emit(text)

        self.editor.clear()

        self.editor.moveCursor(
            QTextCursor.Start
        )

        self.editor.setFocus()

    # ==========================================

    def set_enabled(self, enabled: bool):

        self.editor.setEnabled(enabled)

        self.send_btn.setEnabled(enabled)

    # ==========================================

    def clear(self):

        self.editor.clear()

    # ==========================================

    def text(self):

        return self.editor.toPlainText()

    # ==========================================

    def set_text(self, text):

        self.editor.setPlainText(text)

        self.editor.moveCursor(
            QTextCursor.End
        )