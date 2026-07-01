"""
chat_view.py
========================================
聊天区域
负责管理所有聊天消息
"""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QScrollArea,
)

from GUI.bubble import BubbleWidget


class ChatView(QWidget):

    def __init__(self):
        super().__init__()

        self.setObjectName("ChatView")

        self._thinking_bubble = None

        self.init_ui()

    # ===========================================

    def init_ui(self):

        layout = QVBoxLayout(self)

        layout.setContentsMargins(0, 0, 0, 0)

        # -------------------------------
        # ScrollArea
        # -------------------------------

        self.scroll = QScrollArea()

        self.scroll.setWidgetResizable(True)

        self.scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )

        self.scroll.setFrameShape(
            QScrollArea.NoFrame
        )

        layout.addWidget(self.scroll)

        # -------------------------------
        # Container
        # -------------------------------

        self.container = QWidget()

        self.message_layout = QVBoxLayout(
            self.container
        )

        self.message_layout.setSpacing(12)

        self.message_layout.setContentsMargins(
            20,
            20,
            20,
            20
        )

        self.message_layout.addStretch()

        self.scroll.setWidget(self.container)

    # ===========================================
    # 添加消息
    # ===========================================

    def add_user_message(self, text):

        bubble = BubbleWidget(
            text=text,
            is_user=True
        )

        self.message_layout.insertWidget(
            self.message_layout.count() - 1,
            bubble
        )

        self.scroll_bottom()

    def add_bot_message(self, text):

        bubble = BubbleWidget(
            text=text,
            is_user=False
        )

        self.message_layout.insertWidget(
            self.message_layout.count() - 1,
            bubble
        )

        self.scroll_bottom()

    # ===========================================
    # Thinking
    # ===========================================

    def show_thinking(self):

        if self._thinking_bubble:
            return

        self._thinking_bubble = BubbleWidget(
            "正在思考...",
            is_user=False
        )

        self._thinking_timer = QTimer()

        self._dot_count = 0

        self._thinking_timer.timeout.connect(
            self.update_thinking
        )

        self._thinking_timer.start(400)

        self.message_layout.insertWidget(
            self.message_layout.count() - 1,
            self._thinking_bubble
        )

        self.scroll_bottom()

    def hide_thinking(self):

        if self._thinking_bubble:

            self._thinking_bubble.deleteLater()

            self._thinking_bubble = None
        
        if hasattr(self, "_thinking_timer"):

            self._thinking_timer.stop()

    # ===========================================
    # 清空聊天
    # ===========================================

    def clear(self):

        while self.message_layout.count() > 1:

            item = self.message_layout.takeAt(0)

            widget = item.widget()

            if widget:

                widget.deleteLater()

    # ===========================================
    # 自动滚动
    # ===========================================

    def scroll_bottom(self):

        QTimer.singleShot(
            0,
            lambda: self.scroll.verticalScrollBar().setValue(
                self.scroll.verticalScrollBar().maximum()
            )
        )

    def update_thinking(self):

        if self._thinking_bubble is None:
            return

        self._dot_count = (self._dot_count + 1) % 4

        dots = "." * self._dot_count

        self._thinking_bubble.set_text(
            f"正在思考{dots}"
        )

    def typewriter(self, text):

        self._typing_text = text

        self._typing_index = 0

        self._typing_bubble = BubbleWidget(
            "",
            is_user=False
        )

        self.message_layout.insertWidget(
            self.message_layout.count() - 1,
            self._typing_bubble
        )

        self.scroll_bottom()

        self._typing_timer = QTimer()

        self._typing_timer.timeout.connect(
            self.type_next_character
        )

        self._typing_timer.start(20)

    def type_next_character(self):

        if self._typing_index >= len(self._typing_text):

            self._typing_timer.stop()

            return

        self._typing_bubble.append_text(
            self._typing_text[self._typing_index]
        )

        self._typing_index += 1

        self.scroll_bottom()