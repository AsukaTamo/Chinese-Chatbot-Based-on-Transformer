"""
bubble.py
========================================
聊天气泡组件
"""

from datetime import datetime
from PySide6.QtCore import QPropertyAnimation
from PySide6.QtWidgets import QGraphicsOpacityEffect
from shiboken6 import isValid
from PySide6.QtGui import QPixmap
from PySide6.QtGui import QRegion
from PySide6.QtCore import QRect, QPoint, QSize

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QFrame,
    QSizePolicy,
)


class BubbleWidget(QWidget):
    """
    聊天气泡

    Parameters
    ----------
    text : str
        消息内容

    is_user : bool
        True 用户
        False AI
    """

    def __init__(self, text="", is_user=False):

        super().__init__()

        self.is_user = is_user

        self.message = text

        self.init_ui()

        self.start_animation()

        self.message_label.setAttribute(Qt.WA_TranslucentBackground)

        self.message_label.setAutoFillBackground(False)

        self.message_label.setStyleSheet("""
        QLabel{
            border:none;
            background-color: rgba(0,0,0,0);
            color:white;
        }
        """)

        self._destroyed = False


    # ===================================================

    def init_ui(self):

        root = QHBoxLayout(self)

        root.setContentsMargins(15, 8, 15, 8)

        root.setSpacing(6)

        # -----------------------------
        # 头像
        # -----------------------------

        self.avatar = QLabel()

        self.avatar.setFixedSize(42, 42)

        self.avatar.setAlignment(Qt.AlignCenter)

        if self.is_user:

            pix = QPixmap("GUI/assets/user.png")

        else:

            pix = QPixmap("GUI/assets/robot.jpg")

        self.avatar.setPixmap(
            pix.scaled(
                42,
                42,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

        region = QRegion(QRect(0, 0, 42, 42), QRegion.Ellipse)
        self.avatar.setMask(region)

        # -----------------------------
        # 消息区域
        # -----------------------------

        self.content = QWidget()

        content_layout = QVBoxLayout(self.content)

        content_layout.setContentsMargins(0, 0, 0, 0)

        content_layout.setSpacing(5)

        # 名称

        self.name = QLabel()

        if self.is_user:

            self.name.setText("You")

        else:

            self.name.setText("Transformer")

        self.name.setObjectName("BubbleName")

        if self.is_user:
            self.name.setAlignment(Qt.AlignRight)
        else:
            self.name.setAlignment(Qt.AlignLeft)

        content_layout.addWidget(self.name)
        # -----------------------------
        # 气泡
        # -----------------------------

        self.bubble = QFrame()

        if self.is_user:

            self.bubble.setObjectName("UserBubble")

        else:

            self.bubble.setObjectName("BotBubble")

        bubble_layout = QVBoxLayout(self.bubble)

        bubble_layout.setContentsMargins(
            18,
            12,
            18,
            12
        )

        # 消息

        self.message_label = QLabel(self.message)

        self.message_label.setWordWrap(True)

        self.message_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )

        self.message_label.setSizePolicy(
            QSizePolicy.Preferred,
            QSizePolicy.Preferred
        )

        bubble_layout.addWidget(
            self.message_label
        )

        content_layout.addWidget(self.bubble)

        # 时间

        self.time = QLabel()

        self.time.setObjectName("BubbleTime")

        self.time.setText(
            datetime.now().strftime("%H:%M")
        )

        if self.is_user:
            self.time.setAlignment(Qt.AlignRight)
        else:
            self.time.setAlignment(Qt.AlignLeft)

        content_layout.addWidget(
            self.time
        )

        # 最大宽度

        self.content.setMaximumWidth(650)

        # -----------------------------
        # 左右布局
        # -----------------------------

        if self.is_user:

            root.addStretch()

            root.addWidget(self.content)

            root.addWidget(self.avatar)

        else:

            root.addWidget(self.avatar)

            root.addWidget(self.content)

            root.addStretch()

    # ===================================================

    def set_text(self, text):
        if not isValid(self) or not isValid(self.message_label):
            return

        self.message = text

        self.message_label.setText(text)

    # ===================================================

    def append_text(self, text):
        if not isValid(self) or not isValid(self.message_label):
            return

        self.message += text

        try:
            self.message_label.setText(self.message)
        except RuntimeError:
            # 如果组件已经被销毁，直接忽略，不再更新
            pass

    # ===================================================

    def text(self):

        return self.message
    
    def start_animation(self):

        effect = QGraphicsOpacityEffect(self)

        self.setGraphicsEffect(effect)

        self.animation = QPropertyAnimation(effect, b"opacity")

        self.animation.setDuration(250)

        self.animation.setStartValue(0)

        self.animation.setEndValue(1)

        self.animation.start()
    
    def deleteLater(self):
        self._destroyed = True
        super().deleteLater()
    
    