"""
header.py
=========================================
顶部状态栏
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
)


class Header(QWidget):

    def __init__(self):
        super().__init__()

        self.setObjectName("Header")

        self.init_ui()

    # ======================================================

    def init_ui(self):

        layout = QHBoxLayout(self)

        layout.setContentsMargins(20, 12, 20, 12)

        layout.setSpacing(15)

        # -----------------------------
        # 左侧：标题
        # -----------------------------

        self.title = QLabel("🤖 Transformer ChatBot")

        self.title.setObjectName("HeaderTitle")

        layout.addWidget(self.title)

        layout.addStretch()

        # -----------------------------
        # 中间：模型信息
        # -----------------------------

        info_layout = QVBoxLayout()

        info_layout.setSpacing(2)

        self.mode_label = QLabel("Beam Search")

        self.mode_label.setObjectName("HeaderInfo")

        self.time_label = QLabel("推理时间：-- ms")

        self.time_label.setObjectName("HeaderInfo")

        info_layout.addWidget(self.mode_label)

        info_layout.addWidget(self.time_label)

        layout.addLayout(info_layout)

        layout.addSpacing(25)

        # -----------------------------
        # 右侧：状态
        # -----------------------------

        self.status_label = QLabel("🟢 Ready")

        self.status_label.setObjectName("StatusLabel")

        layout.addWidget(self.status_label)

    # ======================================================
    # 状态接口
    # ======================================================

    def set_ready(self):

        self.status_label.setText("🟢 Ready")

    def set_thinking(self):

        self.status_label.setText("🟡 Thinking...")

    def set_error(self):

        self.status_label.setText("🔴 Error")

    # ======================================================
    # 解码方式
    # ======================================================

    def set_decode_mode(self, mode: str):

        mode_map = {
            "beam": "Beam Search",
            "greedy": "Greedy",
            "sample": "Sampling"
        }

        self.mode_label.setText(
            mode_map.get(mode, mode)
        )

    # ======================================================
    # 推理耗时
    # ======================================================

    def set_inference_time(self, ms: float):

        self.time_label.setText(
            f"推理时间：{ms:.0f} ms"
        )

    # ======================================================
    # 标题（以后可改模型名）
    # ======================================================

    def set_title(self, title: str):

        self.title.setText(title)