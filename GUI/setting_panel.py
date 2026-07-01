"""
setting_panel.py
=====================================
模型参数设置面板
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QGroupBox,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QPushButton,
)


class SettingPanel(QWidget):

    # 参数变化信号
    settings_changed = Signal(dict)

    def __init__(self):
        super().__init__()

        self.setObjectName("SettingPanel")

        self.init_ui()

    # ===================================================

    def init_ui(self):

        layout = QVBoxLayout(self)

        layout.setContentsMargins(15, 15, 15, 15)

        layout.setSpacing(18)

        # -----------------------------------
        # 标题
        # -----------------------------------

        title = QLabel("⚙ 模型参数")

        title.setObjectName("SettingTitle")

        layout.addWidget(title)

        # ===================================
        # 解码方式
        # ===================================

        decode_group = QGroupBox("解码方式")

        decode_layout = QVBoxLayout(decode_group)

        self.decode_mode = QComboBox()

        self.decode_mode.addItems([
            "Beam Search",
            "Greedy",
            "Sampling"
        ])

        decode_layout.addWidget(self.decode_mode)

        layout.addWidget(decode_group)

        # ===================================
        # Temperature
        # ===================================

        temp_group = QGroupBox("Temperature")

        temp_layout = QVBoxLayout(temp_group)

        self.temperature = QDoubleSpinBox()

        self.temperature.setRange(0.1, 2.0)

        self.temperature.setSingleStep(0.1)

        self.temperature.setValue(1.0)

        temp_layout.addWidget(self.temperature)

        layout.addWidget(temp_group)

        # ===================================
        # Beam Size
        # ===================================

        beam_group = QGroupBox("Beam Size")

        beam_layout = QVBoxLayout(beam_group)

        self.beam_size = QSpinBox()

        self.beam_size.setRange(1, 10)

        self.beam_size.setValue(4)

        beam_layout.addWidget(self.beam_size)

        layout.addWidget(beam_group)

        # ===================================
        # Top-K
        # ===================================

        topk_group = QGroupBox("Top-K")

        topk_layout = QVBoxLayout(topk_group)

        self.top_k = QSpinBox()

        self.top_k.setRange(0, 100)

        self.top_k.setValue(50)

        topk_layout.addWidget(self.top_k)

        layout.addWidget(topk_group)

        # ===================================
        # Top-P
        # ===================================

        topp_group = QGroupBox("Top-P")

        topp_layout = QVBoxLayout(topp_group)

        self.top_p = QDoubleSpinBox()

        self.top_p.setRange(0.0, 1.0)

        self.top_p.setSingleStep(0.05)

        self.top_p.setValue(0.9)

        topp_layout.addWidget(self.top_p)

        layout.addWidget(topp_group)

        # ===================================
        # 重置按钮
        # ===================================

        self.reset_btn = QPushButton("恢复默认参数")

        layout.addWidget(self.reset_btn)

        layout.addStretch()

        # ===================================
        # 信号
        # ===================================

        self.decode_mode.currentIndexChanged.connect(
            self.emit_settings
        )

        self.temperature.valueChanged.connect(
            self.emit_settings
        )

        self.beam_size.valueChanged.connect(
            self.emit_settings
        )

        self.top_k.valueChanged.connect(
            self.emit_settings
        )

        self.top_p.valueChanged.connect(
            self.emit_settings
        )

        self.reset_btn.clicked.connect(
            self.reset_default
        )

    # ===================================================

    def emit_settings(self):

        mode = self.decode_mode.currentText()

        if mode == "Beam Search":
            mode = "beam"

        elif mode == "Sampling":
            mode = "sample"

        else:
            mode = "greedy"

        self.settings_changed.emit({

            "mode": mode,

            "temperature": self.temperature.value(),

            "beam_size": self.beam_size.value(),

            "top_k": self.top_k.value(),

            "top_p": self.top_p.value(),

        })

    # ===================================================

    def reset_default(self):

        self.decode_mode.setCurrentIndex(0)

        self.temperature.setValue(1.0)

        self.beam_size.setValue(4)

        self.top_k.setValue(50)

        self.top_p.setValue(0.9)

        self.emit_settings()