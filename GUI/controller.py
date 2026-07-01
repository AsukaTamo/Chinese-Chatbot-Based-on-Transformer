"""
controller.py
=========================================
整个GUI的大脑
"""

from PySide6.QtCore import QObject
from GUI.conversation_manager import ConversationManager

class ChatController(QObject):

    def __init__(
        self,
        header,
        sidebar,
        chat_view,
        input_panel,
        setting_panel,
        worker,
    ):
        super().__init__()

        self.header = header
        self.sidebar = sidebar
        self.chat_view = chat_view
        self.input_panel = input_panel
        self.setting_panel = setting_panel
        self.worker = worker
        self.manager = ConversationManager()
        self.init_connect()

        self.sidebar.conversation_changed.connect(
            self.switch_conversation
        )

        # 初始化参数
        self.update_settings(
            {
                "mode": "beam",
                "temperature": 1.0,
                "beam_size": 4,
                "top_k": 50,
                "top_p": 0.9,
            }
        )

    # ===================================================

    def init_connect(self):

        # 输入
        self.input_panel.sendClicked.connect(
            self.send_message
        )

        # Worker
        self.worker.started.connect(
            self.on_started
        )

        self.worker.finished.connect(
            self.on_finished
        )

        self.worker.error.connect(
            self.on_error
        )

        # 参数变化
        self.setting_panel.settings_changed.connect(
            self.update_settings
        )

        # 新建聊天
        self.sidebar.new_chat_clicked.connect(
            self.new_chat
        )

    # ===================================================

    def send_message(self, text):

        if not text.strip():
            return
        
        conv = self.manager.current()

        # 没有会话：弹提示 + 直接返回
        if conv is None:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(
                None,
                "提示",
                "请先点击左侧「新建聊天」再开始对话"
            )
            return

        # 用户消息
        self.manager.current().messages.append(
            ("user", text)
        )

        self.chat_view.add_user_message(text)

        # 第一条消息作为标题
        conversation = self.manager.current()

        if len(conversation.messages) == 1:

            title = text.replace("\n", " ")[:18]

            conversation.title = title

            self.sidebar.update_current_title(title)

        # Header
        self.header.set_thinking()

        # 输入框锁定
        self.input_panel.set_enabled(False)

        # Thinking
        self.chat_view.show_thinking()

        # Worker开始
        self.worker.generate(text)


    # ===================================================

    def on_started(self):

        pass

    # ===================================================

    def on_finished(self, reply, elapsed):

        self.chat_view.hide_thinking()

        self.chat_view.typewriter(reply)

        self.manager.current().messages.append(
            ("bot", reply)
        )

        self.header.set_ready()

        self.header.set_inference_time(elapsed)

        self.input_panel.set_enabled(True)

    # ===================================================

    def on_error(self, error):

        self.chat_view.hide_thinking()

        self.chat_view.add_bot_message(
            f"发生错误：\n\n{error}"
        )

        self.header.set_error()

        self.input_panel.set_enabled(True)

    # ===================================================

    def update_settings(self, settings):

        self.worker.set_decode_mode(
            settings["mode"]
        )

        self.worker.update_config(
            temperature=settings["temperature"],
            beam_size=settings["beam_size"],
            top_k=settings["top_k"],
            top_p=settings["top_p"],
        )

        self.header.set_decode_mode(
            settings["mode"]
        )

    # ===================================================

    def new_chat(self):

        index = self.manager.new_conversation()

        self.chat_view.clear()

        self.sidebar.add_conversation(

        self.manager.current().title

        )

    def switch_conversation(self, index):

        self.manager.switch(index)

        self.chat_view.clear()

        conversation = self.manager.current()

        for role, text in conversation.messages:

            if role == "user":
                self.chat_view.add_user_message(text)

            else:
                self.chat_view.add_bot_message(text)