"""
worker.py
=========================================
后台推理线程（V2）
"""

import time

from PySide6.QtCore import QThread, Signal

class ChatWorker(QThread):
    """
    后台推理线程
    """

    # 推理开始
    started = Signal()

    # 回复 + 推理耗时(ms)
    finished = Signal(str, float)

    # 错误
    error = Signal(str)

    # 预留流式输出
    stream_update = Signal(str)

    def __init__(self):
        super().__init__()

        self.chatbot = None

        self.message = ""

        # 默认Beam Search
        self.decode_mode = "beam"

    def set_chatbot(self, chatbot):

        self.chatbot = chatbot

    def set_decode_mode(self, mode):

        self.decode_mode = mode

    def update_config(
        self,
        temperature,
        beam_size,
        top_k,
        top_p,
    ):

        if self.chatbot is None:
            return

        config = self.chatbot.config

        config.temperature = temperature
        config.beam_size = beam_size
        config.top_k = top_k
        config.top_p = top_p

    def generate(self, text):

        if self.isRunning():
            return

        self.message = text

        self.start()

    def run(self):

        self.started.emit()

        if self.chatbot is None:

            self.error.emit(
                "ChatBot 未初始化"
            )

            return

        try:

            start = time.perf_counter()

            if self.decode_mode == "beam":

                reply = self.chatbot.reply(

                    self.message,

                    use_beam=True,

                    use_sample=False

                )

            elif self.decode_mode == "sample":

                reply = self.chatbot.reply(

                    self.message,

                    use_beam=False,

                    use_sample=True

                )

            else:

                reply = self.chatbot.reply(

                    self.message,

                    use_beam=False,

                    use_sample=False

                )

            elapsed = (

                time.perf_counter()

                - start

            ) * 1000

            self.finished.emit(

                reply,

                elapsed

            )

        except Exception as e:

            self.error.emit(

                str(e)

            )