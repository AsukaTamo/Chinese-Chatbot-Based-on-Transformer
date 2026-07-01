import sys

from PySide6.QtWidgets import QApplication

from inference import ChatBot, GPTChatBot, QwenChatBot
from config import Config
from GUI.main_window import MainWindow


def load_qss(app):
    with open("GUI/style.qss", "r", encoding="utf-8") as f:
        app.setStyleSheet(f.read())


def main():

    app = QApplication(sys.argv)

    # 加载QSS
    load_qss(app)

    config = Config()

    model_path = config.best_model_path

    if config.model_type == "gpt":
        chatbot = GPTChatBot(model_path, config)
    elif config.model_type == "qwen":
        chatbot = QwenChatBot(model_path, config, pretrained=True)
    else:
        chatbot = ChatBot(model_path, config)

    window = MainWindow(chatbot)

    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
