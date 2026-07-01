import sys

from PySide6.QtWidgets import QApplication

from inference import ChatBot
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

    chatbot = ChatBot(
        config.model_save_dir + "/best_model.pt",
        config
    )

    window = MainWindow(chatbot)

    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()