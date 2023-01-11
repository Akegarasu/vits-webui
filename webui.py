import os
from modules.ui import create_ui

from modules.vits_model import refresh_list, init_load_model


# todo: 批量处理，说话人，inline指定语言cleaner，手动输入symbol，preprocess，

def init():
    if not os.path.exists("outputs"):
        os.mkdir("outputs")
    refresh_list()
    init_load_model()


def run():
    init()
    app = create_ui()
    app.queue(concurrency_count=3).launch(show_api=False)


if __name__ == "__main__":
    run()
