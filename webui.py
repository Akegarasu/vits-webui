import os
import sys
import torch
from modules.ui import create_ui

from modules.vits_model import refresh_list, init_load_model
from modules.options import cmd_opts

# todo: 批量处理，说话人，inline指定语言cleaner，手动输入symbol，preprocess，


def init():
    print(f"Launching webui with arguments: {' '.join(sys.argv[1:])}")
    if not os.path.exists("outputs"):
        os.mkdir("outputs")
    if not torch.cuda.is_available():
        print("CUDA is not available, using cpu mode...")
    refresh_list()
    init_load_model()


def run():
    init()
    app = create_ui()
    if cmd_opts.server_name:
        server_name = cmd_opts.server_name
    else:
        server_name = "0.0.0.0" if cmd_opts.listen else None

    app.queue(default_enabled=False).launch(
        share=cmd_opts.share,
        server_name=server_name,
        server_port=cmd_opts.port,
        inbrowser=cmd_opts.autolaunch,
        show_api=False
    )


if __name__ == "__main__":
    run()
