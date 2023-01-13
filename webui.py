# 転がる岩、君に朝が降る
# Code with Love by @Akegarasu

import os
import sys
import modules.safe
from modules.ui import create_ui

from modules.vits_model import refresh_list, init_load_model
from modules.options import cmd_opts


def init():
    print(f"Launching webui with arguments: {' '.join(sys.argv[1:])}")
    ensure_output_dirs()
    refresh_list()
    init_load_model()


def ensure_output_dirs():
    def check_and_create(p):
        if not os.path.exists(p):
            os.makedirs(p)

    check_and_create("outputs/vits")
    check_and_create("outputs/vits-batch")


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
