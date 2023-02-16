# 転がる岩、君に朝が降る
# Code with Love by @Akegarasu

import os
import sys

# must import before other modules load model.
import modules.safe
import modules.path

from modules.ui import create_ui
import modules.vits_model as vits_model
import modules.sovits_model as sovits_model
from modules.options import cmd_opts


def init():
    print(f"Launching webui with arguments: {' '.join(sys.argv[1:])}")
    ensure_output_dirs()
    vits_model.refresh_list()
    sovits_model.refresh_list()
    # if cmd_opts.ui_debug_mode:
    #     return


def ensure_output_dirs():
    def check_and_create(p):
        if not os.path.exists(p):
            os.makedirs(p)

    check_and_create("outputs/vits")
    check_and_create("outputs/vits-batch")
    check_and_create("outputs/sovits")
    check_and_create("outputs/sovits-batch")
    check_and_create("temp")


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
