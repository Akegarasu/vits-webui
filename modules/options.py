import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--listen", action='store_true', help="launch gradio with 0.0.0.0 as server name")
parser.add_argument("--port", type=int, help="launch gradio with given server port, defaults to 8860 if available", default="8860")
parser.add_argument("--share", action='store_true', help="use share=True for gradio and make the UI accessible through their site")
parser.add_argument("--server-name", type=str, help="sets hostname of server", default=None)
parser.add_argument("--autolaunch", action='store_true', help="open the webui URL in the system's default browser upon launch", default=False)
parser.add_argument("--device-id", type=str, help="select the default CUDA device to use", default=None)
parser.add_argument("--cpu", action='store_true', help="use cpu")

cmd_opts = parser.parse_args()
