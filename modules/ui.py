import os.path

import gradio as gr

import modules.vits_model as vits_model
from modules.process import text2speech
from modules.utils import open_folder
from modules.vits_model import get_model_list, refresh_list

refresh_symbol = "\U0001f504"  # 🔄
folder_symbol = '\U0001f4c2'  # 📂

component_dict = {}
_gradio_template_response_orig = gr.routes.templates.TemplateResponse
script_path = "scripts"


class ToolButton(gr.Button, gr.components.FormComponent):
    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"


def create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_id):
    def refresh():
        refresh_method()
        args = refreshed_args() if callable(refreshed_args) else refreshed_args

        for k, v in args.items():
            setattr(refresh_component, k, v)

        return gr.update(**(args or {}))

    refresh_button = ToolButton(value=refresh_symbol, elem_id=elem_id)
    refresh_button.click(
        fn=refresh,
        inputs=[],
        outputs=[refresh_component]
    )
    return refresh_button


def change_model(model_name):
    vits_model.load_model(model_name)
    speakers = vits_model.curr_vits_model.speakers
    return gr.update(choices=speakers, value=speakers[0])


def create_ui():
    css = "style.css"
    reload_javascript()
    curr_model_list = get_model_list()
    speakers = vits_model.curr_vits_model.speakers
    with gr.Blocks(analytics_enabled=False) as txt2img_interface:
        with gr.Row(elem_id="toprow"):
            with gr.Column(scale=6):
                with gr.Row():
                    with gr.Column(scale=80):
                        with gr.Row():
                            input_text = gr.Textbox(label="Text", show_label=False, lines=4,
                                                    placeholder="Text (press Ctrl+Enter or Alt+Enter to generate)")
            with gr.Column(scale=1):
                with gr.Row():
                    submit = gr.Button("Generate", elem_id=f"vits_generate", variant="primary")

        with gr.Row().style(equal_height=False):
            with gr.Column(variant="panel", elem_id="vits_settings"):
                with gr.Row():
                    model_picker = gr.Dropdown(label="VITS Checkpoint", choices=curr_model_list,
                                               value=vits_model.curr_vits_model.model_name)
                    create_refresh_button(model_picker, refresh_method=refresh_list,
                                          refreshed_args=lambda: {"choices": get_model_list()},
                                          elem_id="vits-model-refresh")
                with gr.Row():
                    process_method = gr.Radio(label="Process Method",
                                              choices=["Simple", "Batch Process", "Multi Speakers"],
                                              value="Simple")

                with gr.Row():
                    speaker_index = gr.Dropdown(label="Speakers",
                                                choices=speakers, value=speakers[0])

                    speed = gr.Slider(value=1, minimum=0.5, maximum=2, step=0.1,
                                      elem_id=f"vits_speed",
                                      label="Speed")

            with gr.Column(variant="panel", elem_id="vits_output"):
                tts_output1 = gr.Textbox(label="Output Message")
                tts_output2 = gr.Audio(label="Output Audio", elem_id=f"vits_audio")

                with gr.Column():
                    with gr.Row(elem_id=f"functional_buttons"):
                        open_folder_button = gr.Button(f"{folder_symbol} Open Folder", elem_id=f'open_vits_folder')
                        save_button = gr.Button('Save', elem_id=f'save')

                        open_folder_button.click(fn=lambda: open_folder("outputs/vits"))

        model_picker.change(
            fn=change_model,
            inputs=[model_picker],
            outputs=[speaker_index]
        )

        submit.click(
            fn=text2speech,
            inputs=[
                input_text,
                speaker_index,
                speed,
                process_method
            ],
            outputs=[
                tts_output1,
                tts_output2
            ]
        )

    with gr.Blocks(analytics_enabled=False) as sovits_interface:
        gr.Markdown("# 请等待更新~")

    interfaces = [
        (txt2img_interface, "VITS", "vits"),
        (sovits_interface, "SO-VITS (dev)", "sovits")
    ]

    with gr.Blocks(css=css, analytics_enabled=False, title="VITS") as demo:
        with gr.Tabs(elem_id="tabs") as tabs:
            for interface, label, ifid in interfaces:
                with gr.TabItem(label, id=ifid, elem_id="tab_" + ifid):
                    interface.render()

    return demo


def reload_javascript():
    scripts_list = [os.path.join(script_path, i) for i in os.listdir(script_path) if i.endswith(".js")]
    with open("script.js", "r", encoding="utf8") as jsfile:
        javascript = f'<script>{jsfile.read()}</script>'

    for path in scripts_list:
        with open(path, "r", encoding="utf8") as jsfile:
            javascript += f"\n<script>{jsfile.read()}</script>"

    # todo: localization
    with open("localizations/zh_cn.json", "r", encoding="utf8") as lf:
        localization_file = lf.read()
    javascript += f"\n<script>var localization={localization_file}</script>"

    # todo: theme
    # if cmd_opts.theme is not None:
    #     javascript += f"\n<script>set_theme('{cmd_opts.theme}');</script>\n"

    def template_response(*args, **kwargs):
        res = _gradio_template_response_orig(*args, **kwargs)
        res.body = res.body.replace(
            b'</head>', f'{javascript}</head>'.encode("utf8"))
        res.init_headers()
        return res

    gr.routes.templates.TemplateResponse = template_response
