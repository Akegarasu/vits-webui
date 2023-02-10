import os.path

import gradio as gr

import modules.sovits_model as sovits_model
import modules.vits_model as vits_model
from modules.localization import gen_localization_js
from modules.options import opts
from modules.process import text2speech, sovits_process
from modules.utils import open_folder
from modules.vits_model import get_model_list, refresh_list

refresh_symbol = "\U0001f504"  # 🔄
folder_symbol = '\U0001f4c2'  # 📂

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


def create_setting_component(key):
    def fun():
        return opts.data[key] if key in opts.data else opts.data_labels[key].default

    info = opts.data_labels[key]
    t = type(info.default)

    args = info.component_args() if callable(info.component_args) else info.component_args

    if info.component is not None:
        comp = info.component
    elif t == str:
        comp = gr.Textbox
    elif t == int:
        comp = gr.Number
    elif t == bool:
        comp = gr.Checkbox
    else:
        raise Exception(f'bad options item type: {str(t)} for key {key}')

    elem_id = "setting_" + key

    if info.refresh is not None:
        with gr.Row():
            res = comp(label=info.label, value=fun(), elem_id=elem_id, **(args or {}))
            create_refresh_button(res, info.refresh, info.component_args, "refresh_" + key)
    else:
        res = comp(label=info.label, value=fun(), elem_id=elem_id, **(args or {}))

    return res


def change_vits_model(model_name):
    vits_model.load_model(model_name)
    speakers = vits_model.get_speakers()
    return gr.update(choices=speakers, value=speakers[0])


def change_sovits_model(model_name):
    sovits_model.load_model(model_name)
    speakers = sovits_model.get_speakers()
    return gr.update(choices=speakers, value=speakers[0])


def create_ui():
    css = "style.css"
    component_dict = {}
    reload_javascript()

    vits_model_list = vits_model.get_model_list()
    vits_speakers = vits_model.get_speakers()

    sovits_model_list = sovits_model.get_model_list()
    sovits_speakers = sovits_model.get_speakers()

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
                    vits_submit_btn = gr.Button("Generate", elem_id=f"vits_generate", variant="primary")

        with gr.Row().style(equal_height=False):
            with gr.Column(variant="panel", elem_id="vits_settings"):
                with gr.Row():
                    vits_model_picker = gr.Dropdown(label="VITS Checkpoint", choices=vits_model_list,
                                                    value=vits_model.get_model_name())
                    create_refresh_button(vits_model_picker, refresh_method=vits_model.refresh_list(),
                                          refreshed_args=lambda: {"choices": vits_model.get_model_list()},
                                          elem_id="vits_model_refresh")
                with gr.Row():
                    process_method = gr.Radio(label="Process Method",
                                              choices=["Simple", "Batch Process", "Multi Speakers"],
                                              value="Simple")

                with gr.Row():
                    speaker_index = gr.Dropdown(label="Speakers",
                                                choices=vits_speakers, value=vits_speakers[0])

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

        vits_model_picker.change(
            fn=change_vits_model,
            inputs=[vits_model_picker],
            outputs=[speaker_index]
        )

        vits_submit_btn.click(
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
        with gr.Row():
            with gr.Column(scale=6, elem_id="sovits_audio_panel"):
                sovits_audio_input = gr.File(label="Upload Audio File", elem_id=f"sovits_input_audio")
            with gr.Column(scale=1):
                with gr.Row():
                    sovits_submit_btn = gr.Button("Generate", elem_id=f"sovits_generate", variant="primary")

        with gr.Row().style(equal_height=False):
            with gr.Column(variant="panel", elem_id="sovits_settings"):
                with gr.Row():
                    sovits_model_picker = gr.Dropdown(label="SO-VITS Checkpoint", choices=sovits_model_list,
                                                      value=sovits_model.get_model_name())
                    create_refresh_button(sovits_model_picker, refresh_method=sovits_model.refresh_list(),
                                          refreshed_args=lambda: {"choices": sovits_model.get_model_list()},
                                          elem_id="sovits_model_refresh")

                with gr.Row():
                    sovits_speaker_index = gr.Dropdown(label="Speakers",
                                                       choices=sovits_speakers, value=sovits_speakers[0])

                with gr.Row():
                    vc_transform = gr.Slider(value=0, minimum=-20, maximum=20, step=1,
                                             elem_id=f"vc_transform",
                                             label="VC Transform")
                    slice_db = gr.Slider(value=-40, minimum=-100, maximum=0, step=5,
                                         elem_id=f"slice_db",
                                         label="Slice db")
            with gr.Column(variant="panel", elem_id="sovits_output"):
                sovits_output1 = gr.Textbox(label="Output Message")
                sovits_output2 = gr.Audio(label="Output Audio", elem_id=f"sovits_output_audio")

        sovits_submit_btn.click(
            fn=sovits_process,
            inputs=[sovits_audio_input, sovits_speaker_index, vc_transform, slice_db],
            outputs=[sovits_output1, sovits_output2]
        )

        sovits_model_picker.change(
            fn=change_sovits_model,
            inputs=[sovits_model_picker],
            outputs=[sovits_speaker_index]
        )

    with gr.Blocks(analytics_enabled=False) as settings_interface:
        settings_component = []

        def run_settings(*args):
            changed = []

            for key, value, comp in zip(opts.data_labels.keys(), args, settings_component):
                assert opts.same_type(value, opts.data_labels[key].default), f"Bad value for setting {key}: {value}; expecting {type(opts.data_labels[key].default).__name__}"

            for key, value, comp in zip(opts.data_labels.keys(), args, settings_component):
                if opts.set(key, value):
                    changed.append(key)

            try:
                opts.save()
            except RuntimeError:
                return f'{len(changed)} settings changed without save: {", ".join(changed)}.'
            return f'{len(changed)} settings changed{": " if len(changed) > 0 else ""}{", ".join(changed)}.'

        with gr.Row():
            with gr.Column(scale=6):
                settings_submit = gr.Button(value="Apply settings", variant='primary', elem_id="settings_submit")
            with gr.Column():
                restart_gradio = gr.Button(value='Reload UI', variant='primary', elem_id="settings_restart_gradio")

        settings_result = gr.HTML(elem_id="settings_result")

        for i, (k, item) in enumerate(opts.data_labels.items()):
            component = create_setting_component(k)
            component_dict[k] = component
            settings_component.append(component)

        settings_submit.click(
            fn=run_settings,
            inputs=settings_component,
            outputs=[settings_result],
        )

    interfaces = [
        (txt2img_interface, "VITS", "vits"),
        (sovits_interface, "SO-VITS (dev)", "sovits"),
        (settings_interface, "Settings", "settings")
    ]

    with gr.Blocks(css=css, analytics_enabled=False, title="VITS") as demo:
        with gr.Tabs(elem_id="tabs") as tabs:
            for interface, label, ifid in interfaces:
                with gr.TabItem(label, id=ifid, elem_id="tab_" + ifid):
                    interface.render()

        component_keys = [k for k in opts.data_labels.keys() if k in component_dict]

        # def get_settings_values():
        #     return [getattr(opts, key) for key in component_keys]
        #
        # demo.load(
        #     fn=get_settings_values,
        #     inputs=[],
        #     outputs=[component_dict[k] for k in component_keys],
        # )

    return demo


def reload_javascript():
    scripts_list = [os.path.join(script_path, i) for i in os.listdir(script_path) if i.endswith(".js")]
    with open("script.js", "r", encoding="utf8") as jsfile:
        javascript = f'<script>{jsfile.read()}</script>'

    for path in scripts_list:
        with open(path, "r", encoding="utf8") as jsfile:
            javascript += f"\n<script>{jsfile.read()}</script>"

    javascript += gen_localization_js(opts.localization)

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
