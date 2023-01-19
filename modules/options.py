import argparse
import json
import os
import gradio as gr

from modules.localization import localization_files

config_file = "config.json"

parser = argparse.ArgumentParser()
parser.add_argument("--listen", action='store_true', help="launch gradio with 0.0.0.0 as server name")
parser.add_argument("--port", type=int, help="launch gradio with given server port, defaults to 8860 if available", default="8860")
parser.add_argument("--share", action='store_true', help="use share=True for gradio and make the UI accessible through their site")
parser.add_argument("--server-name", type=str, help="sets hostname of server", default=None)
parser.add_argument("--autolaunch", action='store_true', help="open the webui URL in the system's default browser upon launch", default=False)
parser.add_argument("--device-id", type=str, help="select the default CUDA device to use", default=None)
parser.add_argument("--cpu", action='store_true', help="use cpu")
parser.add_argument("--disable-safe-unpickle", action='store_true', help="disable safe unpickle")
parser.add_argument("--freeze-settings", action='store_true', help="freeze settings")

cmd_opts = parser.parse_args()


class OptionInfo:
    def __init__(self, default=None, label="", component=None, component_args=None, onchange=None, section=None, refresh=None):
        self.default = default
        self.label = label
        self.component = component
        self.component_args = component_args
        self.onchange = onchange
        self.section = section
        self.refresh = refresh


options_templates = {}

options_templates.update({
    # "vits_model": OptionInfo(None, "VITS checkpoint", gr.Dropdown, lambda: {"choices": list_checkpoint_tiles()}, refresh=refresh_checkpoints),
})

options_templates.update({
    "localization": OptionInfo("None", "Localization (requires restart)", gr.Dropdown, lambda: {"choices": ["None"] + localization_files}),
})


class Options:
    data = None
    data_labels = options_templates
    type_map = {int: float}

    def __init__(self):
        self.data = {k: v.default for k, v in self.data_labels.items()}

    def __setattr__(self, key, value):
        if self.data is not None:
            if key in self.data or key in self.data_labels:
                assert not cmd_opts.freeze_settings, "changing settings is disabled"

                info = opts.data_labels.get(key, None)
                comp_args = info.component_args if info else None
                if isinstance(comp_args, dict) and comp_args.get('visible', True) is False:
                    raise RuntimeError(f"not possible to set {key} because it is restricted")

                self.data[key] = value
                return

        return super(Options, self).__setattr__(key, value)

    def __getattr__(self, item):
        if self.data is not None:
            if item in self.data:
                return self.data[item]

        if item in self.data_labels:
            return self.data_labels[item].default

        return super(Options, self).__getattribute__(item)

    def set(self, key, value):
        oldval = self.data.get(key, None)
        if oldval == value:
            return False

        try:
            setattr(self, key, value)
        except RuntimeError:
            return False

        if self.data_labels[key].onchange is not None:
            try:
                self.data_labels[key].onchange()
            except Exception as e:
                print(e)
                print(f"Error when handling onchange event: changing setting {key} to {value}")
                setattr(self, key, oldval)
                return False

        return True

    def save(self, filename=config_file):
        assert not cmd_opts.freeze_settings, "saving settings is disabled"

        with open(filename, "w", encoding="utf8") as file:
            json.dump(self.data, file, indent=4, ensure_ascii=False)

    def same_type(self, x, y):
        if x is None or y is None:
            return True

        type_x = self.type_map.get(type(x), type(x))
        type_y = self.type_map.get(type(y), type(y))

        return type_x == type_y

    def load(self, filename):
        with open(filename, "r", encoding="utf8") as file:
            self.data = json.load(file)

        bad_settings = 0
        for k, v in self.data.items():
            info = self.data_labels.get(k, None)
            if info is not None and not self.same_type(info.default, v):
                print(f"Warning: bad setting value: {k}: {v} ({type(v).__name__}; expected {type(info.default).__name__})")
                bad_settings += 1

        if bad_settings > 0:
            print(f"The program is likely to not work with bad settings.\nSettings file: {filename}\nEither fix the file, or delete it and restart.")

    def onchange(self, key, func, call=True):
        item = self.data_labels.get(key)
        item.onchange = func

        if call:
            func()

    def dump_json(self):
        d = {k: self.data.get(k, self.data_labels.get(k).default) for k in self.data_labels.keys()}
        return json.dumps(d)

    def add_option(self, key, info):
        self.data_labels[key] = info


opts = Options()
if os.path.exists(config_file):
    opts.load(config_file)
else:
    print("Config not found, generating default config...")
    opts.save()
