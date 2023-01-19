import os

localization_dir = "localizations"
localization_files = os.listdir(localization_dir)


def gen_localization_js(name):
    if name not in localization_files:
        print(f"Load localization file {name} failed. Try set another localization file in settings panel.")
        return ""
    with open(os.path.join("localizations", name), "r", encoding="utf8") as lf:
        localization_file = lf.read()
    js = f"\n<script>var localization={localization_file}</script>"
    return js
