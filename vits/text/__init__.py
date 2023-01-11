from . import cleaners


def text_to_sequence(text, symbols, cleaner_names: str, cleaner=None):
    """
    modified t2s: custom symbols and cleaner
    """
    _symbol_to_id = {s: i for i, s in enumerate(symbols)}

    sequence = []
    if cleaner:
        clean_text = cleaner(text)
    else:
        clean_text = _clean_text(text, cleaner_names)
    for symbol in clean_text:
        if symbol not in _symbol_to_id.keys():
            continue
        symbol_id = _symbol_to_id[symbol]
        sequence += [symbol_id]
    return sequence


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception('Unknown cleaner: %s' % name)
        text = cleaner(text)
    return text
