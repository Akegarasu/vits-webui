function gradioApp() {
    const gradioShadowRoot = document.getElementsByTagName('gradio-app')[0].shadowRoot
    return !!gradioShadowRoot ? gradioShadowRoot : document;
}

function get_uiCurrentTab() {
    return gradioApp().querySelector('.tabs button:not(.border-transparent)')
}

function get_uiCurrentTabContent() {
    return gradioApp().querySelector('.tabitem[id^=tab_]:not([style*="display: none"])')
}

document.addEventListener('keydown', function (e) {
    let handled = false;
    if (e.key !== undefined) {
        if ((e.key === "Enter" && (e.metaKey || e.ctrlKey || e.altKey))) handled = true;
    } else if (e.keyCode !== undefined) {
        if ((e.keyCode === 13 && (e.metaKey || e.ctrlKey || e.altKey))) handled = true;
    }
    if (handled) {
        let button = get_uiCurrentTabContent().querySelector('button[id$=_generate]');
        if (button) {
            button.click();
        }
        e.preventDefault();
    }
})
