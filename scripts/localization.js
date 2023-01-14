// localization = {} -- the dict with translations is created by the backend

ignore_ids_for_localization = {}

re_num = /^[\.\d]+$/
re_emoji = /[\p{Extended_Pictographic}\u{1F3FB}-\u{1F3FF}\u{1F9B0}-\u{1F9B3}]/u

original_lines = {}
translated_lines = {}

function textNodesUnder(el) {
    var n, a = [], walk = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, null, false);
    while (n = walk.nextNode()) a.push(n);
    return a;
}

function canBeTranslated(node, text) {
    if (!text) return false;
    if (!node.parentElement) return false;

    parentType = node.parentElement.nodeName
    if (parentType == 'SCRIPT' || parentType == 'STYLE' || parentType == 'TEXTAREA') return false;

    if (parentType == 'OPTION' || parentType == 'SPAN') {
        pnode = node
        for (var level = 0; level < 4; level++) {
            pnode = pnode.parentElement
            if (!pnode) break;

            if (ignore_ids_for_localization[pnode.id] == parentType) return false;
        }
    }

    if (re_num.test(text)) return false;
    if (re_emoji.test(text)) return false;
    return true
}

function getTranslation(text) {
    if (!text) return undefined

    if (translated_lines[text] === undefined) {
        original_lines[text] = 1
    }

    tl = localization[text]
    if (tl !== undefined) {
        translated_lines[tl] = 1
    }

    return tl
}

function processTextNode(node) {
    text = node.textContent.trim()

    if (!canBeTranslated(node, text)) return

    tl = getTranslation(text)
    if (tl !== undefined) {
        node.textContent = tl
    }
}

function processNode(node) {
    if (node.nodeType == 3) {
        processTextNode(node)
        return
    }

    if (node.title) {
        tl = getTranslation(node.title)
        if (tl !== undefined) {
            node.title = tl
        }
    }

    if (node.placeholder) {
        tl = getTranslation(node.placeholder)
        if (tl !== undefined) {
            node.placeholder = tl
        }
    }

    textNodesUnder(node).forEach(function (node) {
        processTextNode(node)
    })
}

onUiUpdate(function (m) {
    m.forEach(function (mutation) {
        mutation.addedNodes.forEach(function (node) {
            processNode(node)
        })
    });
})


document.addEventListener("DOMContentLoaded", function () {
    processNode(gradioApp())

    if (localization.rtl) {  // if the language is from right to left,
        (new MutationObserver((mutations, observer) => { // wait for the style to load
            mutations.forEach(mutation => {
                mutation.addedNodes.forEach(node => {
                    if (node.tagName === 'STYLE') {
                        observer.disconnect();

                        for (const x of node.sheet.rules) {  // find all rtl media rules
                            if (Array.from(x.media || []).includes('rtl')) {
                                x.media.appendMedium('all');  // enable them
                            }
                        }
                    }
                })
            });
        })).observe(gradioApp(), {childList: true});
    }
})
