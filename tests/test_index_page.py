# ruff: noqa: E501  (embedded JS fixtures)
"""Mic page (index.html) JS checks run under node; no browser, no models."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from gujusub.cloud.registry import catalog

INDEX_HTML = Path(__file__).parents[1] / "gujusub" / "static" / "index.html"
needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is required")


def _script() -> str:
    match = re.search(r"<script>(.*?)</script>", INDEX_HTML.read_text(), re.DOTALL)
    assert match is not None
    return match.group(1)


def _helpers() -> str:
    src = _script()
    start = src.index("// --- pure engine helpers")
    end = src.index("// --- end pure engine helpers ---")
    return src[start:end]


def _node(script: str) -> str:
    return subprocess.run(
        ["node", "-e", script], check=True, capture_output=True, text=True
    ).stdout


@needs_node
def test_inline_script_parses(tmp_path):
    path = tmp_path / "index.js"
    path.write_text(_script())
    subprocess.run(["node", "--check", str(path)], check=True)


@needs_node
def test_static_catalog_matches_registry():
    out = _node(_helpers() + "\nprocess.stdout.write(JSON.stringify(STATIC_CATALOG));")
    assert json.loads(out) == catalog()


@needs_node
def test_helpers():
    _node(_helpers() + r"""
const assert = require("node:assert/strict");
const byId = Object.fromEntries(STATIC_CATALOG.map(p => [p.id, p]));
assert.deepEqual(backendOptions(STATIC_CATALOG)[0], ["local", "Local models (no internet)"]);
assert.equal(backendOptions(STATIC_CATALOG).length, STATIC_CATALOG.length + 1);
assert.deepEqual(providerFacts(byId.soniox),
  ["Gujarati: yes", "Translation: built in", "Language auto-detect: yes"]);
assert.deepEqual(providerFacts(byId.openai),
  ["Gujarati: partial", "Translation: local (IndicTrans2)", "Language auto-detect: no"]);
assert.equal(keyStatusText(undefined), "No key saved");
assert.equal(keyStatusText({ set: false, region: "eastus" }), "No key saved · region eastus");
assert.equal(keyStatusText({ set: true, hint: "••••wxyz", source: "saved" }), "Key: ••••wxyz (saved)");
assert.equal(keyStatusText({ set: true, hint: "••••abcd", source: "env", env: "SONIOX_API_KEY" }),
  "Key: ••••abcd (from env SONIOX_API_KEY)");
assert.deepEqual(engineStatus({}, byId), { text: "Engine: Local", bad: false });
assert.deepEqual(engineStatus({ backend: "local", cloud: null }, byId), { text: "Engine: Local", bad: false });
assert.deepEqual(engineStatus({ backend: "soniox", cloud: null }, byId), { text: "Engine: Soniox", bad: false });
assert.deepEqual(engineStatus({ backend: "soniox", cloud: { provider: "soniox", state: "connected", error: null } }, byId),
  { text: "Engine: Soniox · connected", bad: false });
assert.deepEqual(engineStatus({ backend: "azure", cloud: { state: "error", error: "bad key" } }, byId),
  { text: "Engine: Azure AI Speech · error: bad key", bad: true });
""")


# Minimal DOM stub: enough for the page to build its panel and handle control messages.
DOM_STUB = r"""
const assert = require("node:assert/strict");
class Node_ {
  constructor(tag) { this.tagName = tag; this.children = []; this.style = {}; this.dataset = {};
    this.listeners = {}; this.hidden = false; this.disabled = false; this.value = "";
    this.textContent = ""; this.options = []; this.className = "";
    this.classList = { add() {}, remove() {}, toggle() {} }; }
  append(...k) { this.children.push(...k); if (this.tagName === "select") this.options.push(...k); }
  appendChild(k) { this.append(k); }
  add(o) { this.append(o); }
  replaceChildren(...k) { this.children = k; if (this.tagName === "select") this.options = k; }
  addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
  querySelector(sel) { const tag = sel.replace(/^\./, "");
    const walk = n => { for (const c of n.children || []) { if (c && (c.tagName === tag || c.className === tag)) return c;
      const r = c && walk(c); if (r) return r; } return null; }; return walk(this); }
  set innerHTML(v) { this.children = []; }
}
class Option { constructor(text, value) { this.text = text; this.value = value; this.disabled = false; this.title = ""; } }
const byIdEls = {};
const document = { activeElement: null,
  getElementById(id) { return byIdEls[id] = byIdEls[id] || new Node_("div"); },
  createElement(t) { return new Node_(t); } };
const sent = [];
class WebSocket { static OPEN = 1;
  constructor(url) { this.url = url; this.readyState = 1; WebSocket.last = this; }
  send(m) { sent.push(JSON.parse(m)); } close() {} }
const location = { protocol: "http:", host: "x" };
const localStorage = { getItem() { return null; }, setItem() {} };
const navigator = {};
function setTimeout() { return 0; } function clearTimeout() {}
"""


@needs_node
def test_page_flow_with_stub_dom():
    _node(DOM_STUB + _script() + r"""
const ctlWs = WebSocket.last;
const recv = m => ctlWs.onmessage({ data: JSON.stringify(m) });
const sel = fields.asr_backend[0].input;
assert.equal(sel.options.length, STATIC_CATALOG.length + 1);
assert.equal(prov.box.hidden, true, "panel hidden for local");

recv({ type: "settings", ...DEFAULTS, asr_backend: "local" });
recv({ type: "providers", backends: ["local", "soniox", "azure"],
  catalog: STATIC_CATALOG.filter(p => ["soniox", "azure", "deepgram"].includes(p.id)),
  keys: { soniox: { set: true, hint: "••••wxyz", source: "saved", region: null, env: "SONIOX_API_KEY" } } });
assert.equal(sel.options.length, 4);

sel.value = "deepgram"; sel.listeners.change.forEach(f => f());
assert.deepEqual(sent.pop(), { type: "set", settings: { asr_backend: "deepgram" } });
assert.equal(prov.box.hidden, false);
assert.equal(sectionEls.Confidence.hidden, true);
const auto = fields.asr_mode[0].input.options.find(o => o.value === "auto");
assert.equal(auto.disabled, true);
assert.equal(prov.status.textContent, "No key saved");
assert.equal(prov.remove.hidden, true);
assert.equal(prov.regionRow.hidden, true);
assert.equal(prov.key.type, "password");
assert.equal(prov.key.value, "");

// server refuses the switch, operator saves a key -> switch retried
recv({ type: "error", message: "No API key saved for Deepgram" });
assert.equal(errEl.textContent, "No API key saved for Deepgram");
prov.key.value = " dg-secret "; prov.key.listeners.input.forEach(f => f());
assert.equal(prov.save.disabled, false);
prov.key.listeners.keydown.forEach(f => f({ key: "Enter", preventDefault() {} }));
assert.deepEqual(sent.pop(), { type: "set_key", provider: "deepgram", key: "dg-secret" });
recv({ type: "keys", keys: { deepgram: { set: true, hint: "••••cret", source: "saved", region: null, env: "DEEPGRAM_API_KEY" } } });
assert.equal(prov.key.value, "");
assert.equal(prov.save.disabled, true);
assert.equal(prov.status.textContent, "Key: ••••cret (saved)");
assert.equal(prov.remove.hidden, false);
assert.deepEqual(sent.pop(), { type: "set", settings: { asr_backend: "deepgram" } });
prov.remove.onclick();
assert.deepEqual(sent.pop(), { type: "clear_key", provider: "deepgram" });

// Soniox: auto allowed, translate label notes built-in translation
recv({ type: "settings", ...DEFAULTS, asr_backend: "soniox" });
assert.equal(auto.disabled, false);
assert.equal(fields.translate[0].row.querySelector("label").textContent, "Translate to English (built into Soniox)");

// Azure: region field sent with the key
sel.value = "azure"; sel.listeners.change.forEach(f => f()); sent.pop();
assert.equal(prov.regionRow.hidden, false);
prov.key.value = "az"; prov.region.value = "eastus"; prov.save.onclick();
assert.deepEqual(sent.pop(), { type: "set_key", provider: "azure", key: "az", region: "eastus" });

// status strip
lastStatus = { backend: "soniox", cloud: { provider: "soniox", state: "error", error: "401" }, models: "loading",
  asr: "gu", translate: true, lid: false };
renderStatus();
assert.equal(liveEl2.children[0].textContent, "Engine: Soniox · error: 401");
assert.equal(liveEl2.children[0].className, "bad");
assert.match(liveEl2.children[1], /^ · loading local models… · ASR: Gujarati/);

// back to local: confidence visible, panel hidden, auto enabled
recv({ type: "settings", ...DEFAULTS, asr_backend: "local" });
assert.equal(prov.box.hidden, true);
assert.equal(sectionEls.Confidence.hidden, false);
assert.equal(auto.disabled, false);
assert.equal(fields.translate[0].row.querySelector("label").textContent, "Translate to English");
""")
