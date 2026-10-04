import re
import shutil
import subprocess
from pathlib import Path

import pytest

DISPLAY_HTML = Path(__file__).parents[1] / "gujusub" / "static" / "display.html"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required to run display.html JS")
def test_explicit_empty_translation_clears_same_utterance():
    html = DISPLAY_HTML.read_text()
    match = re.search(
        r"  function handleEvent\(msg\) \{.*?\n  \}\n\n"
        r"  /\* ---------------- WebSocket client",
        html,
        re.DOTALL,
    )
    assert match is not None
    handler = match.group(0).split("\n\n  /*", 1)[0]

    script = f"""
const assert = require("node:assert/strict");
let state = {{ uid: null, committed: "", tail: "", translation: "" }};
function scheduleRender() {{}}
function armClearTimer() {{}}
{handler}

handleEvent({{
  type: "partial",
  utterance_id: 7,
  committed: "ગુજરાતી",
  tail: "",
  translation: "English",
}});
handleEvent({{ type: "partial", utterance_id: 7, committed: "ગુજરાતી text", tail: "" }});
assert.equal(state.translation, "English", "an omitted translation should preserve state");

handleEvent({{
  type: "final",
  utterance_id: 7,
  committed: "ગુજરાતી text",
  tail: "",
  translation: "",
}});
assert.equal(state.translation, "", "an explicit empty translation should clear state");
"""
    subprocess.run(["node", "-e", script], check=True)
