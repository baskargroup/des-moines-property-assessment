"""
Same P6 prompt TEXT as prompt.py — but the 5 few-shot EXAMPLE IMAGES are
re-anchored to NOGWH's own housing stock (user-selected), with matched-style
factual descriptions. Only the example images+descriptions change; the rubric,
scoring logic, and JSON schema are byte-for-byte identical to prompt.py.

The example descriptions are not distributed with this repository; see
data/prompts/README.md.
"""
import os

# Reuse the identical rubric text from prompt.py
from prompt import PROMPT_TEXT, _read_examples  # same module dir (added to sys.path by runner)

ANCHOR_DIR = r"anchors"

EXAMPLES = _read_examples("p6_nogwh_examples.json", ANCHOR_DIR)


def get_prompt_and_examples():
    return PROMPT_TEXT, EXAMPLES


if __name__ == "__main__":
    p, ex = get_prompt_and_examples()
    print(f"Prompt length: {len(p)} chars (identical to original)")
    for path, score, _ in ex:
        ok = "OK" if os.path.exists(path) else "MISSING"
        print(f"  Score {score}: {os.path.basename(path)}  {ok}")
