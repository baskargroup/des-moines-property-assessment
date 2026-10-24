"""
Load the P6 (5-image few-shot) prompt and example image pairs. Returns:
  - prompt_text (str)
  - examples: list[(image_path, score, description)]

The prompt text and example descriptions are not distributed with this repository;
see data/prompts/README.md for the expected files.
"""
import json
import os

EX_DIR = r"../data/download/example_images"
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "prompts")


def _read_prompt():
    with open(os.path.join(PROMPTS_DIR, "p6_prompt.txt"), encoding="utf-8") as f:
        return f.read()


def _read_examples(name, image_dir):
    with open(os.path.join(PROMPTS_DIR, name), encoding="utf-8") as f:
        return [(os.path.join(image_dir, e["file"]), e["score"], e["description"]) for e in json.load(f)]


PROMPT_TEXT = _read_prompt()
EXAMPLES = _read_examples("p6_examples.json", EX_DIR)


def get_prompt_and_examples():
    """Return (prompt_text, examples) for P6."""
    return PROMPT_TEXT, EXAMPLES


if __name__ == "__main__":
    prompt, examples = get_prompt_and_examples()
    print(f"Prompt length: {len(prompt)} chars")
    print(f"Examples: {len(examples)}")
    for path, score, _ in examples:
        ok = "OK" if os.path.exists(path) else "MISSING"
        size = os.path.getsize(path) if os.path.exists(path) else 0
        print(f"  Score {score}: {os.path.basename(path)}  [{size/1024:.0f} KB]  {ok}")
