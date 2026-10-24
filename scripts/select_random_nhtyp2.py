import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from src.config import Config


XML_PATH = os.path.join(Config.ANNOTATIONS_DIR, "NHTyp2.xml")
IMAGES_DIR = os.path.join(Config.RAW_IMAGES_DIR, "NHTyp2")
OUTPUT_DIR = os.path.join(Config.DATA_DIR, "randomly-image-test", "NHTyp2_selected")

# Selection targets
MAX_PER_CLASS = 2
APPROX_TOTAL = 10


def find_image_scores(xml_path: str) -> List[Tuple[str, int]]:
    image_scores: List[Tuple[str, int]] = []

    # Use iterparse for memory efficiency on large XML
    for event, elem in ET.iterparse(xml_path, events=("start", "end")):
        if event == "start" and elem.tag == "image":
            current_image_name = elem.attrib.get("name", "")
            current_score: int = None  # type: ignore

        if event == "end" and elem.tag == "box":
            label = elem.attrib.get("label")
            if label == "expert_score":
                for child in elem:
                    if child.tag == "attribute" and child.attrib.get("name") == "score":
                        text_value = (child.text or "").strip()
                        try:
                            current_score = int(text_value)
                        except ValueError:
                            current_score = None  # type: ignore
                # We don't break here to allow parsing to continue; score captured if present

        if event == "end" and elem.tag == "image":
            if current_image_name:
                if isinstance(current_score, int):
                    image_scores.append((current_image_name, current_score))
            # Clear element to free memory
            elem.clear()

    return image_scores


def group_by_score(image_scores: List[Tuple[str, int]]) -> Dict[int, List[str]]:
    by_score: Dict[int, List[str]] = defaultdict(list)
    for name, score in image_scores:
        by_score[score].append(name)
    return by_score


def choose_samples(by_score: Dict[int, List[str]]) -> List[Tuple[str, int]]:
    # Shuffle lists for randomness
    scores = sorted(by_score.keys())
    random.shuffle(scores)

    # First, take up to MAX_PER_CLASS from each class
    candidates: List[Tuple[str, int]] = []
    for score in scores:
        names = by_score[score][:]
        random.shuffle(names)
        take = min(MAX_PER_CLASS, len(names))
        for name in names[:take]:
            candidates.append((name, score))

    # Cap to APPROX_TOTAL while keeping at most MAX_PER_CLASS per class
    if len(candidates) <= APPROX_TOTAL:
        return candidates

    # Reduce while preserving at most MAX_PER_CLASS per class
    random.shuffle(candidates)
    kept: List[Tuple[str, int]] = []
    per_class_counts: Dict[int, int] = defaultdict(int)
    for name, score in candidates:
        if len(kept) >= APPROX_TOTAL:
            break
        if per_class_counts[score] < MAX_PER_CLASS:
            kept.append((name, score))
            per_class_counts[score] += 1
    return kept


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def encode_score_suffix(score: int) -> str:
    # Subtle encoding: "-v{score}" suffix before extension
    return f"-v{score}"


def copy_and_rename(samples: List[Tuple[str, int]], images_dir: str, output_dir: str) -> List[str]:
    ensure_output_dir(output_dir)
    written: List[str] = []
    for rel_name, score in samples:
        # Images in XML are relative to a subfolder like "NHTyp2/filename.jpg"; normalize
        base_name = os.path.basename(rel_name)
        src_path = os.path.join(images_dir, base_name)
        if not os.path.isfile(src_path):
            # Skip if source is missing
            continue

        name, ext = os.path.splitext(base_name)
        dst_name = f"{name}{encode_score_suffix(score)}{ext}"
        dst_path = os.path.join(output_dir, dst_name)
        shutil.copy2(src_path, dst_path)
        written.append(dst_path)
    return written


def main() -> None:
    image_scores = find_image_scores(XML_PATH)
    by_score = group_by_score(image_scores)
    samples = choose_samples(by_score)
    written_paths = copy_and_rename(samples, IMAGES_DIR, OUTPUT_DIR)
    print("Selected and copied:")
    for p in written_paths:
        print(p)


if __name__ == "__main__":
    main()


