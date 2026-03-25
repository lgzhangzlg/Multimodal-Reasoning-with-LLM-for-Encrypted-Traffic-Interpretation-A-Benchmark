# tinyllava/data/prompt_utils.py
# -*- coding: utf-8 -*-

import re
from typing import Dict, List

SUPER_CLASSES = {"malware", "benign"}


def strip_path_leak(text: str) -> str:
    """Remove 'Path: xxx.pcap.' leakage to avoid cheating."""
    if not text:
        return ""
    return re.sub(r"Path:\s*.*?\.pcap\.\s*", "", str(text), flags=re.IGNORECASE)


def extract_class_from_label_sentence(label_sentence: str) -> str:
    """
    Parse class name from label_sentence.
    Examples:
      "This is BitTorrent traffic." -> "BitTorrent"
      "This is EVSE-A-Charging-Benign traffic." -> "EVSE-A-Charging-Benign"
    """
    s = str(label_sentence).strip()

    m = re.search(r"(?i)\bthis\s+is\s+(.+?)\s+traffic\b", s)
    if m:
        return m.group(1).strip()

    m = re.search(r"(?i)\bthis\s+is\s+(.+?)\s*\.?\s*$", s)
    if m:
        return m.group(1).strip()

    return s.strip()


def remove_leading_class_sentence(nl: str) -> str:
    """
    Remove leading sentence like 'This is XXX traffic.' from nl_description to avoid duplication.
    (We remove the first such sentence regardless of XXX.)
    """
    if not nl:
        return ""
    nl = str(nl).strip()
    m = re.match(r"(?is)^\s*this\s+is\s+.+?\s+traffic(?:\s+sample)?\s*\.?\s*", nl)
    if m:
        nl = nl[m.end():].lstrip()
    return nl


# def build_user_text(sample: Dict, use_hint: bool) -> str:
#     """
#     Unified prompt for BOTH training and evaluation.
#     - use_hint=True only for training with probability (to reduce over-reliance).
#     - evaluation should set use_hint=False.
#     """
#     parts = [
#         "Below is a network traffic sample.",
#         "",
#         "Your output must follow this pattern:",
#         "First line example: This is benign charging traffic.",
#         "Then continue with several sentences describing the traffic in natural language.",
#         "",
#         "Important:",
#         "- The first line must be a normal sentence, not a template.",
#         "- Do not include any format instructions in the output.",
#     ]
#
#     # if use_hint:
#     #     parts += [f"Category hint: {sample.get('label_sentence', '')}", ""]
#
#     return "\n".join(parts)

def build_user_text(sample: Dict, use_hint: bool) -> str:
    return "\n".join([
        "Below is a network traffic sample.",
        "",
        "Write the answer in two parts:",
        "1) One short first line that names the traffic class.",
        "2) A brief description of the traffic behavior and statistics.",
    ])




def reject_superclass(label: str) -> bool:
    """Return True if label is empty or a super-class like Malware/Benign."""
    if not label:
        return True
    return str(label).strip().lower() in SUPER_CLASSES


# -------- parsing model output (first line) --------
def clean_keep_newlines(t: str) -> str:
    if t is None:
        return ""
    s = str(t)

    # remove leading junk like ".png", "> ", code fences
    s = re.sub(r"^\s*(?:[>\-]+\s*)+", "", s)
    s = re.sub(r"^\s*(?:\.(?:png|jpg|jpeg)\b)\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^\s*```.*?\n", "", s)
    s = re.sub(r"\n```$", "", s)

    # keep newlines but normalize spaces per line
    s = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in s.splitlines())
    return s.strip()


def parse_firstline_candidate(gen_text: str) -> str:
    """
    Allowed:
      This is <CLASS> traffic sample.
      This is <CLASS> traffic.
      This is <CLASS>.
      It is <CLASS> traffic.
      Category: <CLASS>
    Return raw candidate string (not validated against label_set here).
    """
    t = clean_keep_newlines(gen_text)
    if not t:
        return ""
    first = t.splitlines()[0].strip()
    first = re.sub(r"^\s*[>\-\*]+\s*", "", first).strip()

    m = re.search(r"(?i)^\s*this\s+is\s+(.+?)\s+traffic(?:\s+sample)?\s*\.?\s*$", first)
    if m:
        return m.group(1).strip()

    m = re.search(r"(?i)^\s*it\s+is\s+(.+?)\s+traffic\s*\.?\s*$", first)
    if m:
        return m.group(1).strip()

    m = re.search(r"(?i)^\s*this\s+is\s+(.+?)\s*\.?\s*$", first)
    if m:
        return m.group(1).strip()

    m = re.search(r"(?i)^\s*(?:category|类别)\s*[:：]\s*(.+?)\s*$", first)
    if m:
        return m.group(1).strip()

    return ""


def normalize_label_candidate(cand: str) -> str:
    if not cand:
        return ""
    x = str(cand).strip().strip(" .;，,。:：")
    x = re.sub(r"\s+", " ", x)

    x = re.sub(r"\btraffic\s+sample\b$", "", x, flags=re.IGNORECASE).strip()
    x = re.sub(r"\btraffic\b$", "", x, flags=re.IGNORECASE).strip()
    return x.strip(" .;，,。:：")
