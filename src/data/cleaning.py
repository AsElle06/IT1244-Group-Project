"""Cleaning the report texts and labels before they go into the models.

Steps decided by the dataset analysis (main.ipynb section 2, summary in 2.10):
- Drop reports labelled NONE: they have no IMDRF code and no problem term, so the label is
  missing. Drop reports whose description is empty (also after cleaning).
- Merge A24-A27 into one class, as the course README's label table does
  ("Unknown / No device problem").
- Merge classes with fewer than config.MIN_CLASS_COUNT training reports into "OTHER",
  decided once on the whole dataset.
- Cut every cleaned text that appears more than config.MAX_COPIES_PER_TEXT times to that many
  copies (cap_copies), so no template makes one fold much heavier in its class.
- Clean the text (clean_text): keep one copy of a description that repeats itself in full
  (45% of Abbott reports); lowercase; remove redaction markers such as "(B)(6)", report and
  field-action IDs, dates, times and years; replace other numbers by "num"; expand
  abbreviations; keep negations and punctuation; no stemming.

All steps are fixed rules applied to the whole dataset before the folds are made (section 3);
nothing is learned from the data here.

Owner: Suling. Status: done.
"""
import re

import pandas as pd

from src import config

MISSING_LABEL = "NONE"
MERGED_CLASSES = {"A24": "A24-A27", "A25": "A24-A27", "A26": "A24-A27", "A27": "A24-A27"}
OTHER_LABEL = "OTHER"

REDACTION_MARKER = r"\(\s*[Bb]\s*\)\s*\(\s*\d\s*\)"  # e.g. "(B)(6)": text removed by the FDA

# Abbreviations found in the descriptions (section 2), expanded so every model sees the same words
ABBREVIATIONS = {
    "cgm": "continuous glucose monitor",
    "cgms": "continuous glucose monitors",
    "bg": "blood glucose",
    "hcp": "healthcare professional",
    "hcps": "healthcare professionals",
    "adc": "abbott diabetes care",
    "dka": "diabetic ketoacidosis",
    "mg/dl": "milligrams per decilitre",
    "mmol/l": "millimoles per litre",
    "egv": "estimated glucose value",
    "egvs": "estimated glucose values",
    "er": "emergency room",
    "icu": "intensive care unit",
    "ems": "emergency medical services",
    "iv": "intravenous",
    "pt": "patient",
    "pts": "patients",
    "os": "operating system",
}

# A mis-encoded apostrophe ("PATIENT�S", "WON�T", "I�M"); elsewhere the character was a space
_BROKEN_APOSTROPHE = re.compile(r"(?<=[A-Za-z])�(?=(?:S|T|M|LL|RE|VE)\b)", re.IGNORECASE)
# Also malformed markers in lowercased text: "(b)((6))", "(b)(6", "(b) 6)"
_REDACTION = re.compile(r"\(+\s*b\s*\)+\s*\(*\s*\d\s*\)*")
# "(CGM)" right after "CONTINUOUS GLUCOSE MONITOR": a definition, dropped so expanding it doesn't repeat the words
_ABBREVIATION_IN_BRACKETS = re.compile(r"\(\s*(?:" + "|".join(map(re.escape, ABBREVIATIONS)) + r")\s*\)")
_DATE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
_TIME = re.compile(r"\b\d{1,2}(?::\d{2}){1,2}\s*(?:am|pm)?\b|\b\d{1,2}\s*(?:am|pm)\b")
# A token with letters and at least 4 digits, e.g. "mw5177687", "corpft-140202", "fa1002-2025"
_ID = re.compile(r"#?\b(?=[a-z0-9-]*[a-z])(?=(?:[a-z-]*\d){4})[a-z0-9]+(?:-[a-z0-9]+)*\b")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_GLUED_UNIT = re.compile(r"(\d)(mg/dl|mmol/l|mg|mmol)\b")  # "50mg/dl" -> "50 mg/dl"
_ABBREVIATION = re.compile(r"(?<![\w/])(" + "|".join(map(re.escape, ABBREVIATIONS)) + r")(?![\w/])")
_NUMBER = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)*(?!\w)")  # not inside words such as "g7"
_SPACES = re.compile(r"\s+")
_SENTENCE_END = re.compile(r"(?<=\.)\s+")


def remove_repeated_text(text: str) -> str:
    """Keep one copy of a text that is the same passage repeated: "A. B. A. B." -> "A. B.".

    45% of Abbott reports contain their whole description two or more times, which doubles
    their length. Partial repeats (e.g. a follow-up appended to the initial report) are kept,
    because the added sentences carry new information.
    """
    sentences = _SENTENCE_END.split(text.strip())
    n = len(sentences)
    for period in range(1, n // 2 + 1):
        if n % period == 0 and sentences == sentences[:period] * (n // period):
            return " ".join(sentences[:period])
    return text


def template_key(texts: pd.Series) -> pd.Series:
    """Each report text reduced to its template, for finding near-identical reports.

    Keeps one copy of a description that repeats itself (remove_repeated_text), upper-cases it
    and removes redaction markers, every token containing a digit (dates, years, IDs, glucose
    values) and punctuation. Reports with the same key differ only in those details, so the
    dataset analysis counts them as one description, and eda.text_frequency uses the key to
    tell templates from one-off text. Same index as texts; missing values give "".
    """
    return (
        texts.fillna("").astype(str).map(remove_repeated_text).str.upper()
        .str.replace(REDACTION_MARKER, " ", regex=True)
        .str.replace(r"\S*\d\S*", " ", regex=True)
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.split().str.join(" ")
    )


def clean_text(text: str) -> str:
    """Normalise one report text. Returns "" for missing values (NaN / None).

    "IT WAS REPORTED THAT AN INACCURACY BETWEEN THE CONTINUOUS GLUCOSE MONITOR (CGM) AND THE
    BLOOD GLUCOSE (BG) METER OCCURRED. THE SENSOR WAS INSERTED INTO THE ARM ON (B)(6) 2025."
    -> "it was reported that an inaccuracy between the continuous glucose monitor and the
    blood glucose meter occurred. the sensor was inserted into the arm on ."
    """
    if text is None or pd.isna(text):
        return ""
    text = remove_repeated_text(str(text))
    text = _BROKEN_APOSTROPHE.sub("'", text).replace("�", " ").lower()
    text = _REDACTION.sub(" ", text)
    text = _ABBREVIATION_IN_BRACKETS.sub(" ", text)
    text = _DATE.sub(" ", text)
    text = _TIME.sub(" ", text)
    text = _ID.sub(" ", text)
    text = _YEAR.sub(" ", text)
    text = _GLUED_UNIT.sub(r"\1 \2", text)
    text = _ABBREVIATION.sub(lambda m: ABBREVIATIONS[m.group(1)], text)
    text = _NUMBER.sub(" num ", text)
    return _SPACES.sub(" ", text).strip()


def clean_labels(labels: pd.Series) -> pd.Series:
    """Merge A24-A27 into one class "A24-A27". Same index and length as labels."""
    return labels.replace(MERGED_CLASSES)


def clean_reports(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all fixed cleaning rules to the whole dataset (before the folds are made).

    Drops reports labelled NONE and reports with no text left after cleaning, cleans
    config.TEXT_COL (clean_text) and merges A24-A27 (clean_labels). Returns the same columns as
    df, fewer rows, index reset to 0..n-1. Rare classes are merged separately (merge_rare_classes).
    """
    df = df[df[config.LABEL_COL] != MISSING_LABEL].copy()
    df[config.TEXT_COL] = df[config.TEXT_COL].map(clean_text)
    df = df[df[config.TEXT_COL] != ""]
    df[config.LABEL_COL] = clean_labels(df[config.LABEL_COL])
    return df.reset_index(drop=True)


def frequent_classes(labels: pd.Series, min_count: float = config.MIN_CLASS_COUNT) -> list[str]:
    """Classes with at least min_count reports: the ones kept as they are."""
    counts = labels.value_counts()
    return sorted(counts[counts >= min_count].index)


def merge_rare_classes(labels: pd.Series, keep: list[str], new_label: str = OTHER_LABEL) -> pd.Series:
    """Replace every label not in keep by new_label. Same index and length as labels."""
    return labels.where(labels.isin(keep), new_label)


def cap_copies(df: pd.DataFrame, max_copies: int = config.MAX_COPIES_PER_TEXT,
               seed: int = config.SEED) -> pd.DataFrame:
    """Randomly cut every cleaned text that appears more than max_copies times to max_copies copies.

    Texts that appear max_copies times or fewer are kept in full. The copies kept are a random
    sample of the text's reports, so its mix of labels stays about the same. Returns the kept
    rows in their original order, index reset to 0..n-1.
    """
    shuffled = df.sample(frac=1, random_state=seed)
    kept = shuffled.groupby(config.TEXT_COL, sort=False).head(max_copies)
    return kept.sort_index().reset_index(drop=True)
