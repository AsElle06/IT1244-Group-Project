"""Exploratory analysis of the whole dataset: distributions and associations.

Runs on the whole dataset, before the folds are made: the folds themselves depend on what this
finds (which labels to drop or merge, how to group near-identical reports). Nothing here is
fitted: everything the models learn (vocabulary, class weights, network weights) comes from
each turn's training folds only.

One function per planned analysis in the proposal; each comment says which decision it informs.
Plot functions show the figure and, if save_as is given, also save it to config.FIGURES_DIR
for the report and slides.

Owner: Leanne. Status: done.
"""
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.feature_selection import chi2

from src import config
from src.data.cleaning import REDACTION_MARKER, template_key

TEXT_COL = config.TEXT_COL
LABEL_COL = config.LABEL_COL
BRAND_COL = config.BRAND_COL
NARRATIVE_COL = "manufacturer_narrative"


def _save_plot(save_as):
    if save_as is not None:
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            config.FIGURES_DIR / save_as,
            dpi=200,
            bbox_inches="tight"
        )


def _word_counts(texts):
    return texts.fillna("").astype(str).str.split().str.len()


def brand_family(brands: pd.Series) -> pd.Series:
    """Manufacturer of each brand_name: "Dexcom", "Abbott" (FreeStyle Libre) or "Other"."""
    names = brands.fillna("").astype(str).str.upper()
    family = np.select(
        [names.str.contains("DEXCOM|G6|G7"), names.str.contains("LIBRE|FREESTYLE")],
        ["Dexcom", "Abbott"],
        "Other"
    )
    return pd.Series(family, index=brands.index)


def _boxplot_by_group(words: pd.Series, groups: pd.Series, order, ax, vert=True):
    """One box per group in the given order, labelled with its number of reports."""
    data = [words[groups == g] for g in order]
    labels = [f"{g}\nn={len(d):,}" if vert else f"{g} (n={len(d):,})" for g, d in zip(order, data)]
    ax.boxplot(data, tick_labels=labels, showfliers=False,
               orientation="vertical" if vert else "horizontal")


# 1. Class counts -> class weights, merging rare classes, why macro-F1

def class_counts(labels: pd.Series, texts: pd.Series | None = None) -> pd.DataFrame:
    """Index: class label, most frequent first. Columns: "count" (int), "share_%" (float).

    If texts is given, also "distinct_texts": the number of different descriptions in the class
    after removing dates, IDs and redaction markers (cleaning.template_key).
    """
    result = pd.DataFrame({
        "count": labels.value_counts(),
        "share_%": labels.value_counts(normalize=True) * 100
    })

    if texts is not None:
        keys = template_key(texts)
        result["distinct_texts"] = keys[keys != ""].groupby(labels).nunique()
        result["distinct_texts"] = result["distinct_texts"].fillna(0).astype(int)

    return result


def plot_class_counts(labels: pd.Series, save_as: str | None = None) -> None:
    """Bar chart of reports per class."""
    counts = class_counts(labels)

    plt.figure(figsize=(12, 6))
    plt.bar(counts.index, counts["count"])

    plt.xlabel("IMDRF Problem Category")
    plt.ylabel("Number of Reports")
    plt.title("Distribution of CGM Reports by Problem Category")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    _save_plot(save_as)
    plt.show()


# 2. Text length -> max input length for the BiLSTM and transformer

def length_percentiles(texts: pd.Series, percentiles: tuple[int, ...] = (50, 90, 95, 99)) -> pd.Series:
    """Words per description at each percentile. Index: "p50", "p90", ... Values: float."""
    results = _word_counts(texts).quantile([p / 100 for p in percentiles])
    results.index = [f"p{p}" for p in percentiles]
    return results


def plot_length_hist(texts: pd.Series, save_as: str | None = None) -> None:
    """Histogram of words per description."""
    lengths = _word_counts(texts)

    plt.figure(figsize=(10, 5))
    plt.hist(lengths, bins=50, edgecolor="white")

    plt.xlabel("Words per Description")
    plt.ylabel("Number of Reports")
    plt.title("Distribution of CGM Report Lengths")

    plt.tight_layout()
    _save_plot(save_as)
    plt.show()


def plot_length_by_class(df: pd.DataFrame, save_as: str | None = None) -> None:
    """Description length per class: one box per class, largest class first, with its size."""
    labels = df[LABEL_COL]
    words = _word_counts(df[TEXT_COL])
    order = labels.value_counts().index

    fig, ax = plt.subplots(figsize=(18, 6))
    _boxplot_by_group(words, labels, order, ax)

    ax.set_title("Description Length by IMDRF Category (largest category first, outliers hidden)")
    ax.set_xlabel("IMDRF Category (n = number of reports)")
    ax.set_ylabel("Number of Words")
    plt.setp(ax.get_xticklabels(), fontsize=7)

    plt.tight_layout()
    _save_plot(save_as)
    plt.show()


def plot_length_by_brand(df: pd.DataFrame, min_reports: int = 1000, save_as: str | None = None) -> None:
    """Description length per brand_name with at least min_reports reports, largest brand first."""
    brands = df[BRAND_COL].fillna("Unknown")
    words = _word_counts(df[TEXT_COL])
    counts = brands.value_counts()
    order = counts[counts >= min_reports].index[::-1]  # largest at the top

    fig, ax = plt.subplots(figsize=(14, 0.6 * len(order) + 1.5))
    _boxplot_by_group(words, brands, order, ax, vert=False)

    ax.set_title(f"Description Length by Brand (brands with at least {min_reports:,} reports, outliers hidden)")
    ax.set_xlabel("Number of Words")

    plt.tight_layout()
    _save_plot(save_as)
    plt.show()


# 3. Empty / very short descriptions -> cleaning: drop them or keep them

def short_description_summary(texts: pd.Series) -> pd.DataFrame:
    """Index: thresholds such as "<= 0 words", "<= 3 words". Columns: "reports" (int), "share_%" (float)."""
    lengths = _word_counts(texts)

    results = []
    for t in [0, 3, 5, 10]:
        count = (lengths <= t).sum()
        results.append({
            "threshold": f"<= {t} words",
            "reports": int(count),
            "share_%": count / len(lengths) * 100 if len(lengths) else 0
        })

    return pd.DataFrame(results).set_index("threshold")


# 4. Duplicate (template) descriptions -> how the folds keep them together

def duplicate_summary(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Returns (stats, top_templates).

    Duplicates are counted twice: on the exact text, and on the text with dates, IDs, numbers
    and redaction markers removed (cleaning.template_key), which also catches near-copies.

    stats: named counts, e.g. "reports", "unique descriptions (exact)",
    "unique descriptions (normalised)", "normalised texts with >1 label".
    top_templates: the most repeated normalised descriptions. Index: normalised text.
    Columns: "count" (int), "labels" (dict of label -> count).
    """
    exact = df[TEXT_COL].fillna("").astype(str).str.strip()
    valid = exact.ne("")

    temp = pd.DataFrame({
        "exact": exact[valid],
        "key": template_key(df.loc[valid, TEXT_COL]),
        "label": df.loc[valid, LABEL_COL].fillna("(missing)").astype(str)
    })

    exact_counts = temp["exact"].value_counts()
    key_counts = temp.loc[temp["key"] != "", "key"].value_counts()

    stats = pd.Series({
        "reports": len(df),
        "reports with non-empty descriptions": len(temp),
        "unique descriptions (exact)": len(exact_counts),
        "reports sharing their exact text": int(exact_counts[exact_counts > 1].sum()),
        "exact texts with >1 label": int((temp.groupby("exact")["label"].nunique() > 1).sum()),
        "unique descriptions (normalised)": len(key_counts),
        "reports sharing their normalised text": int(key_counts[key_counts > 1].sum()),
        "normalised texts with >1 label": int((temp.groupby("key")["label"].nunique() > 1).sum()),
    })

    top_counts = key_counts.head(10)
    top_labels = temp[temp["key"].isin(top_counts.index)].groupby("key")["label"].value_counts()
    top_templates = pd.DataFrame({
        "count": top_counts,
        "labels": [top_labels[text].to_dict() for text in top_counts.index]
    }, index=top_counts.index)
    top_templates.index.name = "text"

    return stats, top_templates


# 5. Template vs free text -> what "doing well" means; evaluate by text frequency and manufacturer

FREQUENCY_GROUPS = ["10+ copies", "2-9 copies", "once"]

# Dexcom's fill-in-the-blank opening sentence, e.g. "IT WAS REPORTED THAT A SIGNAL LOSS OCCURRED."
# The complaint phrase is the blank; a trailing "ISSUE" is dropped ("TRANSMITTER BATTERY ISSUE").
COMPLAINT_PHRASE = r"^\s*IT WAS REPORTED THAT (?:AN? )?([^.]+?)(?: ISSUE)? OCCURRED"


def text_frequency(texts: pd.Series) -> pd.Series:
    """How often each report's normalised text (cleaning.template_key) occurs among texts:
    "10+ copies" (a template), "2-9 copies" or "once" (one-off text). NaN for empty texts.

    Pass the whole dataset: to label a subset such as the test set, compute this on the whole
    dataset and select its rows, so a template counts as a template even if the subset holds one copy.
    """
    keys = template_key(texts)
    size = keys.map(keys.value_counts())
    groups = pd.cut(size, [0, 1, 9, np.inf], labels=["once", "2-9 copies", "10+ copies"])
    return groups.astype(object).where(keys != "")


def text_frequency_by_brand(df: pd.DataFrame) -> pd.DataFrame:
    """Share (%) of each manufacturer's reports in each text-frequency group.

    Index: "Abbott", "Dexcom", "Other", "All". Columns: FREQUENCY_GROUPS, then "reports".
    """
    frequency = text_frequency(df[TEXT_COL])
    family = brand_family(df[BRAND_COL])
    valid = frequency.notna()

    shares = pd.crosstab(family[valid], frequency[valid], normalize="index",
                         margins=True, margins_name="All") * 100
    shares = shares.reindex(columns=FREQUENCY_GROUPS, fill_value=0).rename_axis(index="manufacturer", columns=None)
    shares["reports"] = pd.concat([family[valid].value_counts(), pd.Series({"All": valid.sum()})])
    return shares


def text_frequency_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Index: FREQUENCY_GROUPS. Columns: "reports", "share_%", "median_words",
    "top_label" (most common class) and "top_label_%"."""
    frequency = text_frequency(df[TEXT_COL])
    temp = pd.DataFrame({
        "frequency": frequency,
        "words": _word_counts(df[TEXT_COL]),
        "label": df[LABEL_COL]
    }).dropna(subset=["frequency"])

    grouped = temp.groupby("frequency")
    result = pd.DataFrame({
        "reports": grouped.size(),
        "share_%": grouped.size() / len(temp) * 100,
        "median_words": grouped["words"].median(),
        "top_label": grouped["label"].agg(lambda s: s.value_counts().index[0]),
        "top_label_%": grouped["label"].agg(lambda s: s.value_counts(normalize=True).iloc[0] * 100)
    })
    return result.reindex(FREQUENCY_GROUPS)


def plot_text_frequency_by_brand(df: pd.DataFrame, save_as: str | None = None) -> None:
    """Stacked bar per manufacturer: share of reports whose text appears 10+ times, 2-9 times or once."""
    shares = text_frequency_by_brand(df).loc[["All", "Abbott", "Dexcom"]]
    colors = ["#104281", "#2a78d6", "#86b6ef"]  # one blue, template (dark) -> one-off text (light)

    fig, ax = plt.subplots(figsize=(10, 3.2))
    left = np.zeros(len(shares))
    for group, color in zip(FREQUENCY_GROUPS, colors):
        values = shares[group].to_numpy()
        ax.barh(shares.index, values, left=left, color=color, edgecolor="white", linewidth=2,
                height=0.6, label=group)
        for y, (x, v) in enumerate(zip(left, values)):
            if v >= 4:
                ax.text(x + v / 2, y, f"{v:.1f}%", ha="center", va="center",
                        color="white" if color != colors[-1] else "black", fontsize=9)
        left += values

    ax.set_yticks(range(len(shares)))
    ax.set_yticklabels([f"{name} (n={n:,})" for name, n in zip(shares.index, shares["reports"])])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of reports (%)")
    ax.set_title("How often each report's text appears in the dataset (after removing dates, IDs and numbers)")
    ax.legend(title="Text appears", loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3, frameon=False)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

    plt.tight_layout()
    _save_plot(save_as)
    plt.show()


def complaint_phrase(texts: pd.Series) -> pd.Series:
    """The complaint phrase of each Dexcom template report ("SIGNAL LOSS", "DEXCOM APP CRASH"),
    NaN if the report does not open with "IT WAS REPORTED THAT ... OCCURRED". Same index as texts."""
    return texts.fillna("").astype(str).str.upper().str.extract(COMPLAINT_PHRASE, expand=False)


def _label_split(labels: pd.Series, min_share: float = 5) -> str:
    shares = labels.value_counts(normalize=True) * 100
    return " / ".join(f"{share:.0f}% {label}" for label, share in shares[shares >= min_share].items())


def phrase_summary(df: pd.DataFrame, min_reports: int = 500) -> tuple[pd.Series, pd.DataFrame]:
    """Returns (stats, phrases) for the complaint phrases (complaint_phrase).

    stats: "reports with a phrase_%", "Dexcom reports with a phrase_%", "distinct phrases",
    "phrase predicts label_%" (share of reports with a phrase whose label is that phrase's most
    common label: the accuracy of a phrase -> label lookup table on these reports).
    phrases: phrases with at least min_reports reports, most reports first. Index: phrase.
    Columns: "reports", "top_label", "top_label_%", "label_split" (e.g. "52% A11 / 47% A09").
    """
    phrase = complaint_phrase(df[TEXT_COL])
    family = brand_family(df[BRAND_COL])
    temp = pd.DataFrame({"phrase": phrase, "label": df[LABEL_COL]}).dropna(subset=["phrase"])

    top_count = temp.groupby("phrase")["label"].agg(lambda s: s.value_counts().iloc[0])
    stats = pd.Series({
        "reports with a phrase_%": phrase.notna().mean() * 100,
        "Dexcom reports with a phrase_%": phrase[family == "Dexcom"].notna().mean() * 100,
        "distinct phrases": temp["phrase"].nunique(),
        "phrase predicts label_%": top_count.sum() / len(temp) * 100
    })

    grouped = temp.groupby("phrase")["label"]
    phrases = pd.DataFrame({
        "reports": grouped.size(),
        "top_label": grouped.agg(lambda s: s.value_counts().index[0]),
        "top_label_%": top_count / grouped.size() * 100,
        "label_split": grouped.agg(_label_split)
    })
    phrases = phrases[phrases["reports"] >= min_reports].sort_values("reports", ascending=False)
    return stats, phrases


def phrase_coverage_by_class(df: pd.DataFrame) -> pd.DataFrame:
    """How much a phrase -> label lookup can do for each class.

    Index: class, largest first. Columns: "reports", "with_phrase_%" (share of the class's
    reports that have a complaint phrase) and "lookup_correct_%" (share of the class's reports
    whose phrase's most common label is this class, i.e. that the lookup would get right).
    """
    phrase = complaint_phrase(df[TEXT_COL])
    labels = df[LABEL_COL]
    temp = pd.DataFrame({"phrase": phrase, "label": labels}).dropna(subset=["phrase"])
    phrase_label = temp.groupby("phrase")["label"].agg(lambda s: s.value_counts().index[0])
    correct = phrase.map(phrase_label).eq(labels)

    return pd.DataFrame({
        "reports": labels.value_counts(),
        "with_phrase_%": phrase.notna().groupby(labels).mean() * 100,
        "lookup_correct_%": correct.groupby(labels).mean() * 100
    }).sort_values("reports", ascending=False)


# 6. Reports by brand -> per-brand results, templated vs free text

def brand_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Index: brand. Columns: "reports" (int), "duplicate_%" (float), "median_words" (float).

    duplicate_%: share of the brand's reports whose normalised text (cleaning.template_key)
    is shared with another report of the same brand.
    """
    summary = pd.DataFrame({
        "brand": df[BRAND_COL].fillna("Unknown").astype(str).str.strip().replace("", "Unknown"),
        "key": template_key(df[TEXT_COL]),
        "words": _word_counts(df[TEXT_COL])
    })

    summary["is_duplicate"] = (
        summary["key"].ne("")
        & summary.duplicated(subset=["brand", "key"], keep=False)
    )

    result = summary.groupby("brand").agg(
        reports=("brand", "size"),
        duplicate_ratio=("is_duplicate", "mean"),
        median_words=("words", "median")
    )
    result["duplicate_%"] = result["duplicate_ratio"] * 100

    return result[["reports", "duplicate_%", "median_words"]].sort_values("reports", ascending=False)


def plot_brand_by_class(df: pd.DataFrame, save_as: str | None = None) -> None:
    """Heatmap of each brand's share of reports in each class."""
    brands = df[BRAND_COL].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")

    counts = pd.crosstab(brands, df[LABEL_COL])
    if counts.empty:
        raise ValueError("No brand/category data available.")
    counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index]

    # Convert to within-brand percentages
    shares = counts.div(counts.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(
        figsize=(max(10, 0.8 * len(shares.columns)), max(5, 0.5 * len(shares)))
    )
    heatmap = ax.imshow(shares.to_numpy(), cmap="Blues", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(shares.columns)))
    ax.set_xticklabels(shares.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(shares.index)))
    ax.set_yticklabels(shares.index)

    ax.set_xlabel("IMDRF Category")
    ax.set_ylabel("CGM Brand")
    ax.set_title("IMDRF Category Distribution by Brand")
    fig.colorbar(heatmap, ax=ax, label="Share of Brand Reports (%)")

    plt.tight_layout()
    _save_plot(save_as)
    plt.show()


# 7. Manufacturer narrative -> whether it could add signal (README question); never a model input

def narrative_summary(df: pd.DataFrame, top: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (stats, top_narratives) for manufacturer_narrative, per brand family.

    stats: index brand family plus "All". Columns: "reports", "empty_%", "redaction_only_%"
    (nothing but markers such as "(B)(4)"), "top5_%" (share of the family's reports whose
    narrative is one of its 5 most common narratives), "median_words" (non-empty narratives).
    top_narratives: the `top` most common narratives per family. Columns: "family",
    "narrative" (first 150 characters), "reports", "share_%" (of the family's reports).
    """
    text = df[NARRATIVE_COL].fillna("").astype(str).str.strip()
    stripped = text.str.replace(REDACTION_MARKER, "", regex=True).str.replace(r"[^\w]", "", regex=True)
    temp = pd.DataFrame({
        "family": brand_family(df[BRAND_COL]),
        "text": text,
        "empty": text.eq(""),
        "redaction_only": text.ne("") & stripped.eq(""),
        "words": _word_counts(text)
    })

    rows, tops = {}, []
    for name, group in [*temp.groupby("family"), ("All", temp)]:
        counts = group["text"].value_counts()
        rows[name] = {
            "reports": len(group),
            "empty_%": group["empty"].mean() * 100,
            "redaction_only_%": group["redaction_only"].mean() * 100,
            "top5_%": counts.head(5).sum() / len(group) * 100,
            "median_words": group.loc[~group["empty"], "words"].median()
        }
        if name != "All":
            tops.append(pd.DataFrame({
                "family": name,
                "narrative": counts.index[:top].str[:150],
                "reports": counts.values[:top],
                "share_%": counts.values[:top] / len(group) * 100
            }))

    stats = pd.DataFrame(rows).T.astype({"reports": int})
    return stats, pd.concat(tops, ignore_index=True)


# 8. Reports with more than one IMDRF category -> evaluate single vs multi separately

def count_imdrf_codes(imdrf_categories: pd.Series) -> pd.Series:
    """Number of distinct IMDRF codes listed in each report. Same index as the input, int values."""
    pattern = re.compile(r"\b[A-Z]\d{2,6}\b")

    counts = (
        imdrf_categories.fillna("").astype(str).str.upper()
        .map(lambda text: len(set(pattern.findall(text))))
    )
    return counts.astype(int)


def plot_imdrf_code_counts(imdrf_categories: pd.Series, save_as: str | None = None) -> None:
    """Bar chart: how many reports list 1, 2, 3, ... IMDRF categories."""
    counts = count_imdrf_codes(imdrf_categories).value_counts().sort_index()

    plt.figure(figsize=(9, 5))
    plt.bar(counts.index.astype(str), counts.values)

    plt.xlabel("Number of Distinct IMDRF Codes")
    plt.ylabel("Number of Reports")
    plt.title("Number of IMDRF Codes per CGM Report")

    plt.tight_layout()
    _save_plot(save_as)
    plt.show()


# 9. Distinctive words / phrases per class -> cleaning steps, TF-IDF n-gram range

def distinctive_terms_per_class(texts: pd.Series, labels: pd.Series, k: int = 15,
                                min_texts: int = 20) -> pd.DataFrame:
    """One column per class with at least min_texts distinct texts, k rows: the words/phrases
    most specific to that class (str), most distinctive first.

    Uses one copy of each distinct (normalised text, label) pair, so a template repeated
    thousands of times counts once. Numbers, dates, years and redaction markers are removed
    (cleaning.template_key). Terms are ranked by a one-vs-rest chi-square test on whether a text
    contains the term, keeping only terms more common in the class than in the rest; boilerplate
    that every class shares ("reported", "no injury") therefore ranks low.
    """
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have equal lengths")

    temp = pd.DataFrame({
        "text": template_key(texts).to_numpy(),
        "label": labels.to_numpy()
    }).dropna(subset=["label"])
    temp = temp[temp["text"] != ""].drop_duplicates()

    # Keep important negation words
    stop_words = list(ENGLISH_STOP_WORDS - {"no", "not", "nor"})
    vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words=stop_words, min_df=3, binary=True)
    matrix = vectorizer.fit_transform(temp["text"])
    terms = vectorizer.get_feature_names_out()
    overall_rate = np.asarray(matrix.mean(axis=0)).ravel()

    results = {}
    class_sizes = temp["label"].value_counts()
    for label in class_sizes[class_sizes >= min_texts].index:
        in_class = (temp["label"] == label).to_numpy()
        scores, _ = chi2(matrix, in_class)
        class_rate = np.asarray(matrix[in_class].mean(axis=0)).ravel()
        scores = np.where(class_rate > overall_rate, np.nan_to_num(scores), 0)
        best = np.argsort(-scores)[:k]
        results[str(label)] = pd.Series(terms[best[scores[best] > 0]], dtype="object")

    return pd.DataFrame(results).reindex(range(k))
