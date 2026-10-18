"""Exploratory analysis of the training set: distributions and associations.

One function per planned analysis in the proposal; each comment says which decision it informs.
Plot functions show the figure and, if save_as is given, also save it to config.FIGURES_DIR
for the report and slides.

Owner: Leanne. Status: placeholder. Keep the signatures and return types.
"""
import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from src import config

TODO = "TODO (Leanne)"


TEXT_COL = getattr(config, "TEXT_COL", "event_description")
LABEL_COL = config.LABEL_COL
BRAND_COL = "brand_name"




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

# 1. Class counts -> class weights, merging rare classes, why macro-F1

def class_counts(labels: pd.Series) -> pd.DataFrame:
    """Index: class label, most frequent first. Columns: "count" (int), "share_%" (float)."""

    counts = labels.value_counts()
    shares = labels.value_counts(normalize=True) * 100
    result = pd.DataFrame({
        "count": counts,
        "share_%": shares
    })

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
    lengths = texts.fillna("").str.split().str.len()

    results = lengths.quantile(
        [p / 100 for p in percentiles]
    )

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
    """Description length per class (e.g. one box per class)."""
 
    #Creating a summary DataFrame
    summary = pd.DataFrame({
        "class": df[LABEL_COL],
        "words": _word_counts(df[TEXT_COL])
    })

    #Remove the rows with missing class labels
    summary = summary.dropna(subset=["class"])

    #Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    summary.boxplot(
        column="words",
        by="class",
        ax=ax,
        showfliers=False,
        rot=45
    )

    fig.suptitle("")
    ax.set_title("Description Length by IMDRF Category")
    ax.set_xlabel("IMDRF Category")
    ax.set_ylabel("Number of Words")

    plt.tight_layout()
    _save_plot(save_as)
    plt.show()



# Your conclusion should explain the trade-off between retaining information and 
# reducing memory consumption, computation, and padding.

# 3. Empty / very short descriptions -> cleaning: drop them or keep them

def short_description_summary(texts: pd.Series) -> pd.DataFrame:
    """Index: thresholds such as "<= 0 words", "<= 3 words". Columns: "reports" (int), "share_%" (float)."""
    lengths = _word_counts(texts)

    thresholds = [0, 3, 5, 10]

    results = []

    for t in thresholds:

        count = (lengths <= t).sum()

        percentage = count / len(lengths) * 100 if len(lengths) else 0

        results.append({
            "threshold": f"<= {t} words",
            "reports": int(count),
            "share_%": percentage
        })

    result = pd.DataFrame(results)

    return result.set_index("threshold")


# 4. Duplicate (template) descriptions -> how the split keeps them on one side

def duplicate_summary(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Returns (stats, top_templates).

    stats: named counts, e.g. "reports", "unique descriptions", "duplicate texts with >1 label".
    top_templates: the most repeated descriptions. Index: text. Columns: "count" (int),
    "labels" (dict of label -> count).
    """
    descriptions = (
        df[TEXT_COL]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Exclude empty descriptions from duplicate analysis
    valid = descriptions.ne("")

    temp = pd.DataFrame({
        "text": descriptions[valid],
        "label": df.loc[valid, LABEL_COL]
                   .fillna("(missing)")
                   .astype(str)
    })

    # Count how often each description appears
    counts = temp["text"].value_counts()

    duplicates = counts[counts > 1]

    # Check whether identical descriptions have different labels
    label_counts = temp.groupby("text")["label"].nunique()

    conflicting = (label_counts > 1).sum()

    # Summary statistics
    stats = pd.Series({
        "reports": len(df),
        "reports with non-empty descriptions": len(temp),
        "unique descriptions": len(counts),
        "duplicate texts": len(duplicates),
        "reports in duplicate groups": int(duplicates.sum()),
        "duplicate texts with >1 label": int(conflicting)
    })

    # Ten most repeated descriptions
    top_counts = duplicates.head(10)

    labels_per_text = []

    for text in top_counts.index:

        label_distribution = (
            temp.loc[temp["text"] == text, "label"]
            .value_counts()
            .to_dict()
        )

        labels_per_text.append(label_distribution)

    top_templates = pd.DataFrame({
        "count": top_counts,
        "labels": labels_per_text
    }, index=top_counts.index)

    top_templates.index.name = "text"

    return stats, top_templates


# 5. Reports by brand -> per-brand results, templated vs free text

def brand_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Index: brand. Columns: "reports" (int), "duplicate_%" (float), "median_words" (float)."""
    brands = (
        df[BRAND_COL]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
    )

    descriptions = (
        df[TEXT_COL]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    summary = pd.DataFrame({
        "brand": brands,
        "description": descriptions,
        "words": _word_counts(df[TEXT_COL])
    })

    # Count reports whose text is repeated within the same brand
    summary["is_duplicate"] = (
        summary["description"].ne("")
        & summary.duplicated(
            subset=["brand", "description"],
            keep=False
        )
    )

    result = summary.groupby("brand").agg(
        reports=("brand", "size"),
        duplicate_ratio=("is_duplicate", "mean"),
        median_words=("words", "median")
    )

    result["duplicate_%"] = (
        result["duplicate_ratio"] * 100
    )

    result = result.drop(columns="duplicate_ratio")

    return result[
        ["reports", "duplicate_%", "median_words"]
    ].sort_values("reports", ascending=False)


def plot_brand_by_class(df: pd.DataFrame, save_as: str | None = None) -> None:
    """Heatmap of each brand's share of reports in each class."""
    brands = (
        df[BRAND_COL]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
    )

    # Count reports for each brand and category
    counts = pd.crosstab(brands, df[LABEL_COL])

    if counts.empty:
        raise ValueError("No brand/category data available.")

    counts = counts.loc[
        counts.sum(axis=1).sort_values(ascending=False).index
    ]

    # Convert to within-brand percentages
    shares = counts.div(counts.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(
        figsize=(
            max(10, 0.8 * len(shares.columns)),
            max(5, 0.5 * len(shares))
        )
    )

    heatmap = ax.imshow(
        shares.to_numpy(),
        cmap="Blues",
        aspect="auto",
        vmin=0,
        vmax=100
    )

    ax.set_xticks(range(len(shares.columns)))
    ax.set_xticklabels(
        shares.columns,
        rotation=45,
        ha="right"
    )

    ax.set_yticks(range(len(shares.index)))
    ax.set_yticklabels(shares.index)

    ax.set_xlabel("IMDRF Category")
    ax.set_ylabel("CGM Brand")
    ax.set_title("IMDRF Category Distribution by Brand")

    fig.colorbar(
        heatmap,
        ax=ax,
        label="Share of Brand Reports (%)"
    )

    plt.tight_layout()
    _save_plot(save_as)
    plt.show()


# 6. Reports with more than one IMDRF category -> evaluate single vs multi separately

def count_imdrf_codes(imdrf_categories: pd.Series) -> pd.Series:
    """Number of distinct IMDRF codes listed in each report. Same index as the input, int values."""
    # Matching codes
    pattern = re.compile(r"\b[A-Z]\d{2,6}\b")

    counts = (
        imdrf_categories
        .fillna("")
        .astype(str)
        .str.upper()
        .map(
            lambda text: len(set(pattern.findall(text)))
        )
    )

    return counts.astype(int)


def plot_imdrf_code_counts(imdrf_categories: pd.Series, save_as: str | None = None) -> None:
    """Bar chart: how many reports list 1, 2, 3, ... IMDRF categories."""
    codes_per_report = count_imdrf_codes(imdrf_categories)

    counts = codes_per_report.value_counts().sort_index()

    plt.figure(figsize=(9, 5))

    plt.bar(
        counts.index.astype(str),
        counts.values
    )

    plt.xlabel("Number of Distinct IMDRF Codes")
    plt.ylabel("Number of Reports")
    plt.title("Number of IMDRF Codes per CGM Report")

    plt.tight_layout()
    _save_plot(save_as)
    plt.show()


# 7. Most frequent words / phrases per class -> cleaning steps, TF-IDF n-gram range

def top_ngrams_per_class(texts: pd.Series, labels: pd.Series, k: int = 15) -> pd.DataFrame:
    """One column per class, k rows: that class's most frequent words/phrases (str)."""
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have equal lengths")

    temp = pd.DataFrame({
        "text": (
            texts.reset_index(drop=True)
                 .fillna("")
                 .astype(str)
        ),
        "label": labels.reset_index(drop=True)
    })

    temp = temp.dropna(subset=["label"])

    # Keep important negation words
    stop_words = list(
        ENGLISH_STOP_WORDS - {"no", "not", "nor"}
    )

    results = {}

    for label, group in temp.groupby("label"):

        vectorizer = CountVectorizer(
            ngram_range=(1, 2),
            stop_words=stop_words,
            max_features=10000
        )

        try:
            # Converting texts into word/phrase counts
            matrix = vectorizer.fit_transform(group["text"])

            # Total frequency of each word/phrase
            frequencies = pd.Series(
                matrix.sum(axis=0).A1,
                index=vectorizer.get_feature_names_out()
            )

            # Select the k most frequent terms
            top_terms = (
                frequencies
                .sort_values(ascending=False)
                .head(k)
                .index
                .tolist()
            )

        except ValueError as error:
            if "empty vocabulary" not in str(error):
                raise
            top_terms = []

        results[str(label)] = pd.Series(
            top_terms,
            dtype="object"
        )

    return pd.DataFrame(results).reindex(range(k))
