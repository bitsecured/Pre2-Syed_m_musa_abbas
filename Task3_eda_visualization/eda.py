"""
Exploratory Data Analysis for processed news CSV files.

Run from Task3_eda:
    python eda.py

Reads CSV files from:
    ../Task2_preprocessing/data/processed/

Saves:
    outputs/visualizations/  -> PNG charts
    outputs/reports/         -> eda_report.xlsx
    outputs/statistics/      -> summary CSV files

Use --no-report to skip the Excel report.
"""

import argparse
import glob
import os
import re
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPECTED = [
    "title", "description", "date_of_publication",
    "author_name", "url", "source"
]
MOJIBAKE = re.compile(r"[ÂÃâ][\u0080-\u00BF€™“”‘’…]")
LATIN = re.compile(r"[A-Za-z]")

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR.parent / "Task2_preprocessing" / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"
CHART_DIR = OUTPUT_DIR / "visualizations"
REPORT_DIR = OUTPUT_DIR / "reports"
STATS_DIR = OUTPUT_DIR / "statistics"


def parse_dates(series):
    """Understand ISO dates and day/month/year dates."""
    series = series.astype(str).str.strip()
    iso = pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")
    dmy = pd.to_datetime(series, format="%d/%m/%Y", errors="coerce")
    return iso.fillna(dmy)


def words(series):
    return series.astype(str).str.split().str.len().fillna(0)


def analyse(path):
    name = os.path.basename(path)
    try:
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
    except Exception as exc:
        return None, None, f"could not read {name}: {exc}"

    missing_cols = [col for col in EXPECTED if col not in df.columns]
    n = len(df)
    stats = {"file": name, "rows": n, "columns": len(df.columns)}
    detail = {}

    if n == 0:
        return stats, detail, None

    blanks = {col: int((df[col].astype(str).str.strip() == "").sum())
              for col in df.columns}
    detail["blank values per column"] = pd.Series(blanks)
    stats["blank title"] = blanks.get("title", 0)
    stats["blank description"] = blanks.get("description", 0)

    if "url" in df:
        stats["duplicate urls"] = int(df["url"].duplicated().sum())
        stats["unique urls"] = int(df["url"].nunique())
        stats["invalid urls"] = int(
            (~df["url"].str.match(r"^https?://\S+\.\S+$")).sum()
        )
        detail["top websites (url domain)"] = (
            df["url"].str.extract(r"^https?://([^/]+)")[0]
            .value_counts().head(5)
        )

    if "title" in df:
        stats["duplicate titles"] = int(df["title"].duplicated().sum())

    if "date_of_publication" in df:
        dates = parse_dates(df["date_of_publication"])
        now = pd.Timestamp.now()
        stats["blank/invalid dates"] = int(dates.isna().sum())
        stats["future dates"] = int(
            (dates > now + pd.Timedelta(days=1)).sum()
        )
        valid_dates = dates[dates.notna() & (dates <= now + pd.Timedelta(days=1))]
        stats["oldest date"] = (
            valid_dates.min().strftime("%d/%m/%Y") if len(valid_dates) else "-"
        )
        stats["newest date"] = (
            valid_dates.max().strftime("%d/%m/%Y") if len(valid_dates) else "-"
        )
        stats["days covered"] = (
            int((valid_dates.max() - valid_dates.min()).days + 1)
            if len(valid_dates) else 0
        )
        detail["articles per month"] = (
            valid_dates.dt.to_period("M").value_counts().sort_index().tail(12)
        )
        detail["busiest days"] = (
            valid_dates.dt.strftime("%d/%m/%Y").value_counts().head(5)
        )

    if "author_name" in df:
        authors = df["author_name"].str.strip()
        stats["author Unknown/blank"] = int(
            ((authors == "") | (authors.str.lower() == "unknown")).sum()
        )
        named = authors[
            (authors != "") & (authors.str.lower() != "unknown")
        ]
        stats["unique authors"] = int(
            named.str.split(r",\s*").explode().nunique()
        )
        detail["top authors"] = (
            named.str.split(r",\s*").explode().value_counts().head(8)
        )

    for col in ("title", "description"):
        if col in df:
            word_counts = words(df[col])
            stats[f"{col} avg words"] = round(float(word_counts.mean()), 1)
            stats[f"{col} max words"] = int(word_counts.max())

    if "description" in df and "title" in df:
        same = (
            (df["description"].str.strip() == df["title"].str.strip())
            & (df["description"].str.strip() != "")
        )
        stats["description same as title"] = int(same.sum())

    if "title" in df:
        latin_share = df["title"].map(
            lambda title: len(LATIN.findall(title))
            / max(1, sum(char.isalpha() for char in title))
        )
        stats["non-Latin titles"] = int((latin_share < 0.5).sum())

    title_text = df["title"] if "title" in df else pd.Series("", index=df.index)
    desc_text = (
        df["description"] if "description" in df
        else pd.Series("", index=df.index)
    )
    text = title_text + " " + desc_text
    stats["broken accents (mojibake)"] = int(
        text.str.contains(MOJIBAKE, regex=True).sum()
    )
    stats["#NAME? damage"] = int(
        text.str.contains("#NAME?", regex=False).sum()
    )
    stats["extra spaces / newlines"] = int(
        text.str.contains(r"\s{2,}|\n", regex=True).sum()
    )

    if "source" in df:
        detail["sources"] = df["source"].value_counts().head(5)

    if missing_cols:
        stats["missing columns"] = ", ".join(missing_cols)

    return stats, detail, None


def print_file(stats, detail):
    print("\n" + "=" * 70)
    print(stats["file"])
    print("=" * 70)
    for key, value in stats.items():
        if key != "file":
            print(f"  {key:<28}{value}")
    for title, series in detail.items():
        print(f"\n  {title}:")
        for key, value in series.items():
            print(f"    {str(key)[:60]:<62}{value}")


def save_chart(fig, filename):
    fig.tight_layout()
    fig.savefig(CHART_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_visualizations(path, df):
    """Create useful charts for one processed source file."""
    stem = Path(path).stem
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem)

    if df.empty:
        return

    # 1. Missing values by column
    missing = df.replace(r"^\s*$", pd.NA, regex=True).isna().sum()
    if len(missing):
        fig, ax = plt.subplots(figsize=(9, 5))
        missing.sort_values(ascending=False).plot(kind="bar", ax=ax)
        ax.set_title(f"Missing Values - {stem}")
        ax.set_xlabel("Column")
        ax.set_ylabel("Missing records")
        ax.tick_params(axis="x", rotation=35)
        save_chart(fig, f"{safe_stem}_missing_values.png")

    # 2. Articles published per month
    if "date_of_publication" in df:
        dates = parse_dates(df["date_of_publication"]).dropna()
        if not dates.empty:
            monthly = dates.dt.to_period("M").value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(monthly.index.astype(str), monthly.values, marker="o")
            ax.set_title(f"Articles per Month - {stem}")
            ax.set_xlabel("Month")
            ax.set_ylabel("Number of articles")
            ax.tick_params(axis="x", rotation=45)
            save_chart(fig, f"{safe_stem}_articles_per_month.png")

    # 3. Top authors
    if "author_name" in df:
        authors = df["author_name"].astype(str).str.strip()
        authors = authors[
            (authors != "") & (authors.str.lower() != "unknown")
        ]
        top_authors = (
            authors.str.split(r",\s*").explode().value_counts().head(10)
        )
        if not top_authors.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            top_authors.sort_values().plot(kind="barh", ax=ax)
            ax.set_title(f"Top Authors - {stem}")
            ax.set_xlabel("Number of articles")
            ax.set_ylabel("Author")
            save_chart(fig, f"{safe_stem}_top_authors.png")

    # 4. Title and description word-count distributions
    available = [col for col in ("title", "description") if col in df]
    if available:
        fig, ax = plt.subplots(figsize=(9, 5))
        for col in available:
            counts = words(df[col])
            # Limit extreme outliers so the main distribution remains readable.
            upper = counts.quantile(0.99)
            counts[counts <= upper].plot(
                kind="hist", bins=30, alpha=0.55, ax=ax, label=col
            )
        ax.set_title(f"Text Length Distribution - {stem}")
        ax.set_xlabel("Words per record")
        ax.set_ylabel("Frequency")
        ax.legend()
        save_chart(fig, f"{safe_stem}_text_length.png")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "paths", nargs="*",
        help="optional CSV file(s) or folder; default is Task 2 processed data"
    )
    parser.add_argument(
        "--no-report", action="store_true",
        help="do not write the Excel report"
    )
    args = parser.parse_args()

    for folder in (CHART_DIR, REPORT_DIR, STATS_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    files = []
    selected_paths = args.paths or [str(INPUT_DIR)]
    for item in selected_paths:
        if os.path.isdir(item):
            files.extend(sorted(glob.glob(os.path.join(item, "*.csv"))))
        else:
            files.append(item)

    files = [
        file for file in files
        if os.path.isfile(file) and file.lower().endswith(".csv")
    ]

    if not files:
        print(f"No CSV files found in: {INPUT_DIR}")
        print("Check that Task 2 has saved processed CSV files in data/processed.")
        return

    results = []
    summaries = []
    all_urls = []

    for file in files:
        stats, detail, error = analyse(file)
        if error:
            print("!!", error)
            continue

        print_file(stats, detail)
        summaries.append(stats)
        results.append((os.path.basename(file), detail))

        try:
            df = pd.read_csv(file, dtype=str, encoding="utf-8-sig").fillna("")
            make_visualizations(file, df)
            if "url" in df:
                all_urls.extend(df["url"].dropna().astype(str).tolist())
        except Exception as exc:
            print(f"!! Could not create charts for {os.path.basename(file)}: {exc}")

    if not summaries:
        return

    summary = pd.DataFrame(summaries).set_index("file").T
    print("\n" + "#" * 70)
    print("COMPARISON OF ALL FILES")
    print("#" * 70)
    with pd.option_context(
        "display.width", 250,
        "display.max_columns", 50,
        "display.max_colwidth", 22
    ):
        print(summary.to_string())

    duplicate_across_files = (
        pd.Series(all_urls).duplicated().sum() if all_urls else 0
    )
    total_rows = sum(item.get("rows", 0) for item in summaries)
    print(
        f"\nTOTAL rows in all files: {total_rows}"
        f" | repeated URLs across/within files: {int(duplicate_across_files)}"
    )

    summary.to_csv(
        STATS_DIR / "eda_summary.csv",
        encoding="utf-8-sig"
    )

    # Overall comparison chart: number of articles per processed file.
    row_counts = {
        item["file"]: item["rows"] for item in summaries
    }
    fig, ax = plt.subplots(figsize=(11, 6))
    pd.Series(row_counts).sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Article Count by News Source")
    ax.set_xlabel("Number of articles")
    ax.set_ylabel("Processed CSV")
    save_chart(fig, "overall_article_counts.png")

    if not args.no_report:
        report_path = REPORT_DIR / "eda_report.xlsx"
        with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
            summary.to_excel(writer, sheet_name="summary")
            writer.sheets["summary"].column_dimensions["A"].width = 30
            for col in "BCDEFGHIJKLMNOPQRSTUVWXYZ":
                writer.sheets["summary"].column_dimensions[col].width = 20

            for name, detail in results:
                sheet = re.sub(
                    r"[\\/*?:\[\]]", "",
                    os.path.splitext(name)[0]
                )[:31] or "source"
                row = 0
                for title, series in detail.items():
                    output = series.rename(title).to_frame()
                    output.index = output.index.astype(str)
                    output.to_excel(
                        writer, sheet_name=sheet, startrow=row
                    )
                    row += len(series) + 3
                writer.sheets[sheet].column_dimensions["A"].width = 60
                writer.sheets[sheet].column_dimensions["B"].width = 18

        print(f"\nExcel report saved to: {report_path}")

    print(f"Visualizations saved to: {CHART_DIR}")
    print(f"Summary CSV saved to: {STATS_DIR / 'eda_summary.csv'}")


if __name__ == "__main__":
    main()
