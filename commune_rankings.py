"""
commune_rankings.py
===================
Applies the two rules to the commune table and saves the top-N of each:

    - Severity      : severe_per_100_accidents  (killed + hospitalised per 100 accidents)
    - Vulnerability : pct_vulnerable_accidents  (% of accidents with a pedestrian or cyclist)

For every rule, two files are written in the output folder (default `data/`),
with the year of the data in the name and in the figure title:
    top10_severity_2025.csv / .png
    top10_vulnerability_2025.csv / .png

Notebook usage:
    from commune_rankings import CommuneRanker
    ranker = CommuneRanker(agg.by_commune(), year=2025, min_accidents=30)
    ranker.save("data")

Command line:
    python commune_rankings.py --year 2025 --min-accidents 30 --top-n 10 --out data
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from baac import BAACDataset
from commune_analysis import CommuneAggregator, build_idf_dataset, fetch_idf_communes

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Rules
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class RankingRule:
    """One ranking: which column to sort on and which columns to show."""
    name: str                  # used in file names
    title: str                 # e.g. "severity"
    subtitle: str              # one-line definition shown under the title
    metric: str                # column of by_commune() used to rank (higher = higher priority)
    columns: dict              # {by_commune column: header shown in the table}
    suffix: str = ""           # appended to the metric value (e.g. " %")


SEVERITY = RankingRule(
    name="severity",
    title="severity",
    subtitle="Killed + hospitalised people per 100 accidents",
    metric="severe_per_100_accidents",
    columns={
        "dep": "Dept",
        "accidents": "Accidents",
        "killed": "Killed",
        "hospitalised": "Hospitalised",
        "severe_per_100_accidents": "Severe per\n100 accidents",
    },
)

VULNERABILITY = RankingRule(
    name="vulnerability",
    title="vulnerability",
    subtitle="Share of accidents involving at least one pedestrian or cyclist",
    metric="pct_vulnerable_accidents",
    suffix=" %",
    columns={
        "dep": "Dept",
        "accidents": "Accidents",
        "pedestrians": "Pedestrians",
        "cyclists": "Cyclists",
        "vulnerable_accidents": "Accidents with\nped. or cyclist",
        "pct_vulnerable_accidents": "% vulnerable\naccidents",
    },
)


# ----------------------------------------------------------------------
# Ranker
# ----------------------------------------------------------------------
class CommuneRanker:
    """Selects, renders and saves the top communes for each rule."""

    RULES = (SEVERITY, VULNERABILITY)

    LINE_WIDTH = 0.8        # plain black rules, no fills

    def __init__(self, by_commune: pd.DataFrame, year: int,
                 min_accidents: int = 30, top_n: int = 10):
        needed = {"accidents", "dep"} | {c for r in self.RULES for c in r.columns}
        missing = needed - set(by_commune.columns)
        if missing:
            raise ValueError(f"by_commune() is missing columns: {sorted(missing)}")
        self.by = by_commune
        self.year = year
        self.min_accidents = min_accidents
        self.top_n = top_n

    # ---- selection -----------------------------------------------------
    def top(self, rule: RankingRule) -> pd.DataFrame:
        """Top-N communes for a rule (rank, com, then the rule's columns).

        Communes with fewer than `min_accidents` accidents are excluded:
        a rate over 2 or 3 accidents is too unstable to rank on.
        Ties on the metric are broken by the number of accidents, then by code.
        """
        pool = self.by[self.by["accidents"] >= self.min_accidents].dropna(subset=[rule.metric])
        if len(pool) < self.top_n:
            log.warning("Only %d communes have >= %d accidents (top_n=%d).",
                        len(pool), self.min_accidents, self.top_n)
        df = (pool.reset_index()
              .sort_values([rule.metric, "accidents", "com"], ascending=[False, False, True])
              .head(self.top_n))
        df.insert(0, "rank", range(1, len(df) + 1))
        return df[["rank", "com", *rule.columns]].reset_index(drop=True)

    # ---- rendering -----------------------------------------------------
    @staticmethod
    def _fmt(col: str, value, rule: RankingRule) -> str:
        if col == rule.metric:
            return f"{value:.1f}{rule.suffix}"
        if col in ("rank", "com", "dep"):
            return str(value)
        return f"{int(value):,}"

    def render(self, rule: RankingRule) -> plt.Figure:
        """Plain report-style table: black text, three horizontal rules, no colours."""
        df = self.top(rule)
        headers = ["Rank", "Commune"] + list(rule.columns.values())
        rows = [[self._fmt(c, r[c], rule) for c in ["rank", "com", *rule.columns]]
                for _, r in df.iterrows()]

        n_cols, n_rows = len(headers), len(rows)
        # Layout in inches, so the figure hugs the table whatever the number of rows
        row_h, top_in, bottom_in = 0.30, 1.0, 0.8
        table_h = (n_rows + 1.5) * row_h                  # header row is 1.5 rows tall
        height = table_h + top_in + bottom_in
        fig = plt.figure(figsize=(1.3 * n_cols + 0.8, height), facecolor="white")
        ax = fig.add_axes([0.03, bottom_in / height, 0.94, table_h / height])
        ax.axis("off")

        tbl = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="upper center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10.5)
        tbl.scale(1, 1.6)
        for (i, j), cell in tbl.get_celld().items():
            cell.set_facecolor("none")
            cell.set_edgecolor("black")
            cell.set_linewidth(self.LINE_WIDTH)
            if i == 0:                                    # header: rule above and below
                cell.visible_edges = "TB"
                cell.set_text_props(fontweight="bold")
                cell.set_height(cell.get_height() * 1.5)
            elif i == n_rows:                             # last row: closing rule
                cell.visible_edges = "B"
            else:
                cell.visible_edges = "open"

        fig.text(0.03, 1 - 0.2 / height, f"Top {self.top_n} communes by {rule.title}, {self.year}",
                 fontsize=13, fontweight="bold", ha="left", va="top")
        fig.text(0.03, 1 - 0.55 / height, rule.subtitle, fontsize=10, ha="left", va="top")

        n_with = int((self.by["accidents"] > 0).sum())
        n_pool = int((self.by["accidents"] >= self.min_accidents).sum())
        fig.text(0.03, 0.2 / height,
                 f"Île-de-France, BAAC {self.year}. Only communes with ≥ {self.min_accidents} accidents "
                 f"are ranked ({n_pool} of {n_with} communes with accidents).\n"
                 "Ties are broken by the number of accidents.",
                 fontsize=8.5, color="#333333", ha="left", va="bottom")
        return fig

    # ---- output --------------------------------------------------------
    def save(self, folder: str | Path = "data") -> dict[str, Path]:
        """Write one CSV and one PNG per rule (year in the name); return the paths."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        for rule in self.RULES:
            stem = f"top{self.top_n}_{rule.name}_{self.year}"
            self.top(rule).to_csv(folder / f"{stem}.csv", index=False)
            fig = self.render(rule)
            fig.savefig(folder / f"{stem}.png", dpi=200, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            paths[f"{rule.name}_csv"] = folder / f"{stem}.csv"
            paths[f"{rule.name}_png"] = folder / f"{stem}.png"
        return paths


# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--min-accidents", type=int, default=30)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--out", default="data")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    national = BAACDataset.from_data_gouv(args.year)
    idf = build_idf_dataset(national)
    agg = CommuneAggregator(idf, reference_communes=fetch_idf_communes())

    ranker = CommuneRanker(agg.by_commune(), year=args.year,
                           min_accidents=args.min_accidents, top_n=args.top_n)
    for rule in ranker.RULES:
        print(f"\n--- Top {args.top_n} by {rule.title} ---")
        print(ranker.top(rule).to_string(index=False))
    for label, path in ranker.save(args.out).items():
        print(f"saved {label}: {path.resolve()}")


if __name__ == "__main__":
    main()