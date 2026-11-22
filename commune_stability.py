"""
commune_stability.py
====================
Compares the top communes of two years, for the two rules (severity, vulnerability), to
check that they are a *recurrent* problem and not a one-year fluctuation.

For each rule it prints:
  1. how many of the top-N communes are the same in both years (and how many you would
     expect by pure chance);
  2. the year-to-year rank correlation over all communes (do high-rate communes stay high?);
  3. for every commune in either top-N: both rates, the change, a z-score telling whether the
     change is larger than random fluctuation, and a verdict (KEEP / REVIEW / DROP);
  4. a re-ranking on the two years pooled together (more accidents = more stable rates).

Usage (notebook, `by_year` comes from the pipeline cell):
    from commune_stability import YearComparison
    cmp = YearComparison(by_year[2022], by_year[2023], 2022, 2023,
                         min_accidents=30, top_n=10)
    results = cmp.report()

Nothing is plotted: everything is printed or returned as DataFrames.
"""
from __future__ import annotations

import math

import pandas as pd

from commune_rankings import SEVERITY, VULNERABILITY, RankingRule


def z_test(k1: float, n1: float, k2: float, n2: float, kind: str) -> tuple[float, float]:
    """Is k2/n2 different from k1/n1 by more than random fluctuation?

    Returns (z, p_value); z > 0 means the rate went UP from sample 1 to sample 2.
    kind = "proportion": k out of n accidents (vulnerability).
    kind = "rate"      : k persons over n accidents, Poisson-type counts (severity).
    Normal approximation with a pooled estimate: rough when counts are very small.
    """
    if n1 <= 0 or n2 <= 0:
        return float("nan"), float("nan")
    pooled = (k1 + k2) / (n1 + n2)
    variance = pooled * (1 - pooled) if kind == "proportion" else pooled
    se = math.sqrt(variance * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (k2 / n2 - k1 / n1) / se
    return z, math.erfc(abs(z) / math.sqrt(2))


class YearComparison:
    """Compares two by_commune() tables (year A vs year B) rule by rule."""

    RULES = (SEVERITY, VULNERABILITY)

    # numerator column and noise model for each rule
    SPEC = {
        "severity": {"count": "severe_persons", "kind": "rate"},
        "vulnerability": {"count": "vulnerable_accidents", "kind": "proportion"},
    }
    # "drastic" needs BOTH a big move (in points) and a z-score above z_threshold
    DEFAULT_MAX_CHANGE = {"severity": 10.0, "vulnerability": 10.0}

    def __init__(self, by_a: pd.DataFrame, by_b: pd.DataFrame, year_a: int, year_b: int,
                 min_accidents: int = 30, top_n: int = 10, near_k: int = 20,
                 z_threshold: float = 1.96, max_abs_change: dict | None = None):
        self.a, self.b = self._prepare(by_a), self._prepare(by_b)
        self.ya, self.yb = year_a, year_b
        self.min_accidents = min_accidents
        self.top_n = top_n
        self.near_k = near_k                      # "just outside the top" = rank <= near_k
        self.z_threshold = z_threshold
        self.max_abs_change = {**self.DEFAULT_MAX_CHANGE, **(max_abs_change or {})}

    @staticmethod
    def _prepare(by: pd.DataFrame) -> pd.DataFrame:
        out = by.copy()
        out["severe_persons"] = out["killed"] + out["hospitalised"]
        return out

    # ------------------------------------------------------------------
    # ranking
    # ------------------------------------------------------------------
    def ranking(self, by: pd.DataFrame, rule: RankingRule) -> pd.Series:
        """Rank (1 = highest) of every eligible commune; same tie-breaks as CommuneRanker."""
        pool = by[by["accidents"] >= self.min_accidents].dropna(subset=[rule.metric])
        order = (pool.reset_index()
                 .sort_values([rule.metric, "accidents", "com"], ascending=[False, False, True]))
        return pd.Series(range(1, len(order) + 1), index=order["com"].values, name="rank")

    # ------------------------------------------------------------------
    # 1-2. overlap and persistence
    # ------------------------------------------------------------------
    def global_stats(self, rule: RankingRule) -> dict:
        ra, rb = self.ranking(self.a, rule), self.ranking(self.b, rule)
        top_a, top_b = set(ra[ra <= self.top_n].index), set(rb[rb <= self.top_n].index)
        both_eligible = ra.index.intersection(rb.index)

        x, y = self.a.loc[both_eligible, rule.metric], self.b.loc[both_eligible, rule.metric]
        change = (y - x).abs()
        union = top_a | top_b
        return {
            "eligible_a": len(ra), "eligible_b": len(rb), "eligible_both": len(both_eligible),
            "top_a": top_a, "top_b": top_b,
            "in_both": top_a & top_b, "only_a": top_a - top_b, "only_b": top_b - top_a,
            "jaccard": len(top_a & top_b) / len(union) if union else float("nan"),
            "expected_by_chance": self.top_n * self.top_n / max(len(both_eligible), 1),
            "a_top_within_near_k_of_b": len(top_a & set(rb[rb <= self.near_k].index)),
            "b_top_within_near_k_of_a": len(top_b & set(ra[ra <= self.near_k].index)),
            "spearman": x.rank().corr(y.rank()),
            "median_abs_change": change.median(), "p90_abs_change": change.quantile(0.9),
        }

    # ------------------------------------------------------------------
    # 3. commune-by-commune table with verdicts
    # ------------------------------------------------------------------
    def table(self, rule: RankingRule) -> pd.DataFrame:
        spec = self.SPEC[rule.name]
        max_change = self.max_abs_change[rule.name]
        ra, rb = self.ranking(self.a, rule), self.ranking(self.b, rule)
        top_a, top_b = set(ra[ra <= self.top_n].index), set(rb[rb <= self.top_n].index)
        ya, yb = self.ya, self.yb

        rows = []
        for com in sorted(top_a | top_b):
            def pick(by, col, default=float("nan")):
                return by.at[com, col] if com in by.index else default

            n_a, n_b = pick(self.a, "accidents", 0), pick(self.b, "accidents", 0)
            k_a, k_b = pick(self.a, spec["count"], 0), pick(self.b, spec["count"], 0)
            rate_a, rate_b = pick(self.a, rule.metric), pick(self.b, rule.metric)
            change = rate_b - rate_a
            z, p = z_test(k_a, n_a, k_b, n_b, spec["kind"])
            drastic = bool(abs(change) >= max_change and abs(z) >= self.z_threshold)

            in_a, in_b = com in top_a, com in top_b
            rank_a, rank_b = ra.get(com, float("nan")), rb.get(com, float("nan"))
            if in_a and in_b:
                verdict = "REVIEW: recurrent but volatile" if drastic else "KEEP: recurrent"
            else:
                other = rank_b if in_a else rank_a
                if pd.isna(other):
                    verdict = "REVIEW: too few accidents the other year"
                elif other <= self.near_k and not drastic:
                    verdict = "REVIEW: borderline (just outside the top)"
                else:
                    verdict = "DROP: one-off"

            rows.append({
                "com": com,
                "dep": pick(self.a, "dep", pick(self.b, "dep", "")),
                f"top_{ya}": in_a, f"top_{yb}": in_b,
                f"rank_{ya}": rank_a, f"rank_{yb}": rank_b,
                f"acc_{ya}": int(n_a), f"acc_{yb}": int(n_b),
                f"rate_{ya}": rate_a, f"rate_{yb}": rate_b,
                "change": change,
                "rel_change_%": change / rate_a * 100 if rate_a and not pd.isna(rate_a) else float("nan"),
                "z": z, "p_value": p, "drastic": drastic, "verdict": verdict,
            })

        out = pd.DataFrame(rows)
        out["_order"] = (~(out[f"top_{ya}"] & out[f"top_{yb}"])).astype(int)
        out["_best"] = out[[f"rank_{ya}", f"rank_{yb}"]].min(axis=1).fillna(1e9)
        return (out.sort_values(["_order", "_best"]).drop(columns=["_order", "_best"])
                .reset_index(drop=True))

    # ------------------------------------------------------------------
    # 4. pooled ranking
    # ------------------------------------------------------------------
    def pooled_top(self, rule: RankingRule, min_accidents: int | None = None) -> pd.DataFrame:
        """Top-N when both years are added together (twice the accidents, steadier rates)."""
        cols = ["accidents", "killed", "hospitalised", "vulnerable_accidents"]
        pooled = self.a[cols].add(self.b[cols], fill_value=0)
        if rule.name == "severity":
            pooled["rate"] = ((pooled["killed"] + pooled["hospitalised"]) / pooled["accidents"] * 100).round(1)
        else:
            pooled["rate"] = (pooled["vulnerable_accidents"] / pooled["accidents"] * 100).round(1)

        threshold = min_accidents or 2 * self.min_accidents
        pool = pooled[pooled["accidents"] >= threshold].dropna(subset=["rate"])
        top = (pool.reset_index().sort_values(["rate", "accidents", "com"], ascending=[False, False, True])
               .head(self.top_n).reset_index(drop=True))
        top.insert(0, "rank", range(1, len(top) + 1))

        ra, rb = self.ranking(self.a, rule), self.ranking(self.b, rule)
        top[f"top_{self.ya}"] = top["com"].map(lambda c: bool(ra.get(c, 1e9) <= self.top_n))
        top[f"top_{self.yb}"] = top["com"].map(lambda c: bool(rb.get(c, 1e9) <= self.top_n))
        top.attrs["min_accidents"] = threshold
        return top[["rank", "com", "accidents", "rate", f"top_{self.ya}", f"top_{self.yb}"]]

    # ------------------------------------------------------------------
    # printing
    # ------------------------------------------------------------------
    @staticmethod
    def _show(df: pd.DataFrame) -> None:
        fmt = {"rate": "{:.1f}".format, "change": "{:+.1f}".format, "rel_change_%": "{:+.0f}".format,
               "z": "{:+.2f}".format, "p_value": "{:.3f}".format}
        for c in df.columns:
            if c.startswith("rate_"):
                fmt[c] = "{:.1f}".format
            if c.startswith("rank_"):
                fmt[c] = lambda v: "-" if pd.isna(v) else f"{int(v)}"
        print(df.to_string(index=False, formatters={k: v for k, v in fmt.items() if k in df.columns}))

    def report(self) -> dict:
        ya, yb, n = self.ya, self.yb, self.top_n
        results = {}
        for rule in self.RULES:
            s = self.global_stats(rule)
            print("=" * 100)
            print(f"{rule.name.upper()}: {ya} vs {yb}   [{rule.subtitle}]")
            print("=" * 100)

            print(f"\n1) Overlap of the top {n} (communes with >= {self.min_accidents} accidents only)")
            print(f"   eligible communes: {ya}: {s['eligible_a']} | {yb}: {s['eligible_b']} | both: {s['eligible_both']}")
            print(f"   in the top {n} BOTH years : {len(s['in_both'])}/{n}  {sorted(s['in_both'])}")
            print(f"   only in {ya}              : {len(s['only_a'])}  {sorted(s['only_a'])}")
            print(f"   only in {yb}              : {len(s['only_b'])}  {sorted(s['only_b'])}")
            print(f"   overlap (Jaccard)         : {s['jaccard']:.2f}")
            print(f"   expected by pure chance   : ~{s['expected_by_chance']:.1f} communes "
                  f"(if the two rankings were unrelated)")
            print(f"   {ya} top {n} found within the top {self.near_k} of {yb}: {s['a_top_within_near_k_of_b']}/{n}")
            print(f"   {yb} top {n} found within the top {self.near_k} of {ya}: {s['b_top_within_near_k_of_a']}/{n}")

            print(f"\n2) Do high-rate communes stay high? (all {s['eligible_both']} communes eligible both years)")
            print(f"   rank correlation (Spearman)     : {s['spearman']:.2f}   (1 = same order, 0 = unrelated)")
            print(f"   typical year-to-year change     : median {s['median_abs_change']:.1f} | "
                  f"90th percentile {s['p90_abs_change']:.1f} points")
            print(f"   'drastic' threshold used        : >= {self.max_abs_change[rule.name]:.0f} points "
                  f"AND |z| >= {self.z_threshold}")

            tbl = self.table(rule)
            print(f"\n3) Communes in the top {n} of either year")
            self._show(tbl.drop(columns=["verdict"]))

            print("\n   Verdicts:")
            for verdict, group in tbl.groupby("verdict"):
                print(f"   {verdict:<45} {len(group):>2}  {group['com'].tolist()}")

            pooled = self.pooled_top(rule)
            print(f"\n4) Re-ranked on {ya}+{yb} pooled (>= {2 * self.min_accidents} accidents)")
            self._show(pooled)

            results[rule.name] = {"stats": s, "table": tbl, "pooled": pooled}
        return results