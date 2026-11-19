"""
commune_analysis.py
===================
Filters the BAAC data to the whole Île-de-France region (all communes of
departments 75, 77, 78, 91, 92, 93, 94, 95) and aggregates all tables by
commune code (`com`, kept as it is in the file, no renaming) around two rules:

    1. Severity      -> severe_per_100_accidents
    2. Vulnerability -> pct_vulnerable_accidents (pedestrians and cyclists)

Usage:
    python commune_analysis.py --year 2025 --out output_communes
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import requests

from baac import BAACDataset

log = logging.getLogger(__name__)

IDF_DEPARTMENTS = ["75", "77", "78", "91", "92", "93", "94", "95"]


IDF_REGION_CODE = "11"   # INSEE code of the Île-de-France region


def fetch_idf_communes(region_code: str = IDF_REGION_CODE, timeout: int = 60,
                       with_names: bool = False) -> pd.DataFrame | None:
    """Official list of communes of a region (columns: com, dep[, name]).

    Used as the reference to also list communes with 0 accidents.
    Returns None if the call fails (no internet / API error).
    """
    fields = "code,codeDepartement" + (",nom" if with_names else "")
    url = f"https://geo.api.gouv.fr/communes?codeRegion={region_code}&fields={fields}"
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return pd.DataFrame(r.json()).rename(
            columns={"code": "com", "codeDepartement": "dep", "nom": "name"})
    except requests.RequestException as exc:
        log.warning("Could not fetch the official commune list (%s)", exc)
        return None


class CommuneAggregator:
    """Builds accident-level and grouped tables from a filtered BAACDataset.

    The dataset must have geography attached (dataset.attach_geography(...)),
    so that every table carries `dep` and `com`.
    """

    GRAV = {1: "unharmed", 2: "killed", 3: "hospitalised", 4: "light_injury"}
    CATU = {1: "driver", 2: "passenger", 3: "pedestrian"}
    AGE_COLS = ("V - Age Véhicule", "Age véhicule")   # 2025 / 2022-2024 registration files

    # Vulnerability = pedestrians + cyclists.
    # catu 3 = pedestrian. catv 1 = bicycle, 80 = e-bike (VAE).
    # Verify these codes against the 2025 data dictionary.
    PEDESTRIAN_CATU = 3
    CYCLIST_CATV = {1, 80}

    def __init__(self, dataset: BAACDataset,
                 reference_communes: pd.DataFrame | None = None):
        """`reference_communes`: DataFrame with columns `com` and `dep` listing
        every commune expected in the area (see fetch_idf_communes). When given,
        by_commune() also lists the communes with 0 accidents."""
        self.ds = dataset
        self.reference = reference_communes
        self._accidents: pd.DataFrame | None = None

    # ---- accident-level table (one row per accident) ------------------
    def accident_table(self) -> pd.DataFrame:
        if self._accidents is not None:
            return self._accidents

        base = self.ds.caract.copy()
        base["hour"] = pd.to_numeric(base["hrmn"].astype(str).str[:2], errors="coerce")

        lieux = self.ds.lieux.drop(columns=["dep", "com"], errors="ignore")
        base = base.merge(lieux.drop_duplicates("Num_Acc"), on="Num_Acc",
                          how="left", suffixes=("", "_lieux"), validate="one_to_one")

        # Users, with the category of the vehicle they were in (to spot cyclists)
        veh_cat = (self.ds.vehicules[["Num_Acc", "num_veh", "catv"]]
                   .drop_duplicates(["Num_Acc", "num_veh"]))
        u = self.ds.usagers.merge(veh_cat, on=["Num_Acc", "num_veh"], how="left")
        is_pedestrian = u["catu"].eq(self.PEDESTRIAN_CATU)
        is_cyclist = ~is_pedestrian & u["catv"].isin(self.CYCLIST_CATV)

        flags = pd.DataFrame({
            "Num_Acc": u["Num_Acc"],
            "n_users": 1,
            "unharmed": u["grav"].eq(1).astype(int),
            "killed": u["grav"].eq(2).astype(int),
            "hospitalised": u["grav"].eq(3).astype(int),
            "light_injury": u["grav"].eq(4).astype(int),
            "pedestrians": is_pedestrian.astype(int),
            "cyclists": is_cyclist.astype(int),
        })
        base = base.merge(flags.groupby("Num_Acc").sum(), on="Num_Acc", how="left")

        n_veh = self.ds.vehicules.groupby("Num_Acc").size().rename("n_vehicles")
        base = base.merge(n_veh, on="Num_Acc", how="left")

        try:
            immat = self.ds.immat
            age_col = next((c for c in self.AGE_COLS if c in immat.columns), None)
            if age_col:
                age = (pd.to_numeric(immat[age_col], errors="coerce")
                       .groupby(immat["Num_Acc"]).mean().rename("mean_vehicle_age"))
                base = base.merge(age, on="Num_Acc", how="left")
        except KeyError:
            pass

        count_cols = ["n_users", "unharmed", "killed", "hospitalised",
                      "light_injury", "pedestrians", "cyclists", "n_vehicles"]
        base[count_cols] = base[count_cols].fillna(0).astype(int)

        # Accident-level flag used by the vulnerability rule
        base["vulnerable_involved"] = (base["pedestrians"] + base["cyclists"]) > 0

        self._accidents = base
        return base

    # ---- grouped views -------------------------------------------------
    def by_commune(self) -> pd.DataFrame:
        a = self.accident_table()
        out = a.groupby("com").agg(
            dep=("dep", "first"),
            accidents=("Num_Acc", "size"),
            vehicles=("n_vehicles", "sum"),
            users=("n_users", "sum"),
            killed=("killed", "sum"),
            hospitalised=("hospitalised", "sum"),
            light_injury=("light_injury", "sum"),
            unharmed=("unharmed", "sum"),
            pedestrians=("pedestrians", "sum"),
            cyclists=("cyclists", "sum"),
            vulnerable_accidents=("vulnerable_involved", "sum"),
        )
        out = self._add_empty_communes(out)

        # Rates are NaN (undefined) for communes with 0 accidents, not 0.
        # Rule 1 - severity: killed + hospitalised people per 100 accidents
        out["severe_per_100_accidents"] = (
            (out["killed"] + out["hospitalised"]) / out["accidents"] * 100
        ).round(1)

        # Rule 2 - vulnerability: % of accidents involving at least one
        # pedestrian or cyclist
        out["pct_vulnerable_accidents"] = (
            out["vulnerable_accidents"] / out["accidents"] * 100
        ).round(1)

        return out.sort_values("accidents", ascending=False)

    def _add_empty_communes(self, out: pd.DataFrame) -> pd.DataFrame:
        """Append the reference communes that have no accident (all counts = 0)."""
        if self.reference is None:
            return out
        ref = self.reference.drop_duplicates("com").set_index("com")
        expected = set(ref.index)
        # Official Paris code is 75056; if the file codes Paris differently
        # ('75', '751xx'), do not report 75056 as an empty commune.
        if any(str(c).startswith("75") for c in out.index):
            expected.discard("75056")
        missing = sorted(expected - set(out.index))

        empty = pd.DataFrame(0, index=pd.Index(missing, name="com"),
                             columns=[c for c in out.columns if c != "dep"])
        empty.insert(0, "dep", ref.loc[missing, "dep"].values)
        out = pd.concat([out, empty])
        out["in_reference"] = out.index.isin(ref.index)   # False = code not in official list
        return out

    def by_month(self) -> pd.DataFrame:
        a = self.accident_table()
        return pd.crosstab(a["com"], a["mois"], margins=True, margins_name="Total")

    def by_hour(self) -> pd.DataFrame:
        a = self.accident_table()
        return a.groupby("hour").agg(
            accidents=("Num_Acc", "size"),
            killed=("killed", "sum"),
            hospitalised=("hospitalised", "sum"),
        )

    def by_user_category(self) -> pd.DataFrame:
        u = self.ds.usagers
        return pd.crosstab(
            u["catu"].map(self.CATU).fillna("other"),
            u["grav"].map(self.GRAV).fillna("unknown"),
            margins=True, margins_name="Total",
        )

    def by_vehicle_category(self) -> pd.DataFrame:
        v = self.ds.vehicules
        return pd.crosstab(v["com"], v["catv"])   # catv codes: see data dictionary

    # ---- output --------------------------------------------------------
    def all_aggregates(self) -> dict[str, pd.DataFrame]:
        return {
            "accidents": self.accident_table(),
            "by_commune": self.by_commune(),
            "by_month": self.by_month(),
            "by_hour": self.by_hour(),
            "by_user_category": self.by_user_category(),
            "by_vehicle_category": self.by_vehicle_category(),
        }

    def export(self, folder: str | Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        for name, df in self.all_aggregates().items():
            df.to_csv(folder / f"{name}.csv", index=name != "accidents")


def build_idf_dataset(dataset: BAACDataset) -> BAACDataset:
    """Keep every accident of Île-de-France and attach dep / com to all tables."""
    return dataset.filter_by_departments(IDF_DEPARTMENTS).attach_geography()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--out", default="output_communes")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    national = BAACDataset.from_data_gouv(args.year)
    print(national)

    idf = build_idf_dataset(national)
    print(idf)
    print(idf.summary())

    reference = fetch_idf_communes()
    if reference is None:
        log.warning("No reference list: communes with 0 accidents will not be listed.")
    agg = CommuneAggregator(idf, reference_communes=reference)
    print(agg.by_commune())
    agg.export(args.out)
    print(f"\nExported to {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()