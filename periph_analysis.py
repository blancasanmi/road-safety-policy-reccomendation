"""
periph_analysis.py
==================
Filters the BAAC data to the communes bordering the Paris périphérique and
aggregates all tables.

Usage:
    python periph_analysis.py --year 2025 --out output_periph
    python periph_analysis.py --year 2025 --include-paris
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from baac import BAACDataset

# INSEE codes written from memory: verify them with the geometry check
# (communes touching the Paris polygon) before relying on the results.
PERIPH_COMMUNES: dict[str, str] = {
    # 92 Hauts-de-Seine
    "92051": "Neuilly-sur-Seine", "92044": "Levallois-Perret", "92024": "Clichy",
    "92012": "Boulogne-Billancourt", "92040": "Issy-les-Moulineaux",
    "92075": "Vanves", "92046": "Malakoff", "92049": "Montrouge",
    # 93 Seine-Saint-Denis
    "93070": "Saint-Ouen-sur-Seine", "93066": "Saint-Denis", "93001": "Aubervilliers",
    "93055": "Pantin", "93061": "Le Pré-Saint-Gervais", "93045": "Les Lilas",
    "93006": "Bagnolet", "93048": "Montreuil",
    # 94 Val-de-Marne
    "94080": "Vincennes", "94067": "Saint-Mandé", "94018": "Charenton-le-Pont",
    "94041": "Ivry-sur-Seine", "94043": "Le Kremlin-Bicêtre", "94037": "Gentilly",
}
# Paris is stored as '75' in the `com` column of this file (checked manually);
# the 5-digit code is mapped too, in case other years differ.


def name_resolver(dep: str, com: str) -> str | None:
    """Fallback naming for codes not in PERIPH_COMMUNES.
    Every accident of department 75 is labelled 'Paris', whatever the code
    ('75', '75056', or an arrondissement code such as '75116')."""
    return "Paris" if dep == "75" else None


class PeripheryAggregator:
    """Builds accident-level and grouped tables from a filtered BAACDataset.

    The dataset must have geography attached (dataset.attach_geography(...)),
    so that every table carries `commune_name`.
    """

    GRAV = {1: "unharmed", 2: "killed", 3: "hospitalised", 4: "light_injury"}
    CATU = {1: "driver", 2: "passenger", 3: "pedestrian"}
    AGE_COL = "V - Age Véhicule"

    def __init__(self, dataset: BAACDataset):
        self.ds = dataset
        self._accidents: pd.DataFrame | None = None

    # ---- accident-level table (one row per accident) ------------------
    def accident_table(self) -> pd.DataFrame:
        if self._accidents is not None:
            return self._accidents

        base = self.ds.caract.copy()
        base["hour"] = pd.to_numeric(base["hrmn"].astype(str).str[:2], errors="coerce")

        lieux = self.ds.lieux.drop(columns=["dep", "com", "commune_name"], errors="ignore")
        base = base.merge(lieux.drop_duplicates("Num_Acc"), on="Num_Acc",
                          how="left", suffixes=("", "_lieux"), validate="one_to_one")

        u = self.ds.usagers
        flags = pd.DataFrame({
            "Num_Acc": u["Num_Acc"],
            "n_users": 1,
            "unharmed": u["grav"].eq(1).astype(int),
            "killed": u["grav"].eq(2).astype(int),
            "hospitalised": u["grav"].eq(3).astype(int),
            "light_injury": u["grav"].eq(4).astype(int),
            "pedestrians": u["catu"].eq(3).astype(int),
        })
        base = base.merge(flags.groupby("Num_Acc").sum(), on="Num_Acc", how="left")

        n_veh = self.ds.vehicules.groupby("Num_Acc").size().rename("n_vehicles")
        base = base.merge(n_veh, on="Num_Acc", how="left")

        try:
            immat = self.ds.immat
            if self.AGE_COL in immat.columns:
                age = immat.groupby("Num_Acc")[self.AGE_COL].mean().rename("mean_vehicle_age")
                base = base.merge(age, on="Num_Acc", how="left")
        except KeyError:
            pass

        count_cols = ["n_users", "unharmed", "killed", "hospitalised",
                      "light_injury", "pedestrians", "n_vehicles"]
        base[count_cols] = base[count_cols].fillna(0).astype(int)
        self._accidents = base
        return base

    # ---- grouped views -------------------------------------------------
    def by_commune(self) -> pd.DataFrame:
        a = self.accident_table()
        out = a.groupby("commune_name").agg(
            accidents=("Num_Acc", "size"),
            vehicles=("n_vehicles", "sum"),
            users=("n_users", "sum"),
            killed=("killed", "sum"),
            hospitalised=("hospitalised", "sum"),
            light_injury=("light_injury", "sum"),
            unharmed=("unharmed", "sum"),
            pedestrians=("pedestrians", "sum"),
        )
        out["severe_per_100_accidents"] = (
            (out["killed"] + out["hospitalised"]) / out["accidents"] * 100
        ).round(1)
        return out.sort_values("accidents", ascending=False)

    def by_month(self) -> pd.DataFrame:
        a = self.accident_table()
        return pd.crosstab(a["commune_name"], a["mois"], margins=True, margins_name="Total")

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
        return pd.crosstab(v["commune_name"], v["catv"])   # catv codes: see data dictionary

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


def build_periphery_dataset(dataset: BAACDataset, include_paris: bool = False) -> BAACDataset:
    """Filter a national dataset to the périphérique communes (+ optionally Paris)."""
    ring = dataset.filter_by_communes(
        PERIPH_COMMUNES.keys(),
        include_departments=["75"] if include_paris else (),
    )
    return ring.attach_geography(PERIPH_COMMUNES, resolver=name_resolver)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--include-paris", action="store_true")
    parser.add_argument("--out", default="output_periph")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    national = BAACDataset.from_data_gouv(args.year)
    print(national)

    ring = build_periphery_dataset(national, include_paris=args.include_paris)
    print(ring)
    print(ring.summary())

    agg = PeripheryAggregator(ring)
    print(agg.by_commune())
    agg.export(args.out)
    print(f"\nExported to {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()