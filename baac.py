"""
baac.py
=======
Classes to load, clean and filter the French BAAC road-accident files
(Caract, Lieux, Usagers, Vehicules, vehicules-immatricule).

Design
------
- DataGouvLoader : talks to data.gouv.fr and returns raw DataFrames (I/O only).
- BAACDataset    : holds the five tables of ONE year, cleans them once, and
                   exposes filtering methods. Every filter returns a NEW
                   BAACDataset, so the original is never modified.

Nothing here knows about the périphérique: that logic lives in the analysis
script, so this module can be reused for any other area.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd
import requests

log = logging.getLogger(__name__)

DEFAULT_SLUG = (
    "bases-de-donnees-annuelles-des-accidents-corporels-"
    "de-la-circulation-routiere-annees-de-2005-a-2025"
)
API_URL = "https://www.data.gouv.fr/api/1/datasets/{slug}/"
RESOURCE_URL = "https://www.data.gouv.fr/fr/datasets/r/{rid}"

TABLE_KEYS = ("caract", "lieux", "usagers", "vehicules", "immat")
GEO_COLS = ["dep", "com", "commune_name"]


def classify_title(title: str) -> str | None:
    """Map a file title to one of TABLE_KEYS ('immat' must be tested first)."""
    t = title.lower()
    if "immat" in t:
        return "immat"
    for key in ("caract", "lieux", "usagers", "vehicules"):
        if key in t:
            return key
    return None


# ----------------------------------------------------------------------
# Loader
# ----------------------------------------------------------------------
class DataGouvLoader:
    """Downloads the CSV resources of one year from data.gouv.fr."""

    # Columns that must stay strings (leading zeros, '2A'/'2B', ids with spaces)
    STR_COLS = {"Num_Acc", "id_vehicule", "id_usager", "num_veh", "dep", "com"}

    def __init__(self, slug: str = DEFAULT_SLUG, timeout: int = 120):
        self.slug = slug
        self.timeout = timeout

    def list_resources(self, year: int) -> dict[str, dict]:
        """Return {table_key: resource_metadata} for the given year."""
        meta = requests.get(API_URL.format(slug=self.slug), timeout=self.timeout)
        meta.raise_for_status()
        found: dict[str, dict] = {}
        for res in meta.json()["resources"]:
            if res.get("format") == "csv" and str(year) in res["title"]:
                key = classify_title(res["title"])
                if key:
                    found[key] = res
        return found

    def _read_csv(self, content: bytes) -> pd.DataFrame:
        for enc in ("utf-8-sig", "latin-1"):
            try:
                header = content[:5000].decode(enc).splitlines()[0]
                sep = ";" if header.count(";") > header.count(",") else ","
                raw_cols = pd.read_csv(
                    io.BytesIO(content), sep=sep, encoding=enc, nrows=0
                ).columns
                dtype = {c: str for c in raw_cols if c.strip() in self.STR_COLS}
                return pd.read_csv(
                    io.BytesIO(content), sep=sep, encoding=enc,
                    dtype=dtype, low_memory=False,
                )
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not decode CSV with utf-8 or latin-1")

    def load(self, year: int) -> dict[str, pd.DataFrame]:
        """Download and parse every table of the year -> {table_key: DataFrame}."""
        tables = {}
        for key, res in self.list_resources(year).items():
            log.info("Downloading %s (%s)", key, res["title"])
            r = requests.get(RESOURCE_URL.format(rid=res["id"]), timeout=self.timeout)
            r.raise_for_status()
            tables[key] = self._read_csv(r.content)
        return tables


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class BAACDataset:
    """The BAAC tables of one year, cleaned and filterable.

    Tables are stored under the keys 'caract', 'lieux', 'usagers',
    'vehicules' and 'immat'. 'caract' is mandatory because it is the only
    table holding the geography (dep, com, lat, long).
    """

    def __init__(self, tables: Mapping[str, pd.DataFrame], year: int | None = None):
        if "caract" not in tables:
            raise ValueError("A 'caract' table is required (it holds dep/com).")
        unknown = set(tables) - set(TABLE_KEYS)
        if unknown:
            raise ValueError(f"Unknown table keys: {unknown}")
        self.year = year
        self._tables = {k: self._clean(k, df) for k, df in tables.items()}

    # ---- constructors -------------------------------------------------
    @classmethod
    def from_data_gouv(cls, year: int, loader: DataGouvLoader | None = None):
        loader = loader or DataGouvLoader()
        return cls(loader.load(year), year=year)

    @classmethod
    def from_titled_frames(cls, frames: Mapping[str, pd.DataFrame], year: int | None = None):
        """Build from a {file_title: DataFrame} dict, e.g. your notebook's `dfs`."""
        tables = {}
        for title, df in frames.items():
            key = classify_title(title)
            if key:
                tables[key] = df
        return cls(tables, year=year)

    # ---- cleaning -----------------------------------------------------
    @staticmethod
    def _clean(key: str, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = df.columns.str.strip()              # 'id_vehicule\n' -> 'id_vehicule'
        df = df.loc[:, ~df.columns.duplicated()]         # avoid 2-D column selections
        df["Num_Acc"] = df["Num_Acc"].astype(str).str.strip()
        if key == "caract":
            # NO zero-padding: 'com' is kept exactly as stored in the file
            df["dep"] = df["dep"].astype(str).str.strip()
            df["com"] = df["com"].astype(str).str.strip()
            for c in ("lat", "long"):                    # decimal commas -> floats
                df[c] = pd.to_numeric(
                    df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce"
                )
        return df

    # ---- access -------------------------------------------------------
    def table(self, key: str) -> pd.DataFrame:
        if key not in self._tables:
            raise KeyError(f"Table '{key}' not loaded. Available: {list(self._tables)}")
        return self._tables[key]

    caract = property(lambda self: self.table("caract"))
    lieux = property(lambda self: self.table("lieux"))
    usagers = property(lambda self: self.table("usagers"))
    vehicules = property(lambda self: self.table("vehicules"))
    immat = property(lambda self: self.table("immat"))

    @property
    def accident_ids(self) -> set[str]:
        return set(self.caract["Num_Acc"])

    def _derive(self, tables: Mapping[str, pd.DataFrame]) -> "BAACDataset":
        return BAACDataset(tables, year=self.year)

    # ---- filtering (each returns a new dataset) -----------------------
    def filter_by_accidents(self, accident_ids: Iterable[str]) -> "BAACDataset":
        ids = set(accident_ids)
        return self._derive(
            {k: df[df["Num_Acc"].isin(ids)] for k, df in self._tables.items()}
        )

    def filter_by_communes(
        self,
        communes: Iterable[str],
        include_departments: Iterable[str] = (),
    ) -> "BAACDataset":
        """Keep accidents whose `com` is in `communes`, or whose `dep` is in
        `include_departments` (e.g. ['75'] to also keep Paris)."""
        communes = {str(c) for c in communes}
        deps = {str(d) for d in include_departments}
        c = self.caract
        mask = c["com"].isin(communes) | c["dep"].isin(deps)
        if not mask.any():
            raise ValueError(
                "No accident matched. Check the format of `com` in your file. "
                f"Sample values: {c['com'].drop_duplicates().head(8).tolist()}"
            )
        not_found = sorted(communes - set(c.loc[mask, "com"]))
        if not_found:
            log.warning("Communes with no accident: %s", not_found)
        return self.filter_by_accidents(c.loc[mask, "Num_Acc"])

    def filter_by_departments(
        self,
        departments: Iterable[str],
    ) -> "BAACDataset":
        """Keep accidents whose `dep` is in `departments`.

        Example: ``ds.filter_by_departments(ILE_DE_FRANCE_DEPS)``
        """
        deps = {str(d) for d in departments}
        c = self.caract
        mask = c["dep"].isin(deps)
        if not mask.any():
            raise ValueError(
                "No accident matched the departments "
                f"{sorted(deps)}. Sample dep values in the data: "
                f"{c['dep'].drop_duplicates().head(10).tolist()}"
            )
        log.info(
            "Keeping %d accidents in departments %s (out of %d total)",
            mask.sum(), sorted(deps), len(c),
        )
        return self.filter_by_accidents(c.loc[mask, "Num_Acc"])

    # ---- enrichment ---------------------------------------------------
    @staticmethod
    def _label_communes(geo: pd.DataFrame, names, resolver) -> pd.Series:
        """Return a commune name for every row of `geo` (needs dep, com).

        1. exact match of `com` in `names`;
        2. else `resolver(dep, com)` if given (may return None);
        3. else 'Unknown (<com>)', and the unresolved codes are logged.
        """
        label = geo["com"].map(dict(names or {}))
        if resolver is not None:
            todo = label.isna()
            label.loc[todo] = [resolver(d, c) for d, c in
                               zip(geo.loc[todo, "dep"], geo.loc[todo, "com"])]
        unresolved = label.isna()
        if unresolved.any():
            log.warning("Unnamed communes: %s",
                        sorted(geo.loc[unresolved, "com"].unique()))
        return label.fillna("Unknown (" + geo["com"] + ")")

    def attach_geography(self, names: Mapping[str, str] | None = None,
                         resolver=None) -> "BAACDataset":
        """Copy dep / com / commune_name from Caract into every other table.

        `names`    : {com code: name}
        `resolver` : optional callable (dep, com) -> name | None, used for the
                     codes missing from `names` (e.g. all of dep 75 -> 'Paris').
        """
        geo = self.caract[["Num_Acc", "dep", "com"]].drop_duplicates("Num_Acc").copy()
        geo["commune_name"] = self._label_communes(geo, names, resolver)

        tables = {}
        for key, df in self._tables.items():
            if key == "caract":
                df = df.drop(columns=["commune_name"], errors="ignore").merge(
                    geo[["Num_Acc", "commune_name"]], on="Num_Acc",
                    how="left", validate="many_to_one")
            else:
                df = df.drop(columns=[c for c in GEO_COLS if c in df.columns])
                df = df.merge(geo, on="Num_Acc", how="left", validate="many_to_one")
            tables[key] = df
        return self._derive(tables)

    # ---- diagnostics / persistence -----------------------------------
    def summary(self) -> pd.DataFrame:
        return pd.DataFrame({
            "rows": {k: len(df) for k, df in self._tables.items()},
            "columns": {k: df.shape[1] for k, df in self._tables.items()},
            "unique_accidents": {k: df["Num_Acc"].nunique() for k, df in self._tables.items()},
        })

    def save(self, folder: str | Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        for key, df in self._tables.items():
            df.to_csv(folder / f"{key}_{self.year or 'x'}.csv", index=False)

    def __repr__(self) -> str:
        sizes = ", ".join(f"{k}={len(df):,}" for k, df in self._tables.items())
        return f"BAACDataset(year={self.year}, {sizes})"