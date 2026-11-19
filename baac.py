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
import re
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
RESOURCE_URL = "https://www.data.gouv.fr/api/1/datasets/r/{rid}"   # permalink used by the site itself

TABLE_KEYS = ("caract", "lieux", "usagers", "vehicules", "immat")
CORE_TABLES = ("caract", "lieux", "usagers", "vehicules")   # every year must have these
GEO_COLS = ["dep", "com", "commune_name"]


def classify_title(title: str) -> str | None:
    """Map a file title to one of TABLE_KEYS ('immat' must be tested first)."""
    t = title.lower()
    if "immat" in t:
        return "immat"
    if "caract" in t or "carct" in t:        # 'carcteristiques' (typo) exists in some years
        return "caract"
    for key in ("lieux", "usagers", "vehicules"):
        if key in t:
            return key
    return None


# Some years/files name the accident key differently ('Accident_Id' in the 2022 Caract file,
# 'Id_accident' in the 2022-2024 vehicle registration files)
COLUMN_ALIASES = {"accident_id": "Num_Acc", "id_accident": "Num_Acc"}


def normalise_column(name) -> str:
    """Clean a column name: remove BOM remnants and whitespace, map known aliases.

    A UTF-8 BOM read with the wrong encoding shows up as 'ï»¿Num_Acc' (or '\ufeffNum_Acc'),
    and a header cell can contain a stray newline ('id_vehicule\n').
    """
    name = str(name).replace("\ufeff", "").replace("ï»¿", "").strip()
    return COLUMN_ALIASES.get(name.lower(), name)


# ----------------------------------------------------------------------
# Loader
# ----------------------------------------------------------------------
class DataGouvLoader:
    """Downloads the CSV resources of one year from data.gouv.fr.

    cache_dir : if given, every raw CSV is stored there and reused on later runs
                (the file name contains the resource's last-modified stamp, so a
                republished file is downloaded again).
    `sources` : after load(), {table_key: file title actually used}.
    """

    # Columns that must stay strings (leading zeros, '2A'/'2B', ids with spaces)
    STR_COLS = {"Num_Acc", "id_vehicule", "id_usager", "num_veh", "dep", "com"}

    def __init__(self, slug: str = DEFAULT_SLUG, timeout: int = 120,
                 cache_dir: str | Path | None = None):
        self.slug = slug
        self.timeout = timeout
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.sources: dict[str, str] = {}

    @staticmethod
    def _title_has_year(title: str, year: int) -> bool:
        """True if `year` appears as a whole number in the title (2024 != 20240)."""
        return re.search(rf"(?<!\d){year}(?!\d)", title or "") is not None

    def list_resources(self, year: int) -> dict[str, dict]:
        """Return {table_key: resource_metadata} for the given year."""
        meta = requests.get(API_URL.format(slug=self.slug), timeout=self.timeout)
        meta.raise_for_status()
        csvs = [r for r in meta.json()["resources"]
                if (r.get("format") or "").lower() == "csv"
                and self._title_has_year(r.get("title"), year)]

        found: dict[str, dict] = {}
        # oldest first, so that on a duplicate the most recently modified file wins
        for res in sorted(csvs, key=lambda r: r.get("last_modified") or ""):
            key = classify_title(res["title"])
            if key is None:
                log.warning("Ignoring unrecognised file for %s: %s", year, res["title"])
                continue
            if key in found:
                log.warning("Two files match '%s' for %s: '%s' and '%s'; using the latter.",
                            key, year, found[key]["title"], res["title"])
            found[key] = res

        missing = [k for k in CORE_TABLES if k not in found]
        if missing:
            raise ValueError(
                f"No file found for {missing} in {year}. "
                f"CSV titles containing {year}: {[r['title'] for r in csvs] or 'none'}")
        return found

    def _read_csv(self, content: bytes) -> pd.DataFrame:
        first_line = content.split(b"\n", 1)[0]          # only the header, never a cut multi-byte char
        for enc in ("utf-8-sig", "latin-1"):
            try:
                header = first_line.decode(enc)
                sep = ";" if header.count(";") > header.count(",") else ","
                raw_cols = pd.read_csv(
                    io.BytesIO(content), sep=sep, encoding=enc, nrows=0
                ).columns
                dtype = {c: str for c in raw_cols if normalise_column(c) in self.STR_COLS}
                return pd.read_csv(
                    io.BytesIO(content), sep=sep, encoding=enc,
                    dtype=dtype, low_memory=False,
                )
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not decode CSV with utf-8 or latin-1")

    def _download(self, res: dict) -> bytes:
        cache_file = None
        if self.cache_dir:
            stamp = re.sub(r"[^0-9A-Za-z]", "", res.get("last_modified") or "na")
            cache_file = self.cache_dir / f"{res['id']}_{stamp}.csv"
            if cache_file.exists():
                return cache_file.read_bytes()
        r = requests.get(RESOURCE_URL.format(rid=res["id"]), timeout=self.timeout)
        r.raise_for_status()
        if cache_file:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_bytes(r.content)
        return r.content

    def load(self, year: int) -> dict[str, pd.DataFrame]:
        """Download and parse every table of the year -> {table_key: DataFrame}."""
        resources = self.list_resources(year)
        self.sources = {k: r["title"] for k, r in resources.items()}
        tables = {}
        for key, res in resources.items():
            log.info("%s: reading %s", year, res["title"])
            tables[key] = self._read_csv(self._download(res))
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
        self.sources: dict[str, str] = {}      # {table_key: file title}, set by from_data_gouv
        self._tables = {}
        for key, df in tables.items():
            try:
                self._tables[key] = self._clean(key, df)
            except ValueError as exc:
                if key in CORE_TABLES:
                    raise
                log.warning("Skipping optional table '%s': %s", key, exc)

    # ---- constructors -------------------------------------------------
    @classmethod
    def from_data_gouv(cls, year: int, loader: DataGouvLoader | None = None):
        loader = loader or DataGouvLoader()
        ds = cls(loader.load(year), year=year)
        ds.sources = dict(loader.sources)
        ds.check_year()          # raises if the files are not really those of `year`
        return ds

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
        df.columns = [normalise_column(c) for c in df.columns]   # BOM, 'id_vehicule\n', Accident_Id
        df = df.loc[:, ~df.columns.duplicated()]                 # avoid 2-D column selections
        if "Num_Acc" not in df.columns:
            raise ValueError(
                f"Table '{key}' has no accident key column (Num_Acc / Accident_Id). "
                f"Columns found: {list(df.columns)[:12]}")
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

    @property
    def table_keys(self) -> tuple[str, ...]:
        return tuple(self._tables)

    def check_year(self) -> None:
        """Raise if the tables are not those of `self.year`.

        Num_Acc always starts with the year of the accident (e.g. 2024xxxxxxxx),
        so a table from another year is detected immediately. Also checks that the
        other tables only refer to accidents present in Caract.
        """
        if self.year is None:
            return
        prefix, caract_ids = str(self.year), self.accident_ids
        for key, df in list(self._tables.items()):
            years = df["Num_Acc"].str[:4]
            share = years.eq(prefix).mean()
            orphans = 0.0 if key == "caract" else (~df["Num_Acc"].isin(caract_ids)).mean()
            problem = None
            if share < 0.99:
                problem = (f"only {share:.0%} of its Num_Acc start with {prefix} "
                           f"(found {years.value_counts().head(3).to_dict()})")
            elif orphans > 0.01:
                problem = f"{orphans:.0%} of its rows refer to accidents missing from 'caract'"
            if problem is None:
                continue
            if key in CORE_TABLES:
                raise ValueError(f"Table '{key}' does not look like {self.year}: {problem}. "
                                 f"Files used: {self.sources}")
            log.warning("Dropping optional table '%s' for %s: %s.", key, self.year, problem)
            del self._tables[key]

    def _derive(self, tables: Mapping[str, pd.DataFrame]) -> "BAACDataset":
        new = BAACDataset(tables, year=self.year)
        new.sources = self.sources
        return new

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

    def filter_by_departments(self, departments: Iterable[str]) -> "BAACDataset":
        """Keep every accident whose `dep` is in `departments` (all its communes)."""
        deps = {str(d) for d in departments}
        c = self.caract
        mask = c["dep"].isin(deps)
        if not mask.any():
            raise ValueError(
                "No accident matched. Check the format of `dep` in your file. "
                f"Sample values: {c['dep'].drop_duplicates().head(10).tolist()}"
            )
        missing = sorted(deps - set(c.loc[mask, "dep"]))
        if missing:
            log.warning("Departments with no accident: %s", missing)
        return self.filter_by_accidents(c.loc[mask, "Num_Acc"])

    # ---- enrichment ---------------------------------------------------
    @staticmethod
    def _label_communes(geo: pd.DataFrame, names, resolver) -> pd.Series:
        """Return a commune name for every row of `geo` (needs dep, com).

        1. exact match of `com` in `names`;
        2. else `resolver(dep, com)` if given (may return None);
        3. else 'Unknown (<com>)', and the unresolved codes are logged.
        """
        label = geo["com"].map(dict(names or {})).astype(object)
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
        """Copy dep / com from Caract into every other table.

        If `names` ({com code: name}) and/or `resolver` ((dep, com) -> name | None)
        is given, a `commune_name` column is added too. Otherwise communes stay
        identified by their `com` code only.
        """
        geo = self.caract[["Num_Acc", "dep", "com"]].drop_duplicates("Num_Acc").copy()
        named = names is not None or resolver is not None
        if named:
            geo["commune_name"] = self._label_communes(geo, names, resolver)

        tables = {}
        for key, df in self._tables.items():
            if key == "caract":
                df = df.drop(columns=["commune_name"], errors="ignore")
                if named:
                    df = df.merge(geo[["Num_Acc", "commune_name"]], on="Num_Acc",
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