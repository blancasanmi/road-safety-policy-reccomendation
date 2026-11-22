"""
vehicle_categories.py
=====================
Which vehicle categories (`catv`) exist in the Vehicules tables, and can we see e-scooters?

Prints, for the years you pass:
  1. every `catv` code with its count per year (whole country) and a label;
  2. the e-scooter codes only (EDP = engin de déplacement personnel), with their share of vehicles;
  3. who rides them in Île-de-France (driver / passenger / pedestrian) and how badly they were hurt;
  4. their `motor` field (type of motorisation).

The labels follow the standard BAAC nomenclature used since 2019. They were typed from memory:
check them against the official data dictionary. A code that is not in the list below is printed
as "NOT IN MY LIST", which is how you spot a category that is new or different in a given year.

Usage (notebook, `national` comes from the loading cell):
    from vehicle_categories import check_vehicle_categories
    res = check_vehicle_categories(national)
"""
from __future__ import annotations

import pandas as pd

from commune_analysis import build_idf_dataset

CATV_LABELS = {
    0: "Undetermined", 1: "Bicycle", 2: "Moped (< 50 cc)", 3: "Voiturette (light quadricycle)",
    7: "Car (VL) alone", 10: "Van (VU) alone", 13: "HGV 3.5-7.5 t", 14: "HGV > 7.5 t",
    15: "HGV > 3.5 t + trailer", 16: "Road tractor alone", 17: "Road tractor + semi-trailer",
    20: "Special machinery", 21: "Farm tractor",
    30: "Scooter < 50 cc", 31: "Motorcycle 50-125 cc", 32: "Scooter 50-125 cc",
    33: "Motorcycle > 125 cc", 34: "Scooter > 125 cc", 35: "Light quad (<= 50 cc)",
    36: "Heavy quad (> 50 cc)", 37: "Bus", 38: "Coach", 39: "Train", 40: "Tram",
    41: "3-wheeler <= 50 cc", 42: "3-wheeler 50-125 cc", 43: "3-wheeler > 125 cc",
    50: "EDP motorised (e-scooter...)", 60: "EDP non-motorised (kick scooter...)",
    80: "E-bike (VAE)", 99: "Other vehicle",
}
CATU_LABELS = {1: "driver", 2: "passenger", 3: "pedestrian"}
GRAV_LABELS = {1: "unharmed", 2: "killed", 3: "hospitalised", 4: "light_injury"}


def vehicle_category_table(national: dict) -> pd.DataFrame:
    """Count of vehicles per catv code (rows) and year (columns), with a label."""
    counts = {}
    for year, ds in national.items():
        catv = pd.to_numeric(ds.vehicules["catv"], errors="coerce").astype("Int64")
        counts[year] = catv.value_counts(dropna=False)
    table = pd.DataFrame(counts).fillna(0).astype(int).sort_index()
    table.index.name = "catv"
    labels = ["(missing)" if pd.isna(c) else CATV_LABELS.get(int(c), "NOT IN MY LIST")
              for c in table.index]
    table.insert(0, "label", labels)
    return table


def edp_users(ds, edp_codes: tuple[int, ...]) -> pd.DataFrame:
    """Users of the given catv codes in Île-de-France, with their role and severity."""
    idf = build_idf_dataset(ds)
    veh = (idf.vehicules[["Num_Acc", "num_veh", "catv"]]
           .drop_duplicates(["Num_Acc", "num_veh"]))
    users = idf.usagers.merge(veh, on=["Num_Acc", "num_veh"], how="left")
    users["catv"] = pd.to_numeric(users["catv"], errors="coerce")
    return users[users["catv"].isin(edp_codes)]


def check_vehicle_categories(national: dict, edp_codes: tuple[int, ...] = (50, 60)) -> dict:
    """Print the category listing and the e-scooter checks; return the tables."""
    years = list(national)
    out: dict = {}

    # 1. every category -------------------------------------------------------
    table = vehicle_category_table(national)
    print("=" * 90)
    print("1) Vehicle categories (catv), number of vehicles per year, whole country")
    print("=" * 90)
    print(table.to_string())
    unknown = table[table["label"] == "NOT IN MY LIST"]
    print(f"\nCodes not in my label list: {unknown.index.tolist() or 'none'}")
    out["table"] = table

    # 2. e-scooter codes ------------------------------------------------------
    print("\n" + "=" * 90)
    print(f"2) E-scooter codes {edp_codes}: vehicles and share of all vehicles")
    print("=" * 90)
    edp = table[table.index.isin(edp_codes)]
    if edp.empty:
        print(f"No vehicle with catv in {edp_codes} in any of the years {years}.")
        print("-> e-scooters are NOT identifiable in these files under those codes.")
    else:
        totals = table[years].sum()
        shares = (edp[years] / totals * 100).round(3)
        print("count:")
        print(edp.to_string())
        print("\nshare of all vehicles (%):")
        print(shares.to_string())
        missing_years = [y for y in years if edp[y].sum() == 0]
        if missing_years:
            print(f"\nWARNING: no EDP vehicle at all in {missing_years}.")
    out["edp"] = edp

    # 3. who rides them, Île-de-France ---------------------------------------------
    print("\n" + "=" * 90)
    print("3) Users of these codes in Île-de-France: role and severity")
    print("=" * 90)
    rows = []
    for year, ds in national.items():
        u = edp_users(ds, edp_codes)
        for code, g in u.groupby("catv"):
            rows.append({
                "year": year, "catv": int(code), "label": CATV_LABELS.get(int(code), "?"),
                "users": len(g),
                **{CATU_LABELS[k]: int((g["catu"] == k).sum()) for k in CATU_LABELS},
                **{GRAV_LABELS[k]: int((g["grav"] == k).sum()) for k in GRAV_LABELS},
            })
    users = pd.DataFrame(rows)
    if users.empty:
        print("No user linked to these codes in Île-de-France.")
    else:
        print(users.to_string(index=False))
        print("\nTo compare with official figures: injured = hospitalised + light_injury.")
        print("If `pedestrian` is 0 for catv 60, kick-scooter users may be coded as pedestrians "
              "(catu = 3, no vehicle), in which case they are already inside the pedestrian count.")
    out["users"] = users

    # 4. motorisation ---------------------------------------------------------------
    print("\n" + "=" * 90)
    print("4) `motor` field (type of motorisation) of these vehicles in Île-de-France")
    print("=" * 90)
    parts = []
    for year, ds in national.items():
        v = build_idf_dataset(ds).vehicules
        v = v[pd.to_numeric(v["catv"], errors="coerce").isin(edp_codes)]
        if len(v) and "motor" in v.columns:
            parts.append(v.assign(year=year, catv=pd.to_numeric(v["catv"], errors="coerce").astype(int)))
    if parts:
        motor = (pd.concat(parts).groupby(["year", "catv", "motor"]).size()
                 .rename("vehicles").reset_index())
        print(motor.to_string(index=False))
        out["motor"] = motor
    else:
        print("Nothing to show.")
    return out