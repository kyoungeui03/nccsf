#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "data" / "rhc" / "raw_rhc.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "rhc" / "cleaned_rhc.csv"

MCF_POLICY_FEATURES = [
    "adld3pc",
    "age",
    "aps1",
    "scoma1",
    "meanbp1",
    "surv2md1",
    "dnr1",
    "cat1",
]

MCF_ALL_CONFOUNDERS = [
    "adld3pc",
    "age",
    "alb1",
    "amihx",
    "aps1",
    "bili1",
    "ca",
    "card",
    "cardiohx",
    "cat1",
    "cat2",
    "cat2_miss",
    "chfhx",
    "chrpulhx",
    "crea1",
    "das2d3pc",
    "dementhx",
    "dnr1",
    "edu",
    "gastr",
    "gibledhx",
    "hema",
    "hema1",
    "hrt1",
    "immunhx",
    "income",
    "liverhx",
    "malighx",
    "meanbp1",
    "meta",
    "neuro",
    "ninsclas",
    "ortho",
    "paco21",
    "pafi1",
    "ph1",
    "pot1",
    "psychhx",
    "race",
    "renal",
    "renalhx",
    "resp",
    "resp1",
    "scoma1",
    "seps",
    "sex",
    "sod1",
    "surv2md1",
    "temp1",
    "transhx",
    "trauma",
    "urin1",
    "urin1_miss",
    "wblc1",
    "wtkilo1",
]


def build_cleaned_rhc(df: pd.DataFrame) -> pd.DataFrame:
    a_raw = df["swang1"]
    if a_raw.dtype == "O":
        a = (a_raw == "RHC").astype(int)
    else:
        a = (a_raw.astype(float) > 0).astype(int)

    y = df["t3d30"]

    exclude_cols = ["swang1", "t3d30", "pafi1", "paco21", "ph1", "hema1", "Unnamed: 0"]
    x = df.drop(columns=exclude_cols).copy()

    if "cat1" in x.columns:
        x["cat1_CHF"] = (x["cat1"] == "CHF").astype(int)
        x["cat1_Cirrhosis"] = (x["cat1"] == "Cirrhosis").astype(int)
        x["cat1_Colon_Cancer"] = (x["cat1"] == "Colon Cancer").astype(int)
        x["cat1_Coma"] = (x["cat1"] == "Coma").astype(int)
        x["cat1_COPD"] = (x["cat1"] == "COPD").astype(int)
        x["cat1_Lung_Cancer"] = (x["cat1"] == "Lung Cancer").astype(int)
        x["cat1_MOSF_Malignancy"] = (x["cat1"] == "MOSF w/Malignancy").astype(int)
        x["cat1_MOSF_Sepsis"] = (x["cat1"] == "MOSF w/Sepsis").astype(int)
        x = x.drop(columns=["cat1"])

    if "cat2" in x.columns:
        x["cat2_Cirrhosis"] = (x["cat2"] == "Cirrhosis").astype(int)
        x["cat2_Colon_Cancer"] = (x["cat2"] == "Colon Cancer").astype(int)
        x["cat2_Coma"] = (x["cat2"] == "Coma").astype(int)
        x["cat2_Lung_Cancer"] = (x["cat2"] == "Lung Cancer").astype(int)
        x["cat2_MOSF_Malignancy"] = (x["cat2"] == "MOSF w/Malignancy").astype(int)
        x["cat2_MOSF_Sepsis"] = (x["cat2"] == "MOSF w/Sepsis").astype(int)
        x = x.drop(columns=["cat2"])

    if "income" in x.columns:
        x["income1"] = (x["income"] == "$11-$25k").astype(int)
        x["income2"] = (x["income"] == "$25-$50k").astype(int)
        x["income3"] = (x["income"] == "> $50k").astype(int)
        x = x.drop(columns=["income"])

    if "ca" in x.columns:
        x["ca_Yes"] = (x["ca"] == "Yes").astype(int)
        x["ca_Metastatic"] = (x["ca"] == "Metastatic").astype(int)
        x = x.drop(columns=["ca"])

    if "ninsclas" in x.columns:
        x["ninsclas_Medicaid"] = (x["ninsclas"] == "Medicaid").astype(int)
        x["ninsclas_Medicare"] = (x["ninsclas"] == "Medicare").astype(int)
        x["ninsclas_Medicare_and_Medicaid"] = (x["ninsclas"] == "Medicare & Medicaid").astype(int)
        x["ninsclas_No_insurance"] = (x["ninsclas"] == "No Insurance").astype(int)
        x["ninsclas_Private_and_Medicare"] = (x["ninsclas"] == "Private & Medicare").astype(int)
        x = x.drop(columns=["ninsclas"])

    if "sex" in x.columns and x["sex"].dtype == "O":
        x["sex"] = (x["sex"] == "Female").astype(int)

    if "race" in x.columns and x["race"].dtype == "O":
        x["race_black"] = (x["race"] == "black").astype(int)
        x["race_other"] = (x["race"] == "other").astype(int)
        x = x.drop(columns=["race"])

    z = df[["pafi1", "paco21"]].copy()
    w = df[["ph1", "hema1"]].copy()

    analysis_df = pd.concat(
        [
            y.rename("Y"),
            a.rename("A"),
            x,
            z,
            w,
        ],
        axis=1,
    )

    cols_to_drop = ["adld3p", "urin1", "dthdte", "dschdte", "sadmdte", "death", "dth30", "ptid"]
    analysis_df = analysis_df.drop(columns=[col for col in cols_to_drop if col in analysis_df.columns])

    for col in analysis_df.columns:
        if analysis_df[col].dtype == object:
            unique_vals = analysis_df[col].dropna().unique()
            if len(unique_vals) > 0 and set(unique_vals).issubset({"Yes", "No"}):
                analysis_df[col] = (analysis_df[col] == "Yes").astype(int)

    diagnosis_rename = {
        "resp": "resp_Yes",
        "card": "card_Yes",
        "neuro": "neuro_Yes",
        "gastr": "gastr_Yes",
        "renal": "renal_Yes",
        "meta": "meta_Yes",
        "hema": "hema_Yes",
        "seps": "seps_Yes",
        "trauma": "trauma_Yes",
        "ortho": "ortho_Yes",
        "sex": "sex_Female",
    }
    analysis_df = analysis_df.rename(columns={k: v for k, v in diagnosis_rename.items() if k in analysis_df.columns})
    return analysis_df


def _as_binary(series: pd.Series, *, positive: str = "Yes") -> pd.Series:
    if series.dtype == object:
        return (series == positive).astype(int)
    return (pd.to_numeric(series, errors="coerce").fillna(0) > 0).astype(int)


def _as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _fill_with_median(series: pd.Series) -> pd.Series:
    median = series.dropna().median()
    if pd.isna(median):
        median = 0.0
    return series.fillna(float(median))


def build_rhc_mcf_matched(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    a_raw = df["swang1"]
    if a_raw.dtype == object:
        a = (a_raw == "RHC").astype(int)
    else:
        a = (pd.to_numeric(a_raw, errors="coerce").fillna(0) > 0).astype(int)

    analysis = pd.DataFrame(
        {
            "Y": (df["dth30"] != "Yes").astype(float),
            "A": a.astype(int),
        }
    )

    analysis["adld3pc"] = _fill_with_median(_as_numeric(df["adld3p"]))
    analysis["age"] = _fill_with_median(_as_numeric(df["age"]))
    analysis["alb1"] = _fill_with_median(_as_numeric(df["alb1"]))
    analysis["amihx"] = _as_binary(df["amihx"])
    analysis["aps1"] = _fill_with_median(_as_numeric(df["aps1"]))
    analysis["bili1"] = _fill_with_median(_as_numeric(df["bili1"]))
    analysis["ca"] = df["ca"].map({"No": 0, "Yes": 1, "Metastatic": 2}).fillna(0).astype(int)
    analysis["card"] = _as_binary(df["card"])
    analysis["cardiohx"] = _as_binary(df["cardiohx"])
    analysis["cat1"] = (
        df["cat1"]
        .map(
            {
                "ARF": 0,
                "CHF": 1,
                "COPD": 2,
                "Cirrhosis": 3,
                "Colon Cancer": 4,
                "Coma": 5,
                "Lung Cancer": 6,
                "MOSF w/Malignancy": 7,
                "MOSF w/Sepsis": 8,
            }
        )
        .fillna(0)
        .astype(int)
    )
    analysis["cat2"] = (
        df["cat2"]
        .map(
            {
                "MOSF w/Malignancy": 1,
                "MOSF w/Sepsis": 2,
            }
        )
        .fillna(0)
        .astype(int)
    )
    analysis["cat2_miss"] = df["cat2"].isna().astype(int)
    analysis["chfhx"] = _as_binary(df["chfhx"])
    analysis["chrpulhx"] = _as_binary(df["chrpulhx"])
    analysis["crea1"] = _fill_with_median(_as_numeric(df["crea1"]))
    analysis["das2d3pc"] = _fill_with_median(_as_numeric(df["das2d3pc"]))
    analysis["dementhx"] = _as_binary(df["dementhx"])
    analysis["dnr1"] = _as_binary(df["dnr1"])
    analysis["edu"] = _fill_with_median(_as_numeric(df["edu"]))
    analysis["gastr"] = _as_binary(df["gastr"])
    analysis["gibledhx"] = _as_binary(df["gibledhx"])
    analysis["hema"] = _as_binary(df["hema"])
    analysis["hema1"] = _fill_with_median(_as_numeric(df["hema1"]))
    analysis["hrt1"] = _fill_with_median(_as_numeric(df["hrt1"]))
    analysis["immunhx"] = _as_binary(df["immunhx"])
    analysis["income"] = (
        df["income"]
        .map(
            {
                "Under $11k": 0,
                "$11-$25k": 1,
                "$25-$50k": 2,
                "> $50k": 3,
            }
        )
        .fillna(0)
        .astype(int)
    )
    analysis["liverhx"] = _as_binary(df["liverhx"])
    analysis["malighx"] = _as_binary(df["malighx"])
    analysis["meanbp1"] = _fill_with_median(_as_numeric(df["meanbp1"]))
    analysis["meta"] = _as_binary(df["meta"])
    analysis["neuro"] = _as_binary(df["neuro"])
    analysis["ninsclas"] = (
        df["ninsclas"]
        .map(
            {
                "Medicaid": 0,
                "Medicare": 1,
                "Medicare & Medicaid": 2,
                "No Insurance": 3,
                "Private": 4,
                "Private & Medicare": 5,
            }
        )
        .fillna(0)
        .astype(int)
    )
    analysis["ortho"] = _as_binary(df["ortho"])
    analysis["paco21"] = _fill_with_median(_as_numeric(df["paco21"]))
    analysis["pafi1"] = _fill_with_median(_as_numeric(df["pafi1"]))
    analysis["ph1"] = _fill_with_median(_as_numeric(df["ph1"]))
    analysis["pot1"] = _fill_with_median(_as_numeric(df["pot1"]))
    analysis["psychhx"] = _as_binary(df["psychhx"])
    analysis["race"] = (
        df["race"]
        .map(
            {
                "black": 0,
                "asian": 1,
                "other": 1,
                "white": 2,
            }
        )
        .fillna(1)
        .astype(int)
    )
    analysis["renal"] = _as_binary(df["renal"])
    analysis["renalhx"] = _as_binary(df["renalhx"])
    analysis["resp"] = _as_binary(df["resp"])
    analysis["resp1"] = _fill_with_median(_as_numeric(df["resp1"]))
    analysis["scoma1"] = _fill_with_median(_as_numeric(df["scoma1"]))
    analysis["seps"] = _as_binary(df["seps"])
    analysis["sex"] = df["sex"].map({"Female": 0, "Male": 1}).fillna(0).astype(int)
    analysis["sod1"] = _fill_with_median(_as_numeric(df["sod1"]))
    analysis["surv2md1"] = _fill_with_median(_as_numeric(df["surv2md1"]))
    analysis["temp1"] = _fill_with_median(_as_numeric(df["temp1"]))
    analysis["transhx"] = _as_binary(df["transhx"])
    analysis["trauma"] = _as_binary(df["trauma"])
    analysis["urin1"] = _fill_with_median(_as_numeric(df["urin1"]))
    analysis["urin1_miss"] = _as_numeric(df["urin1"]).isna().astype(int)
    analysis["wblc1"] = _fill_with_median(_as_numeric(df["wblc1"]))
    analysis["wtkilo1"] = _fill_with_median(_as_numeric(df["wtkilo1"]))

    z_cols = [col for col in ["pafi1", "paco21"] if col in analysis.columns]
    w_cols = [col for col in ["ph1", "hema1"] if col in analysis.columns]
    x_cols = [col for col in MCF_ALL_CONFOUNDERS if col not in {*z_cols, *w_cols}]
    metadata = {
        "n": int(len(analysis)),
        "treatment_rate": float(analysis["A"].mean()),
        "mean_outcome": float(analysis["Y"].mean()),
        "treated_survival_rate": float(analysis.loc[analysis["A"] == 1, "Y"].mean()),
        "control_survival_rate": float(analysis.loc[analysis["A"] == 0, "Y"].mean()),
        "x_cols": x_cols,
        "w_cols": w_cols,
        "z_cols": z_cols,
        "all_confounders": MCF_ALL_CONFOUNDERS,
        "policy_features": MCF_POLICY_FEATURES,
        "paper_matched_outcome": "survival at 6 months coded from dth30",
        "notes": [
            "Matches the MCF paper's 55-confounder design as closely as possible from raw_rhc.csv.",
            "adld3pc is proxied by raw adld3p because adld3pc is not present in raw_rhc.csv.",
            "cat2_miss and urin1_miss are derived missingness indicators.",
            "pafi1,paco21 are assigned to Z and ph1,hema1 are assigned to W so that B2 still sees the full 55-confounder set.",
        ],
    }
    return analysis.loc[:, ["Y", "A", *x_cols, *z_cols, *w_cols]], metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess raw RHC CSV into the cleaned semi-synthetic analysis table.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    cleaned = build_cleaned_rhc(df)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.output_csv, index=False)
    print(f"Saved {args.output_csv}")
    print(f"Shape: {cleaned.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
