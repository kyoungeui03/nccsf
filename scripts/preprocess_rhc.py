#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "data" / "rhc" / "raw_rhc.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "rhc" / "cleaned_rhc.csv"


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
