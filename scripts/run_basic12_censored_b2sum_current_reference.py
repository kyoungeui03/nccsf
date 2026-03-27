#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_PACKAGE_ROOT = PROJECT_ROOT / "python-package"
if str(PYTHON_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGE_ROOT))
from grf.benchmarks.econml_8variant import CASE_SPECS, _evaluate_predictions, prepare_case
from grf.censored import B2SummaryCensoredSurvivalForest

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--tag", type=str, default="full")
    p.add_argument("--case-ids", nargs="*", type=int)
    return p.parse_args()

def summarize(df):
    s=(df.groupby("name",as_index=False).agg(avg_rmse=("rmse","mean"),avg_mae=("mae","mean"),avg_pearson=("pearson","mean"),avg_bias=("bias","mean"),avg_sign_acc=("sign_acc","mean"),avg_time=("total_time","mean")).sort_values(["avg_rmse","avg_mae","avg_pearson"], ascending=[True,True,False]).reset_index(drop=True))
    s.insert(0,"rank", range(1, len(s)+1))
    return s

def metric_row(name, preds, case, case_spec, elapsed, backend):
    row = _evaluate_predictions(name, preds, case.true_cate, elapsed, backend=backend)
    row.update(case_id=int(case_spec["case_id"]), case_slug=str(case_spec["slug"]), case_title=str(case_spec["title"]), target="RMST", estimand_horizon=float(case.horizon), horizon_quantile=None, n=int(case.cfg.n), p_x=int(case.cfg.p_x), p_w=int(case.cfg.p_w), p_z=int(case.cfg.p_z), seed=int(case.cfg.seed), target_censor_rate=float(case.cfg.target_censor_rate), actual_censor_rate=float(1.0 - case.delta.mean()), linear_treatment=bool(case.cfg.linear_treatment), linear_outcome=bool(case.cfg.linear_outcome), tau_log_hr=float(case.cfg.tau_log_hr))
    return row

def main():
    args=parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected=set(args.case_ids) if args.case_ids else None
    case_specs=[c for c in CASE_SPECS if selected is None or int(c["case_id"]) in selected]
    frames=[]
    for case_spec in case_specs:
        case=prepare_case(case_spec, target="RMST", horizon_quantile=0.60)
        start=time.time()
        model=B2SummaryCensoredSurvivalForest(target='RMST', horizon=None, surv_scalar_mode='pair', censoring_estimator='nelson-aalen')
        model.fit_components(case.X, case.A, case.Y, case.delta, case.Z, case.W)
        preds=model.effect_from_components(case.X, case.W, case.Z).ravel()
        row=metric_row('B2Sum (SurvPair & NA)', preds, case, case_spec, time.time()-start, model.__class__.__name__)
        df=pd.DataFrame([row]).sort_values(["case_id","name"])
        frames.append(df)
        pd.concat(frames, ignore_index=True).to_csv(args.output_dir / f"results_{args.tag}.csv", index=False)
    res=pd.concat(frames, ignore_index=True).sort_values(["case_id","name"]).reset_index(drop=True)
    res.to_csv(args.output_dir / f"results_{args.tag}.csv", index=False)
    summarize(res).to_csv(args.output_dir / f"summary_{args.tag}.csv", index=False)
if __name__ == "__main__":
    raise SystemExit(main())
