# Anonymization Audit Report

## Summary
This audit focuses on identity leaks that could reveal authorship or institutional affiliation. The highest-risk issues are repeated absolute local filesystem paths containing a username (e.g., `/Users/kyoungeuihong/...`) across README, docs, data metadata, and output artifacts. No emails, ORCID, or social handles were detected in the sampled scans. Git metadata/history and remotes cannot be verified via repository contents.

> **Note:** Code-search results may be incomplete due to search result limits. Consider running the manual commands below to confirm no additional leaks are present.

## High-risk leaks
1) **Local filesystem paths with username**
- **File:** `README.md` (lines 18–45, 51–87, 93–145)  
  **Matched text:** `/Users/kyoungeuihong/Desktop/csf_grf_new/...`  
  **Why risky:** Leaks local username and machine path.  
  **Suggested replacement:** Use repository-relative paths (e.g., `docs/R_BASELINE_SETUP.md`, `censored/`) or `<REPO_ROOT>` placeholders.

- **File:** `docs/R_BASELINE_SETUP.md` (lines 33–49)  
  **Matched text:** `/Users/kyoungeuihong/Desktop/csf_grf_new/...`  
  **Why risky:** Leaks local username and machine path.  
  **Suggested replacement:** Use `cd <REPO_ROOT>` and relative paths such as `python3 scripts/...`.

- **File:** `docs/FINAL_MODEL_KEEP_DELETE_MANIFEST.md` (line 5)  
  **Matched text:** link to `/Users/kyoungeuihong/Desktop/csf_grf_new/...`  
  **Why risky:** Leaks local username and machine path.  
  **Suggested replacement:** Use repository-relative links (e.g., `docs/FINAL_REPRODUCIBILITY_NOTE.md`).

- **File:** `docs/FINAL_MODEL_ARCHITECTURE.md` (lines 60–88)  
  **Matched text:** `/Users/kyoungeuihong/Desktop/csf_grf_new/...`  
  **Why risky:** Leaks local username and machine path.  
  **Suggested replacement:** Replace with repo-relative paths (e.g., `censored/run_5model_benchmark.py`).

- **File:** `data/synthetic_scenarios/catalog.csv` (line 1+)  
  **Matched text:** `/Users/kyoungeuihong/Desktop/csf_grf_new/data/...`  
  **Why risky:** Leaks local username and machine path inside data metadata.  
  **Suggested replacement:** Store relative paths (e.g., `data/synthetic_scenarios/...`).

- **File:** `data/synthetic_scenarios/catalog.json` (lines ~8–40+)  
  **Matched text:** `/Users/kyoungeuihong/Desktop/csf_grf_new/data/...`  
  **Why risky:** Same as above.  
  **Suggested replacement:** Use relative paths.

- **File:** `data/synthetic_scenarios/survival_default/metadata.json` (lines 31–34)  
  **Matched text:** `/Users/kyoungeuihong/Desktop/csf_grf_new/data/...`  
  **Why risky:** Same as above.  
  **Suggested replacement:** Use relative paths.

- **File:** `outputs/final_paper_results/censored_synthetic_4model/README.txt` (line 10)  
  **Matched text:** `/Users/kyoungeuihong/Desktop/csf_grf_new/outputs/...`  
  **Why risky:** Leaks local username and machine path in output artifacts.  
  **Suggested replacement:** Use `outputs/...` relative paths.

## Medium-risk leaks
1) **External repository URL**
- **File:** `THIRD_PARTY_NOTICE.md` (line 3)  
  **Matched text:** `https://github.com/grf-labs/grf`  
  **Why risky:** Exposes upstream project identity; not necessarily author identity, but can hint at provenance.  
  **Suggested replacement:** Replace with `[ANONYMIZED_REPO_URL]` if required by anonymization policy, or keep but note in anonymization checklist.

2) **Paper attribution**
- **File:** `Archive/README.md` (line 4)  
  **Matched text:** “Cui et al. (2023)”  
  **Why risky:** Academic reference to a specific paper is usually acceptable but can hint at lineage; treat as medium risk depending on anonymity constraints.  
  **Suggested replacement:** “prior work (2023)” if strict anonymity required.

## Low-risk cleanup
- **File:** `README.md` (title `# csf_grf_new`)  
  **Why risky:** Project name may correlate with prior public repos; low risk but consider neutralizing.  
  **Suggested replacement:** “Anonymous Submission Codebase”.

## Recommended changes
- Replace **all absolute local paths** (`/Users/kyoungeuihong/...`) with **relative paths** or `<REPO_ROOT>` placeholders.
- Replace external GitHub URLs in documentation with `[ANONYMIZED_REPO_URL]` where required.
- Replace any project-specific titles in README with neutral names.
- Confirm whether third-party notice text must remain intact for licensing; do not remove license terms—only redact URLs if allowed.

## Files safe to keep
- `pyproject.toml` (no author/URL/email fields)
- `requirements.txt` (dependency list only)
- Most code files (not inspected for comments; consider targeted scan)

## Exact commands to clean the repo
```bash
# Find absolute paths (macOS/Linux)
rg -n "/Users/|/home/|C:\\Users\\" .

# Find emails
rg -n "[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}" -i .

# Find GitHub URLs
rg -n "github\.com/" .

# Find possible institution names
rg -n "\b(University|Institute|Laboratory|Lab|Department|School)\b" -i .

# Find social/identity handles
rg -n "\b(ORCID|Google Scholar|LinkedIn|Twitter|X\.com)\b" -i .

# List git remotes (run locally in repo)
git remote -v

# Show commit authors/emails (run locally in repo)
git log --format='%an <%ae>' | sort -u
```
