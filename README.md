# Mini-Project

This workspace now contains a reproducible classical machine learning pipeline and an IEEE-style paper draft for the UCI Heart Disease mini-project.

## Structure

- `src/run_experiments.py`: end-to-end experiment runner
- `results/`: generated summaries, tables, and figures
- `paper/paper.tex`: IEEE conference paper draft
- `paper/references.bib`: bibliography for the paper
- `requirements.txt`: Python dependencies used in this workspace
- `heart+disease/`: provided dataset files

## How To Run

Run the experiments from the project root:

```powershell
python src\run_experiments.py
```

This generates:

- `results/tables/baseline_holdout_results.csv`
- `results/tables/baseline_cv_results.csv`
- `results/tables/improved_model_cv_results.csv`
- `results/tables/improved_holdout_results.csv`
- `results/tables/ablation_results.csv`
- `results/tables/top_features.csv`
- `results/figures/class_balance.png`
- `results/figures/missingness.png`
- `results/figures/correlation_heatmap.png`
- `results/figures/roc_curves.png`
- `results/overall_summary.json`

## Notes

- The script uses `heart+disease/processed.cleveland.data` because the provided `WARNING` file states that `cleveland.data` is corrupted.
- The paper draft uses the IEEE conference format sections requested in the assignment: abstract, introduction, related work, method, results, discussion, and conclusion.
- A LaTeX compiler was not available in the current environment, so `paper/paper.tex` was generated but not compiled here.
