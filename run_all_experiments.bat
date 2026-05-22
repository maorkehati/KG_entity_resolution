@echo off
python scripts\99_run_all_experiments.py ^
  --variant pairwise_50_medium_unseen100 ^
  --run-name neural_logreg ^
  --hard-negative-experiment symbolic_hn_strong ^
  --soft-risk ^
  --hard-negative-training ^
  --structured-models ^
  --final-figures ^
  --report-ablations
