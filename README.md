
# GROUP37_CENG313 – Image Matching Challenge 2025

**Group ID:** 37

## Team Members

| Name | Student ID | GitHub |
|---|---|---|
| Kayra Orbay | 23118080017 | KayraOrbay |
| Ayşegül Pınar Yılmaz | 23118080004 | aysegul08 |

## Competition

[Image Matching Challenge 2025](https://www.kaggle.com/competitions/image-matching-challenge-2025)

## How to Run

1. Download the data from the [Kaggle Data page](https://www.kaggle.com/competitions/image-matching-challenge-2025/data). Create a `data/` folder in the repository root and put `train_labels.csv`, `train_thresholds.csv`, `sample_submission.csv` and the extracted `train/` folder in it (`data/train/<dataset>/<image>`). The data is not committed because of Kaggle rules and file size.
2. Open `assignment1_eda_baseline.ipynb` in Colab: [Open in Colab](https://colab.research.google.com/github/ceng313-2026/GROUP37_CENG313/blob/main/assignment1_eda_baseline.ipynb)
3. Run all cells. The first cell clones the repository and installs `requirements.txt`. If the CSV files are missing, the notebook asks you to upload them. The images must be extracted under `data/train/` before running the baseline cells.

## Scores

**Local held-out evaluation.** Following the course rules, we held out 20% of the training images (389 of 1945) with seed 42, stratified by dataset and scene. The score is calculated with our local implementation of the competition metric, so it can differ from the official Kaggle score.

| Model | Pose mAA | Clustering precision | Score |
|---|---|---|---|
| SIFT pairwise baseline (mean over 13 datasets) | 0.0089 | 0.5775 | **0.0169** |

The per-dataset scores are in the notebook.

**Kaggle:** We submitted the baseline as a late submission on 13.11.2026 ([submission notebook](https://www.kaggle.com/code/kayraorbay/notebookf452731e5f?scriptVersionId=369238802)). The score was still pending at the time of writing. For comparison, the winning team reported 52.64 on the public leaderboard ([1st place write-up](https://www.kaggle.com/competitions/image-matching-challenge-2025/writeups/ns64-1st-place-solution)).
