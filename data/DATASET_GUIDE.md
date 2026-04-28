# Dataset Guide (CAP 4630)

## Recommended Sources

- Kaggle Datasets: <https://www.kaggle.com/datasets>
- Roboflow Universe: <https://universe.roboflow.com/>

Search terms:

- `waste classification dataset`
- `recyclable waste images`
- `trashnet`

## Required Class Labels

Use these six labels for consistency in training, evaluation, and demo:

1. `plastic`
2. `paper`
3. `glass`
4. `metal`
5. `organic`
6. `other`

## Data Quality Rules

- Keep only RGB images.
- Remove duplicate files when possible.
- Remove obvious mislabeled images if discovered.
- Aim for class balance. If imbalance exists, use class weights in training.

## Folder Format

```text
data/raw/
  plastic/
  paper/
  glass/
  metal/
  organic/
  other/
```

## Fast MVP Target

- Minimum per class: 150 images
- Better target per class: 300-800 images
- Keep first run small enough to finish quickly tonight.
