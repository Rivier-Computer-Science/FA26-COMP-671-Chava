# SMS Spam Classification with Character-Level Robustness

**Student:** Tejaswi Chava  
**Course:** COMP-671 Deep Learning

This project compares a word/character TF-IDF logistic-regression baseline with a small character-level TextCNN on the UCI SMS Spam Collection. It asks whether character-level training augmentation improves recall under a fixed synthetic text-corruption stress test.

## Research questions

1. Does TextCNN outperform a strong sparse-feature baseline on held-out SMS messages?
2. Does mild character augmentation improve spam recall on deterministically perturbed test text?
3. What failure modes and limitations are visible in the measured results?

## Data and protocol

The dataset has 5,574 labeled messages and is downloaded from the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection). Each record contains a `ham` or `spam` label and its message text. Raw data are not committed. Before splitting, exact texts are grouped by case-folding and trimming whitespace; the split is stratified by the group label using seed 42, so repeated normalized texts stay in one partition. In the recorded run the partitions were train 3,918 (3,391 ham / 527 spam), validation 828 (717 / 111), and test 828 (719 / 109). The actual counts are also written to `reports/run_metadata.json`. Thresholds are selected only on validation data to maximize spam recall under a ham false-positive constraint of 5%. The held-out test set is reserved for final reporting. For the synthetic stress test, each held-out message is independently and deterministically transformed: two character edits are attempted independently, each with probability 0.8—replace one alphabetic character with a digit and insert an internal space before an alphabetic character. This is a narrow character-edit test, not a complete model of real-world obfuscation.

## Methods

- **Baseline:** word unigram/bigram and character 2–5-gram TF-IDF features with class-balanced logistic regression.
- **Neural model:** character embeddings, parallel width-3 and width-5 convolutions, masked global max pooling, dropout, and a binary output.
- **Ablation:** identical TextCNN training with or without mild character augmentation.
- **Metrics:** average precision (PR-AUC), spam recall, ham false-positive rate, precision, macro-F1, and confusion counts.
- **Seeds:** 17, 42, and 73 for neural runs.
- **Key settings:** 256-character input, embedding size 16, 24 filters at widths 3 and 5, dropout 0.30, Adam learning rate 0.001, batch size 256, up to 10 epochs, early-stopping patience 2, and augmentation probability 0.35. All settings are recorded in `configs/default.json`.
- **Training augmentation:** the same two edits are attempted independently with probability 0.35. The augmented condition applies this only to training examples; validation and test messages are unchanged for fitting.

## Setup and run

The recorded run used CPython 3.9.6 on macOS arm64.

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e .
python -m sms_spam.cli run --output-dir outputs
```

Windows PowerShell:

```powershell
py -3.9 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e .
python -m sms_spam.cli run --output-dir outputs
```

`requirements-lock.txt` records the exact package versions used for the reported run (CPython 3.9.6 on macOS arm64). `requirements.txt` lists supported dependency ranges for other environments. The single `run` command downloads and preprocesses the data, trains the TF-IDF baseline and both TextCNN conditions, selects thresholds on validation, and evaluates clean and perturbed test data. It writes `results.csv`, `summary.csv`, `run_metadata.json`, `performance.png`, local model checkpoints, and a local error-analysis CSV under `outputs/`. No manual preprocessing is needed. The checkpoint files are intentionally not committed; the command above retrains and saves the models.

Predict one message after training:

```bash
python -m sms_spam.cli predict --checkpoint outputs/textcnn_augmented_seed42.pt --text "Congratulations, claim your prize now"
```

## Project files

```text
configs/default.json       fixed split, architecture, and training settings
requirements-lock.txt      exact package versions for the reported run
requirements.txt           supported dependency ranges
src/sms_spam/data.py       UCI download, parsing, grouped split, augmentation
src/sms_spam/models.py     character encoding and TextCNN
src/sms_spam/experiment.py baseline, training, evaluation, outputs
src/sms_spam/cli.py        experiment and single-message inference commands
reports/                   aggregate metrics, figure, results discussion, report
output/pdf/                final PDF report
```

The run writes the configuration, split sizes, metrics, plot, model checkpoints, and local error-analysis records. Do not publish unredacted message text from error files. Raw data, checkpoints, and generated outputs are excluded from version control except for aggregate report artifacts. To rebuild the PDF report from the tracked results and figure, run `python reports/build_report.py` after installing the locked dependencies.

## Results at a glance

The rerun found that TF-IDF plus logistic regression had the strongest ranking and recall on the held-out test set. It achieved 0.991 spam recall with a 0.047 ham false-positive rate on clean messages and 0.991 recall with a 0.050 false-positive rate on perturbed messages. The TextCNN variants had lower average precision and higher perturbed false-positive rates. The controlled augmentation comparison did not meet its predeclared five-percentage-point perturbed-recall improvement goal. See `reports/initial_experiment_results.md` for details and limitations.

## Responsible use

This historical, English-language benchmark does not establish real-world performance. False positives can hide legitimate communication. Predictions should be reviewed by a person; this prototype should not automatically block or delete messages. Avoid exposing raw message text or identifiers in reports.

## References

- Almeida, T., Hidalgo, J., & Yamakami, A. (2011). Contributions to the study of SMS spam filtering: New collection and results. *Proceedings of the 11th ACM Symposium on Document Engineering*. https://doi.org/10.1145/2034691.2034742
- Kim, Y. (2014). Convolutional neural networks for sentence classification. *EMNLP 2014*, 1746–1751. https://doi.org/10.3115/v1/D14-1181
- Prince, S. J. D. (2023). *Understanding Deep Learning*. MIT Press.
