"""Train baselines and TextCNNs, select thresholds on validation, evaluate test."""

from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import FeatureUnion
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import augment_message, download_dataset, load_messages, make_grouped_split, obfuscate_message
from .models import TextCNN, encode_message


class MessageDataset(Dataset):
    def __init__(self, frame, augment=False, probability=0.0, seed=0):
        self.texts = frame.text.astype(str).tolist()
        self.labels = frame.label.map({"ham": 0, "spam": 1}).astype(int).tolist()
        self.augment = augment
        self.probability = probability
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        text = self.texts[index]
        if self.augment:
            text = augment_message(text, self.rng, self.probability)
        return text, self.labels[index]


def _loader(frame, cfg, shuffle=False, augment=False, seed=0):
    ds = MessageDataset(frame, augment, cfg["augmentation_probability"], seed)
    def collate(batch):
        texts, labels = zip(*batch)
        x = torch.tensor([encode_message(t, cfg["max_char_length"]) for t in texts], dtype=torch.long)
        y = torch.tensor(labels, dtype=torch.float32)
        return x, y
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle,
                      generator=generator, num_workers=0, collate_fn=collate)


def _probabilities(model, loader):
    model.eval()
    values = []
    with torch.no_grad():
        for x, _ in loader:
            values.append(torch.sigmoid(model(x)).cpu().numpy())
    return np.concatenate(values)


def select_threshold(y, probabilities, fpr_limit):
    best = None
    for threshold in np.unique(np.r_[0.0, probabilities, 1.000001]):
        pred = probabilities >= threshold
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        fpr = fp / max(1, fp + tn)
        recall = tp / max(1, tp + fn)
        precision = tp / max(1, tp + fp)
        if fpr <= fpr_limit + 1e-12:
            candidate = (recall, precision, float(threshold))
            if best is None or candidate > best:
                best = candidate
    if best is None:
        raise RuntimeError("No validation threshold met the ham false-positive constraint")
    return best[2]


def metrics(y, probability, threshold):
    prediction = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "pr_auc": average_precision_score(y, probability),
        "spam_precision": precision_score(y, prediction, zero_division=0),
        "spam_recall": recall_score(y, prediction, zero_division=0),
        "macro_f1": f1_score(y, prediction, average="macro", zero_division=0),
        "ham_false_positive_rate": fp / max(1, fp + tn),
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
    }


def _labels(frame):
    return frame.label.map({"ham": 0, "spam": 1}).to_numpy()


def _feature_model():
    return FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b",
                                  ngram_range=(1, 2), max_features=30000,
                                  sublinear_tf=True, lowercase=True)),
        ("char", TfidfVectorizer(analyzer="char", ngram_range=(2, 5),
                                  max_features=40000, sublinear_tf=True, lowercase=True)),
    ])


def _evaluate_model_row(name, seed, val_y, val_prob, clean_y, clean_prob,
                        perturbed_prob, cfg):
    threshold = select_threshold(val_y, val_prob, cfg["ham_false_positive_target"])
    row = {"condition": name, "seed": seed, "threshold": threshold}
    row.update({f"clean_{key}": val for key, val in metrics(clean_y, clean_prob, threshold).items()})
    row.update({f"perturbed_{key}": val for key, val in metrics(clean_y, perturbed_prob, threshold).items()})
    return row, threshold


def run_experiments(project_root: Path, output_dir: Path, config_path: Path | None = None):
    cfg = json.loads((config_path or project_root / "configs/default.json").read_text())
    archive = download_dataset(project_root / "data/raw/sms-spam-collection.zip", cfg["dataset_url"])
    data = load_messages(archive)
    train, val, test = make_grouped_split(data, cfg["split_seed"], cfg["train_fraction"], cfg["validation_fraction"], cfg["test_fraction"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"config": cfg, "total_rows": len(data),
                "unique_normalized_texts": int(data.group.nunique()),
                "partitions": {n: {"rows": len(f), "label_counts": f.label.value_counts().to_dict()}
                               for n, f in (("train", train), ("validation", val), ("test", test))}}
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    y_train, y_val, y_test = map(_labels, (train, val, test))
    perturbed_text = [obfuscate_message(text, cfg["split_seed"] + 1000 + i) for i, text in enumerate(test.text)]
    rows = []

    features = _feature_model()
    x_train = features.fit_transform(train.text)
    x_val = features.transform(val.text)
    x_test = features.transform(test.text)
    x_pert = features.transform(perturbed_text)
    baseline = LogisticRegression(class_weight="balanced", max_iter=2000,
                                  solver="liblinear", random_state=cfg["split_seed"])
    baseline.fit(x_train, y_train)
    val_prob = baseline.predict_proba(x_val)[:, 1]
    clean_prob = baseline.predict_proba(x_test)[:, 1]
    pert_prob = baseline.predict_proba(x_pert)[:, 1]
    row, threshold = _evaluate_model_row("tfidf_baseline", "", y_val, val_prob,
                                         y_test, clean_prob, pert_prob, cfg)
    rows.append(row)
    joblib.dump({"features": features, "classifier": baseline, "threshold": threshold}, output_dir / "tfidf_baseline.joblib")

    for augmented in (False, True):
        condition = "textcnn_augmented" if augmented else "textcnn_plain"
        for seed in cfg["seeds"]:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(1)
            train_loader = _loader(train, cfg, shuffle=True, augment=augmented, seed=seed)
            val_loader = _loader(val, cfg, seed=seed)
            test_loader = _loader(test, cfg, seed=seed)
            pert_frame = test.copy(); pert_frame["text"] = perturbed_text
            pert_loader = _loader(pert_frame, cfg, seed=seed)
            model = TextCNN(cfg["embedding_dim"], cfg["filters_per_kernel"],
                            tuple(cfg["kernel_sizes"]), cfg["dropout"])
            counts = np.bincount(y_train, minlength=2)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(counts[0] / counts[1], dtype=torch.float32))
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
            best_ap, best_state, patience = -1.0, None, 0
            for _epoch in range(cfg["max_epochs"]):
                model.train()
                for x, y in train_loader:
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_fn(model(x), y)
                    loss.backward(); optimizer.step()
                val_prob = _probabilities(model, val_loader)
                ap = average_precision_score(y_val, val_prob)
                if ap > best_ap + 1e-6:
                    best_ap = float(ap)
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                    if patience >= cfg["early_stopping_patience"]:
                        break
            model.load_state_dict(best_state)
            val_prob = _probabilities(model, val_loader)
            clean_prob = _probabilities(model, test_loader)
            pert_prob = _probabilities(model, pert_loader)
            row, threshold = _evaluate_model_row(condition, seed, y_val, val_prob,
                                                  y_test, clean_prob, pert_prob, cfg)
            row["best_validation_pr_auc"] = best_ap
            rows.append(row)
            torch.save({"state_dict": best_state,
                        "model_config": {"embedding_dim": cfg["embedding_dim"], "filters_per_kernel": cfg["filters_per_kernel"], "kernel_sizes": tuple(cfg["kernel_sizes"]), "dropout": cfg["dropout"]},
                        "threshold": threshold, "max_char_length": cfg["max_char_length"]},
                       output_dir / f"{condition}_seed{seed}.pt")
            print(f"Completed {condition} seed {seed}: validation AP {best_ap:.4f}")

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "results.csv", index=False)
    numeric = [c for c in results if c.startswith(("clean_", "perturbed_")) and results[c].dtype != object]
    summary = results.groupby("condition")[numeric].agg(["mean", "std"])
    summary.to_csv(output_dir / "summary.csv")
    _plot_results(results, output_dir / "performance.png")
    _write_error_analysis(test, results, output_dir, features, baseline, cfg, train, val)
    return output_dir / "results.csv"


def _plot_results(results, path):
    conditions = ["tfidf_baseline", "textcnn_plain", "textcnn_augmented"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(3); width = 0.34
    for offset, metric, label, color in [(-width/2, "clean_spam_recall", "Clean test", "#21618C"),
                                         (width/2, "perturbed_spam_recall", "Perturbed test", "#16818C")]:
        means = [results.loc[results.condition == c, metric].mean() for c in conditions]
        errs = [results.loc[results.condition == c, metric].std(ddof=1) if (results.condition == c).sum() > 1 else 0 for c in conditions]
        ax.bar(x + offset, means, width, label=label, color=color, yerr=errs, capsize=3)
    ax.set_ylabel("Spam recall at validation-selected threshold")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x, ["TF-IDF +\nlogistic regression", "TextCNN\n(no augmentation)", "TextCNN\n(augmentation)"])
    ax.set_title("Held-out spam recall: clean and perturbed messages")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=0.2); fig.tight_layout()
    fig.savefig(path, dpi=180); plt.close(fig)


def _write_error_analysis(test, results, output_dir, features, baseline, cfg, train, val):
    # This local file contains message text; it is intentionally ignored by Git.
    y = _labels(test)
    p = baseline.predict_proba(features.transform(test.text))[:, 1]
    threshold = float(results.loc[results.condition == "tfidf_baseline", "threshold"].iloc[0])
    pred = p >= threshold
    rows = [{"index": int(i), "label": test.iloc[i].label,
             "predicted": "spam" if pred[i] else "ham", "spam_probability": float(p[i]),
             "text": test.iloc[i].text}
            for i in np.flatnonzero(pred != y)]
    pd.DataFrame(rows).to_csv(output_dir / "error_analysis.csv", index=False)
