"""Command-line experiment runner and single-message inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .experiment import run_experiments
from .models import TextCNN, encode_message


def main():
    parser = argparse.ArgumentParser(description="SMS spam classification project")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="train and evaluate all model conditions")
    run.add_argument("--output-dir", type=Path, default=Path("outputs"))
    run.add_argument("--config", type=Path, default=Path("configs/default.json"))
    predict = commands.add_parser("predict", help="classify one message with a TextCNN checkpoint")
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--text", required=True)
    args = parser.parse_args()
    if args.command == "run":
        result = run_experiments(Path.cwd(), args.output_dir, args.config)
        print(f"Results saved to {result}")
    else:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model = TextCNN(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        x = torch.tensor([encode_message(args.text, checkpoint["max_char_length"])])
        with torch.no_grad():
            probability = torch.sigmoid(model(x)).item()
        threshold = checkpoint["threshold"]
        print({"prediction": "spam" if probability >= threshold else "ham",
               "spam_probability": round(probability, 4), "threshold": round(threshold, 4)})


if __name__ == "__main__":
    main()
