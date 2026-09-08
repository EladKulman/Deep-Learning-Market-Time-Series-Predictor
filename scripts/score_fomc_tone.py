#!/usr/bin/env python3
"""Score FOMC statements for hawkish / dovish tone with a purpose-trained classifier.

Default model: LorenzoAleCon29/roberta-large-fomc-hawkish-dovish, a sentence-level
three-way classifier. The gated gtfintechlab/FOMC-RoBERTa is an optional alternative.
It replaces the six-word keyword counts used previously.

Per statement we compute
  fomc_hawkish_share   share of sentences classified hawkish
  fomc_dovish_share    share of sentences classified dovish
  fomc_net_hawkish     hawkish share minus dovish share (-1..1)
  fomc_tone_change     net hawkishness minus the previous statement's
and write a calendar-day file where the statement-day rows carry the scores and
fomc_net_hawkish_ewma is an exponentially weighted level (half-life 3 statements)
that is forward-filled between meetings.

Statements are released at 14:00 ET, before the close, so the day-T value is known at
the day-T close.

Input : data/raw/fomc/fomc_statements_raw.jsonl (from fetch_fomc_statements.py)
Output: data/processed/fomc_tone_daily.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd
import torch
from chronos_data import fingerprint
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# The reference model, gtfintechlab/FOMC-RoBERTa (CC BY-NC 4.0), is a gated repository:
# accept its terms on huggingface.co and run `huggingface-cli login`, then pass
# --model-id gtfintechlab/FOMC-RoBERTa. The default below is an ungated RoBERTa-large
# fine-tuned on the same hawkish/dovish FOMC sentence dataset with explicit labels.
DEFAULT_MODEL_ID = "LorenzoAleCon29/roberta-large-fomc-hawkish-dovish"
DEFAULT_REVISION = "f4759d4ad3f1182f81d87e47ba603261740d36cf"
PROBES = {
    "hawkish": "In light of elevated inflation, the Committee decided to raise the target range for the federal funds rate.",
    "dovish": "The Committee decided to lower the target range for the federal funds rate to support the recovery.",
    "neutral": "The Committee will continue to monitor the implications of incoming information for the economic outlook.",
}
SENTENCE_SPLIT = re.compile(r"(?<=[.;!?])\s+(?=[A-Z])")
BOILERPLATE = ("voting for the", "voting against", "for media inquiries", "implementation note",
               "for release at", "share", "last update:")


def split_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in SENTENCE_SPLIT.split(text) if len(s.strip()) > 25]
    return [s for s in sentences if not any(marker in s.lower() for marker in BOILERPLATE)]


def load_model(model_id: str, device: str, revision: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision).to(device).eval()
    labels = {int(k): v.lower() for k, v in model.config.id2label.items()}
    if not {"hawkish", "dovish"} <= set(labels.values()):
        raise SystemExit(f"{model_id} does not expose hawkish/dovish labels: {labels}")
    logging.info("Loaded %s on %s; labels %s", model_id, device, labels)
    return tokenizer, model, labels


@torch.no_grad()
def classify(sentences: list[str], tokenizer, model, labels: dict[int, str], device: str, batch_size: int = 16) -> list[str]:
    out = []
    for i in range(0, len(sentences), batch_size):
        batch = tokenizer(sentences[i:i + batch_size], padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        preds = model(**batch).logits.argmax(dim=-1).tolist()
        out.extend(labels[p] for p in preds)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Score FOMC statements with FOMC-RoBERTa")
    parser.add_argument("--input", type=Path, default=Path("data/raw/fomc/fomc_statements_raw.jsonl"))
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--start", default="2006-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", help="Model commit SHA; the default classifier uses the tested pinned revision")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines()]
    records = sorted((r for r in records if args.start <= r["date"] <= end), key=lambda r: r["date"])
    if not records:
        raise ValueError("No statements in the requested range")
    revision = args.model_revision or (DEFAULT_REVISION if args.model_id == DEFAULT_MODEL_ID else None)
    tokenizer, model, labels = load_model(args.model_id, args.device, revision)

    probe_results = dict(zip(PROBES, classify(list(PROBES.values()), tokenizer, model, labels, args.device)))
    logging.info("Probe sentences -> %s", probe_results)
    if probe_results["hawkish"] != "hawkish" or probe_results["dovish"] != "dovish":
        raise SystemExit("Model failed the hawkish/dovish probe sentences; check the label map.")

    rows = []
    sentence_rows = []
    for record in records:
        sentences = split_sentences(record["text"])
        if not sentences:
            raise ValueError(f"No sentences for {record['date']}; refusing incomplete scores")
        predicted = classify(sentences, tokenizer, model, labels, args.device)
        counts = pd.Series(predicted).value_counts()
        n = len(predicted)
        hawkish = counts.get("hawkish", 0) / n
        dovish = counts.get("dovish", 0) / n
        rows.append({"date": record["date"], "fomc_n_sentences": n, "fomc_hawkish_share": hawkish,
                     "fomc_dovish_share": dovish, "fomc_net_hawkish": hawkish - dovish})
        sentence_rows.extend({"date": record["date"], "label": lab, "sentence": sen} for lab, sen in zip(predicted, sentences))
        logging.info("%s: %3d sentences, hawkish %.2f dovish %.2f", record["date"], n, hawkish, dovish)

    scores = pd.DataFrame(rows)
    scores["fomc_tone_change"] = scores["fomc_net_hawkish"].diff()
    scores["fomc_net_hawkish_ewma"] = scores["fomc_net_hawkish"].ewm(halflife=3).mean()

    raw_dir = Path(args.out_dir) / "raw" / "fomc"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sentence_rows).to_csv(raw_dir / "fomc_sentence_labels.csv", index=False)
    scores.to_csv(raw_dir / "fomc_statement_scores.csv", index=False)

    daily = pd.DataFrame({"date": pd.date_range(args.start, end, freq="D").strftime("%Y-%m-%d")})
    daily = daily.merge(scores, on="date", how="left")
    daily["fomc_net_hawkish_ewma"] = daily["fomc_net_hawkish_ewma"].ffill()
    daily["fomc_net_hawkish_last"] = daily["fomc_net_hawkish"].ffill()

    path = Path(args.out_dir) / "processed" / "fomc_tone_daily.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(path, index=False)
    metadata = {"model_id": args.model_id, "model_revision": revision or getattr(model.config, "_commit_hash", None),
                "input_sha256": fingerprint(args.input), "scores_sha256": fingerprint(raw_dir / "fomc_statement_scores.csv"),
                "daily_sha256": fingerprint(path), "statement_count": len(scores), "start": args.start, "end": end,
                "labels": labels, "probes": probe_results, "device": args.device}
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    logging.info("Wrote %s (%d statements scored, %d days)", path, len(scores), len(daily))


if __name__ == "__main__":
    main()
