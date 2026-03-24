import pandas as pd
from collections import Counter
import re
import os

STOPWORDS = {
    "with", "and", "or", "the", "a", "an", "of", "to", "in", "for",
    "on", "at", "by", "from", "is", "was", "are", "be", "has", "had",
    "it", "its", "this", "that", "no", "not", "as", "but", "due",
    "after", "since", "per", "over", "into", "up", "if",
}

def tokenize(text):
    if pd.isna(text):
        return []
    text = text.lower()
    tokens = re.findall(r"[a-z]+", text)
    return [t for t in tokens if t not in STOPWORDS]

def main():
    df = pd.read_csv("../data/clean_train.csv")
    output_dir = os.path.dirname(os.path.abspath(__file__))

    for acuity in sorted(df["triage_acuity"].unique()):
        subset = df[df["triage_acuity"] == acuity]["chief_complaint_raw"]
        all_tokens = []
        for complaint in subset:
            all_tokens.extend(tokenize(complaint))

        counts = Counter(all_tokens).most_common()
        out_path = os.path.join(output_dir, f"acuity_{acuity}_top_words.txt")
        with open(out_path, "w") as f:
            f.write(f"Acuity Level {acuity} — All Chief Complaint Word Counts\n")
            f.write(f"Total complaints: {len(subset)}\n")
            f.write("=" * 45 + "\n")
            f.write(f"{'Rank':<6}{'Word':<25}{'Count':<10}{'% of complaints'}\n")
            f.write("-" * 55 + "\n")
            for rank, (word, count) in enumerate(counts, 1):
                pct = count / len(subset) * 100
                f.write(f"{rank:<6}{word:<25}{count:<10}{pct:.1f}%\n")

        print(f"Written: {out_path}")

if __name__ == "__main__":
    main()
