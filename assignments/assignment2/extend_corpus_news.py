from datasets import load_dataset

print("Downloading financial news articles...")
ds = load_dataset("ashraq/financial-news-articles", split="train")

new_lines = []
for sample in ds:
    text = sample.get("text", "").strip().lower()
    for line in text.split(". "):
        line = line.strip().rstrip(".")
        if 20 < len(line) < 500:
            new_lines.append(line)

print(f"Got {len(new_lines)} lines from financial news")

with open("data/earnings22_train/corpus.txt", "a") as f:
    for line in new_lines:
        f.write(line + "\n")

print(f"Appended {len(new_lines)} lines to corpus.txt")
