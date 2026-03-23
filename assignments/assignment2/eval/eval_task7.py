import pandas as pd
import jiwer
from wav2vec2decoder import Wav2Vec2Decoder
import torchaudio

datasets = {
    "LibriSpeech": "data/librispeech_test_other/manifest.csv",
    "Earnings22": "data/earnings22_test/manifest.csv",
}

# Best alpha/beta from Task 4 sweep
decoder = Wav2Vec2Decoder(lm_model_path="lm/3-gram.pruned.1e-7.arpa.gz", beam_width=5, alpha=0.05, beta=0.5)

methods = ["greedy", "beam", "beam_lm", "beam_lm_rescore"]

results = []
for ds_name, manifest_path in datasets.items():
    df = pd.read_csv(manifest_path)
    for method in methods:
        hypotheses, references = [], []
        for _, row in df.iterrows():
            audio, sr = torchaudio.load(row["path"])
            hyp = decoder.decode(audio, method=method)
            hypotheses.append(hyp)
            references.append(row["text"])
        wer = jiwer.wer(references, hypotheses)
        cer = jiwer.cer(references, hypotheses)
        print(f"{ds_name} | {method:20s} | WER={wer:.2%} CER={cer:.2%}")
        results.append({"dataset": ds_name, "method": method, "wer": wer, "cer": cer})

results_df = pd.DataFrame(results)
results_df.to_csv("results_task7.csv", index=False)
print("\nResults saved to results_task7.csv")
