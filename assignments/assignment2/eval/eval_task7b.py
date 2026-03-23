import pandas as pd
import jiwer
from wav2vec2decoder import Wav2Vec2Decoder
import torchaudio

df = pd.read_csv("data/earnings22_test/manifest.csv")
temperatures = [0.5, 1.0, 1.5, 2.0]

# Best alpha/beta from Task 4
decoder = Wav2Vec2Decoder(lm_model_path="lm/3-gram.pruned.1e-7.arpa.gz", beam_width=5, alpha=0.05, beta=0.5)

for t in temperatures:
    decoder.temperature = t
    for method in ["greedy", "beam_lm"]:
        hypotheses, references = [], []
        for _, row in df.iterrows():
            audio, sr = torchaudio.load(row["path"])
            hyp = decoder.decode(audio, method=method)
            hypotheses.append(hyp)
            references.append(row["text"])
        wer = jiwer.wer(references, hypotheses)
        cer = jiwer.cer(references, hypotheses)
        print(f"T={t:.1f} | {method:20s} | WER={wer:.2%} CER={cer:.2%}")
