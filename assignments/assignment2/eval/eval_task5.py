import pandas as pd
import jiwer
from wav2vec2decoder import Wav2Vec2Decoder
import torchaudio

df = pd.read_csv("data/librispeech_test_other/manifest.csv")

# Best alpha/beta from Task 4
best_alpha = 0.05
best_beta = 0.5

lm_models = {
    "3-gram": "lm/3-gram.pruned.1e-7.arpa.gz",
    "4-gram": "lm/4-gram.arpa.gz",
}

for lm_name, lm_path in lm_models.items():
    print(f"\nEvaluating {lm_name} LM...")
    decoder = Wav2Vec2Decoder(lm_model_path=lm_path, beam_width=5, alpha=best_alpha, beta=best_beta)
    hypotheses, references = [], []
    for _, row in df.iterrows():
        audio, sr = torchaudio.load(row["path"])
        hyp = decoder.decode(audio, method="beam_lm")
        hypotheses.append(hyp)
        references.append(row["text"])
    wer = jiwer.wer(references, hypotheses)
    cer = jiwer.cer(references, hypotheses)
    print(f"{lm_name}: WER={wer:.2%} CER={cer:.2%}")
