import pandas as pd
import jiwer
from wav2vec2decoder import Wav2Vec2Decoder
import torchaudio

df = pd.read_csv("data/librispeech_test_other/manifest.csv")
alphas = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
betas = [0.0, 0.5, 1.0, 1.5]

decoder = Wav2Vec2Decoder(lm_model_path="lm/3-gram.pruned.1e-7.arpa.gz", beam_width=5)

results = []
for alpha in alphas:
    for beta in betas:
        decoder.alpha = alpha
        decoder.beta = beta
        hypotheses, references = [], []
        for _, row in df.iterrows():
            audio, sr = torchaudio.load(row["path"])
            hyp = decoder.decode(audio, method="beam_lm")
            hypotheses.append(hyp)
            references.append(row["text"])
        wer = jiwer.wer(references, hypotheses)
        cer = jiwer.cer(references, hypotheses)
        print(f"alpha={alpha:.2f} beta={beta:.1f} WER={wer:.2%} CER={cer:.2%}")
        results.append({"alpha": alpha, "beta": beta, "wer": wer, "cer": cer})

results_df = pd.DataFrame(results)
results_df.to_csv("results_lm_sweep.csv", index=False)
print("Finished!")
