import pandas as pd
import jiwer
from wav2vec2decoder import Wav2Vec2Decoder
import torchaudio

datasets = {
    "LibriSpeech": "data/librispeech_test_other/manifest.csv",
    "Earnings22": "data/earnings22_test/manifest.csv",
}

lm_models = {
    "LibriSpeech 3-gram": "lm/3-gram.pruned.1e-7.arpa.gz",
    "Financial 3-gram": "lm/financial-3gram.arpa.gz",
}

methods = ["beam_lm", "beam_lm_rescore"]

# Best alpha/beta from Task 4
alpha, beta = 0.05, 0.5

for lm_name, lm_path in lm_models.items():
    decoder = Wav2Vec2Decoder(lm_model_path=lm_path, beam_width=5, alpha=alpha, beta=beta)
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
            print(f"{lm_name:20s} | {ds_name:12s} | {method:20s} | WER={wer:.2%} CER={cer:.2%}")
