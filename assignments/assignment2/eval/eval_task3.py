import pandas as pd
import jiwer
from wav2vec2decoder import Wav2Vec2Decoder
import torchaudio

df = pd.read_csv("data/librispeech_test_other/manifest.csv")
temperatures = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]

decoder = Wav2Vec2Decoder(lm_model_path=None, beam_width=5)

for t in temperatures:
    decoder.temperature = t
    hypotheses, references = [], []
    for _, row in df.iterrows():
        audio, sr = torchaudio.load(row["path"])
        hyp = decoder.decode(audio, method="greedy")
        hypotheses.append(hyp)
        references.append(row["text"])
    wer = jiwer.wer(references, hypotheses)
    cer = jiwer.cer(references, hypotheses)
    print(f"Temperature={t:.1f} WER={wer:.2%} CER={cer:.2%}")
