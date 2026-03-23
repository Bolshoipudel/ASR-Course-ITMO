import pandas as pd
import jiwer
from wav2vec2decoder import Wav2Vec2Decoder
import torchaudio

df = pd.read_csv("data/librispeech_test_other/manifest.csv")
beam_widths = [1, 3, 5, 10]

decoder = Wav2Vec2Decoder(lm_model_path=None, beam_width=1)

for bw in beam_widths:
    decoder.beam_width = bw
    hypotheses, references = [], []
    for _, row in df.iterrows():
        audio, sr = torchaudio.load(row["path"])
        hyp = decoder.decode(audio, method="beam")
        hypotheses.append(hyp)
        references.append(row["text"])
    wer = jiwer.wer(references, hypotheses)
    cer = jiwer.cer(references, hypotheses)
    print(f"beam_width={bw:3d} | WER={wer:.2%} CER={cer:.2%}")
