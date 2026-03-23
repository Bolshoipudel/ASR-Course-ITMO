import pandas as pd
from wav2vec2decoder import Wav2Vec2Decoder
import torchaudio

df = pd.read_csv("data/librispeech_test_other/manifest.csv")

# Best alpha/beta from sweeps
decoder = Wav2Vec2Decoder(lm_model_path="lm/3-gram.pruned.1e-7.arpa.gz", beam_width=5, alpha=0.05, beta=0.5)

differences = []
for _, row in df.iterrows():
    audio, sr = torchaudio.load(row["path"])
    ref = row["text"]
    beam = decoder.decode(audio, method="beam")
    sf = decoder.decode(audio, method="beam_lm")
    rs = decoder.decode(audio, method="beam_lm_rescore")

    if beam != sf or beam != rs:
        differences.append({"ref": ref, "beam": beam, "sf": sf, "rs": rs})
        print(f"REF:  {ref}")
        print(f"BEAM: {beam}")
        print(f"SF:   {sf}")
        print(f"RS:   {rs}")
        print()

    if len(differences) >= 10:
        break

print(f"Found {len(differences)} samples with differences")
