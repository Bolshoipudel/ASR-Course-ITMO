import torch
import torchaudio
from melbanks import LogMelFilterBanks
import soundfile as sf

data, sr = sf.read(r"C:\Users\Григорий\Downloads\Астероид.wav")
signal = torch.tensor(data, dtype=torch.float32)
if signal.dim() == 1:
    signal = signal.unsqueeze(0) 


melspec = torchaudio.transforms.MelSpectrogram(
    hop_length=160,
    n_mels=80
)(signal)

logmelbanks = LogMelFilterBanks()(signal)

print(f"Reference shape: {torch.log(melspec + 1e-6).shape}")
print(f"File shape:      {logmelbanks.shape}")

assert torch.log(melspec + 1e-6).shape == logmelbanks.shape
assert torch.allclose(torch.log(melspec + 1e-6), logmelbanks)
