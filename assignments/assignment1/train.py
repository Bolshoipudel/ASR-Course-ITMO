import torch
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
import os
import torch.nn as nn
from melbanks import LogMelFilterBanks
import time
import matplotlib.pyplot as plt
from ptflops import get_model_complexity_info
import soundfile as sf

import torchaudio._torchcodec
torchaudio._torchcodec.TORCHCODEC_AVAILABLE = True

def _patched_load(filepath, *args, **kwargs):
    data, sr = sf.read(filepath)
    waveform = torch.tensor(data, dtype=torch.float32)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.T  
    return waveform, sr

torchaudio.load = _patched_load


class YesNoDataset(SPEECHCOMMANDS):
    def __init__(self, subset):
        super().__init__(root="./data", download=True, subset=subset)
        self._walker = [
            w for w in self._walker
            if os.path.basename(os.path.dirname(w)) in ("yes", "no")
        ]

train_set = YesNoDataset("training")
val_set = YesNoDataset("validation")
test_set = YesNoDataset("testing")

print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

def collate_fn(batch):
    waveforms = []
    labels = []
    for waveform, sr, label, *_ in batch:
        waveforms.append(waveform.squeeze(0))
        labels.append(1 if label == "yes" else 0)

    waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
    labels = torch.tensor(labels)
    return waveforms, labels

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=False, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, collate_fn=collate_fn)

class SpeechCNN(nn.Module):
    def __init__(self, n_mels=80, groups=1):
        super().__init__()
        self.melbanks = LogMelFilterBanks(n_mels=n_mels)

        self.conv1 = nn.Conv1d(n_mels, 32, kernel_size=5, padding=2, groups=groups)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2, groups=groups)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=5, padding=2, groups=groups)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()

        self.fc = nn.Linear(64, 1) 
        
    def forward(self, x):
        x = self.melbanks(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = x.mean(dim=-1)
        x = self.fc(x)
        return x.squeeze(-1)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.BCEWithLogitsLoss()

def run_experiment(n_mels=80, groups=1, num_epochs=20):
    model = SpeechCNN(n_mels=n_mels, groups=groups).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_params = sum(p.numel() for p in model.parameters())
    train_losses = []
    epoch_times = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        start_time = time.time()
        for waveforms, labels in train_loader:
            waveforms, labels = waveforms.to(device), labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(waveforms)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        epoch_time = time.time() - start_time
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        epoch_times.append(epoch_time)

        # Val
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for waveforms, labels in val_loader:
                waveforms, labels = waveforms.to(device), labels.float().to(device)
                outputs = model(waveforms)
                predicted = (outputs > 0).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        print(f"  Epoch {epoch+1}/{num_epochs} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {epoch_time:.2f}s")

    # Test
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for waveforms, labels in test_loader:
            waveforms, labels = waveforms.to(device), labels.float().to(device)
            outputs = model(waveforms)
            predicted = (outputs > 0).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    test_acc = correct / total

    return {
        "n_mels": n_mels, "groups": groups,
        "num_params": num_params, "test_acc": test_acc,
        "train_losses": train_losses, "epoch_times": epoch_times
    }

print("Experiments: n_mels")
results_mels = []
for n_mels in [20, 40, 80]:
    print(f"\n--- n_mels={n_mels} ---")
    r = run_experiment(n_mels=n_mels)
    results_mels.append(r)
    print(f"n_mels={n_mels} | Params: {r['num_params']} | Test Acc: {r['test_acc']:.4f}")

print("\nExperiments: groups")
results_groups = []
for groups in [2, 4, 8, 16]:
    print(f"\ngroups={groups} ---")
    r = run_experiment(groups=groups)
    results_groups.append(r)
    print(f"groups={groups} | Params: {r['num_params']} | Test Acc: {r['test_acc']:.4f}")

print("\nFLOPs")
flops_list = []
for groups in [1, 2, 4, 8, 16]:
    m = SpeechCNN(groups=groups)
    flops, params = get_model_complexity_info(m, (16000,), as_strings=False, print_per_layer_stat=False)
    flops_list.append((groups, flops, params))
    print(f"groups={groups} | FLOPs: {flops:,} | Params: {params:,}")


plt.figure()
for r in results_mels:
    plt.plot(r["train_losses"], label=f"n_mels={r['n_mels']}")
plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.legend()
plt.title("Train Loss vs n_mels")
plt.savefig("loss_vs_nmels.png")
plt.close()

plt.figure()
plt.bar([str(r["n_mels"]) for r in results_mels], [r["test_acc"] for r in results_mels])
plt.xlabel("n_mels")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy vs n_mels")
plt.savefig("acc_vs_nmels.png")
plt.close()

plt.figure()
avg_times = [sum(r["epoch_times"][1:]) / len(r["epoch_times"][1:]) for r in results_groups]
plt.plot([r["groups"] for r in results_groups], avg_times, marker='o')
plt.xlabel("Groups")
plt.ylabel("Avg Epoch Time (s)")
plt.title("Epoch Training Time vs Groups")
plt.savefig("time_vs_groups.png")
plt.close()

plt.figure()
plt.plot([r["groups"] for r in results_groups], [r["num_params"] for r in results_groups], marker='o')
plt.xlabel("Groups")
plt.ylabel("Number of Parameters")
plt.title("Model Parameters vs Groups")
plt.savefig("params_vs_groups.png")
plt.close()

plt.figure()
plt.plot([g for g, f, p in flops_list], [f for g, f, p in flops_list], marker='o')
plt.xlabel("Groups")
plt.ylabel("FLOPs")
plt.title("FLOPs vs Groups")
plt.savefig("flops_vs_groups.png")
plt.close()

data, sr = sf.read(r"C:\Users\Григорий\Downloads\Астероид.wav")
signal = torch.tensor(data, dtype=torch.float32)
if signal.dim() == 1:
    signal = signal.unsqueeze(0)

melspec_ref = torchaudio.transforms.MelSpectrogram(hop_length=160, n_mels=80)(signal)
log_melspec_ref = torch.log(melspec_ref + 1e-6)
log_melspec_ours = LogMelFilterBanks()(signal)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
axes[0].imshow(log_melspec_ref.squeeze(0).numpy(), aspect='auto', origin='lower')
axes[0].set_title("torchaudio MelSpectrogram (log)")
axes[0].set_ylabel("Mel bin")
axes[1].imshow(log_melspec_ours.squeeze(0).numpy(), aspect='auto', origin='lower')
axes[1].set_title("Our LogMelFilterBanks")
axes[1].set_ylabel("Mel bin")
axes[1].set_xlabel("Frame")
plt.tight_layout()
plt.savefig("melbanks_comparison.png")
plt.close()
