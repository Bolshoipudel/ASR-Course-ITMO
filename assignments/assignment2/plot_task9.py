import matplotlib.pyplot as plt
import numpy as np

labels = ["LibriSpeech\n(in-domain)", "Earnings22\n(out-of-domain)"]

libri_lm_wer = [10.98, 55.63]
libri_lm_cer = [3.73, 25.50]
fin_lm_wer = [11.02, 53.83]
fin_lm_cer = [3.78, 25.24]

x = np.arange(len(labels))
width = 0.2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# WER chart
bars1 = ax1.bar(x - width/2, libri_lm_wer, width, label="LibriSpeech 3-gram", color="#4C72B0")
bars2 = ax1.bar(x + width/2, fin_lm_wer, width, label="Financial 3-gram", color="#DD8452")
ax1.set_ylabel("WER (%)")
ax1.set_title("WER by Domain and LM (Shallow Fusion)")
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend()
ax1.bar_label(bars1, fmt="%.1f%%", padding=3)
ax1.bar_label(bars2, fmt="%.1f%%", padding=3)

# CER chart
bars3 = ax2.bar(x - width/2, libri_lm_cer, width, label="LibriSpeech 3-gram", color="#4C72B0")
bars4 = ax2.bar(x + width/2, fin_lm_cer, width, label="Financial 3-gram", color="#DD8452")
ax2.set_ylabel("CER (%)")
ax2.set_title("CER by Domain and LM (Shallow Fusion)")
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.legend()
ax2.bar_label(bars3, fmt="%.1f%%", padding=3)
ax2.bar_label(bars4, fmt="%.1f%%", padding=3)

plt.tight_layout()
plt.savefig("task9_lm_comparison.png", dpi=150)
