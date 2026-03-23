import matplotlib.pyplot as plt

temperatures = [0.5, 1.0, 1.5, 2.0]
greedy_wer = [54.97, 54.97, 54.97, 54.97]
beam_lm_wer = [55.15, 55.63, 56.33, 57.38]

plt.figure(figsize=(8, 5))
plt.plot(temperatures, greedy_wer, "o-", label="Greedy", color="#4C72B0")
plt.plot(temperatures, beam_lm_wer, "s-", label="Beam + LM (Shallow Fusion)", color="#DD8452")
plt.xlabel("Temperature")
plt.ylabel("WER (%)")
plt.title("WER vs Temperature on Earnings22 (out-of-domain)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(temperatures)
plt.tight_layout()
plt.savefig("task7b_temperature.png", dpi=150)