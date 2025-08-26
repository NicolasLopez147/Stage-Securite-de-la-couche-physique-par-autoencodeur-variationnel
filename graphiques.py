import matplotlib.pyplot as plt

snr = [0, 5, 10, 20]
ber_b = [0.292480, 0.156494, 0.053955, 0.021729]
ber_e = [0.399902, 0.286133, 0.307129, 0.379150]

plt.figure()
plt.plot(snr, ber_b, marker='o', label='Bob')
plt.plot(snr, ber_e, marker='o', label='Ève')
plt.title("Taux d'erreur binaire (BER) en fonction du SNR")
plt.xlabel("SNR (dB)")
plt.ylabel("Taux d'erreur binaire (BER)")
plt.legend(title="Courbes")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('ber_vs_snr_fr.png', dpi=200)
print("Graphique enregistré dans ber_vs_snr_fr.png")
