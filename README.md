# Stage – Sécurité de la couche physique par auto‑encodeur variationnel (VAE)

> **Python : 3.13.3**

Ce dépôt regroupe le code, les figures et les scripts d’évaluation produits pendant le stage consacré à la **sécurité de la couche physique (PHY)** sur canal **MIMO Rayleigh** avec **CSI imparfaite**. Nous concevons et évaluons deux versions d’un transcepteur appris par **auto‑encodeur variationnel (VAE)** :
- **Version 1** : base **SVAE/iCSI‑SVAE** avec normalisation de puissance et décodeur recevant \((y_B,\widehat H_{AB},p)\). Entraînement au **MSE pur**.
- **Version 2** : **apprentissage en deux étapes** + **critère adversaire**. L’encodeur est **conditionné par \(H_{AB}\)** et un poids \(\lambda_{\mathrm{adv}}\) est augmenté progressivement pour dégrader Ève **sans** sacrifier Bob. Un **ré‑entraînement d’Ève** (encodeur figé) est aussi fourni pour justifier le plancher d’erreur.

---

## 📁 Arborescence du dépôt (extrait)

```
.
├── Autres versions/
├── Version 1/
│   └── v2.1.py                       # Script principal de la Version 1
├── Version 2/
│   └── src/
│       ├── train_bob.py              # Étape 1 : pré‑entraînement côté Bob
│       ├── test_model.py             # Évaluation Bob/Eve (figures, BER vs SNR)
│       ├── train_eve.py              # Ré‑entraînement d’Ève (encodeur figé)
│       └── test_eve.py               # Traces BER_E vs époques pour différents SNR_E
├── graphiques.py                     # Utilitaires/figures
├── README.md
└── (images) RSB_=8_dB.png, RSB_=32_dB_(proche_de_40_dB).png, Tous_les_RSB.png,
              ber_vs_snr_fr1.png, ber_vs_snr_fr2.png, ber_vs_snr_fr3.png
```

> Les dossiers `checkpoints/` et `results/` sont créés automatiquement par les scripts de la **Version 2** (sous `Version 2/src/`) si besoin.

---

## ⚙️ Prérequis (bibliothèques uniquement)

- `torch`
- `numpy`
- `matplotlib`
- `tqdm`

> La version de Python utilisée: 3.13.3 


## ▶️ Exécution – Version 1 (baseline SVAE / iCSI‑SVAE)

1. Se placer dans le répertoire de la version 1 :
   ```bash
   cd "Version 1"
   ```
2. Lancer le script principal :
   ```bash
   python v2.1.py
   ```
3. **Hyperparamètres** : les constantes (p.ex. `SNR_BdB`, `SNR_EdB`, tailles MIMO, etc.) se règlent en **tête de fichier**.  
   - **PNR des pilotes = 30 dB pour toutes les expériences** (cohérent avec le rapport).  
   - Config typique : \(N_A=2\), \(N_B=4\), canal MIMO Rayleigh, bruit AWGN.
4. **Sorties attendues** : courbes **BER vs SNR** de Bob/Ève (p.ex. `ber_vs_snr_fr.png`) et métriques dans la console.

---

## ▶️ Exécution – Version 2 (deux étapes + adversaire)

> **Ordre d’exécution des programmes** (depuis `Version 2/src/`) :

```bash
cd "Version 2/src"

# 1) Étape 1 – Entraînement côté Bob uniquement
python train_bob.py

# 2) Ré‑entraînement d’Ève avec l’encodeur gelé (justification de confidentialité)
python train_eve.py

# 3) Évaluation du modèle obtenu (BER_B / BER_E, génération de figures)
python test_model.py

# 4) Traces BER_E vs époques et comparaison multi‑SNR_E
python test_eve.py
```

- Les scripts créent/chargent automatiquement les **checkpoints** dans `Version 2/src/checkpoints/` et déposent les figures dans `Version 2/src/results/`.  
- Les figures typiques générées : `ber_vs_snr_fr1.png`, `ber_vs_snr_fr2.png`, `ber_vs_snr_fr3.png`, `RSB_=8_dB.png`, `RSB_=32_dB_(proche_de_40_dB).png`, `Tous_les_RSB.png`.

---

## 📊 Résultats (résumé rapide)

**Version 1 – baseline (MSE pur, \(N_A=2, N_B=4\), PNR pilotes = 30 dB)**  
BER en fin d’entraînement (40k itérations, modèles indépendants par SNR) :

| SNR (dB) | 0     | 5     | 10    | 20    |
|----------|-------|-------|-------|-------|
| **BER\_B** | 0,376 | 0,378 | 0,040 | 0,009 |
| **BER\_E** | 0,484 | 0,542 | 0,550 | 0,595 |

> **Observation :** genou clair ≈ 10 dB pour Bob ; Ève reste élevée et quasi‑plate.

**Version 2 – deux étapes + critère adversaire**  

- **Étape 1 (Bob seul)** : \(\mathrm{BER}_B=\{0,2893;\;0,1599;\;0,0371;\;0,0205\}\) pour SNR \(\{0,5,10,20\}\) dB.  
- **Étape 2 (adverse, Bob+Ève)** :  
  \(\mathrm{BER}_B=\{0,2925;\;0,1565;\;0,0540;\;0,0217\}\),  
  \(\mathrm{BER}_E=\{0,3999;\;0,2861;\;0,3071;\;0,3792\}\).  
- **Ré‑entraînement d’Ève (encodeur figé)** : **plancher** \(\mathrm{BER}_E\approx 0,40\) **indépendant du SNR\_E** (8–32 dB).

> **Lecture :** l’encodeur **dépend de \(H_{AB}\)** ; l’information manquante chez Ève crée un **plafond informationnel** qui ne disparaît pas en augmentant le SNR\_E.

---

## 📝 À propos du rapport

Le **rapport du PRE** (non confidentiel) décrit en détail :
- les schémas classiques (**bruit artificiel** et **brouillage coopératif**) en PHY sécurité ;
- la conception **VAE** (normalisation de puissance, **iCSI‑SVAE**), les **pertes** utilisées et le choix **PNR pilotes = 30 dB** partout ;
- la **Version 1** (baseline) et la **Version 2** (deux étapes + adversaire), ainsi que le protocole de **ré‑entraînement d’Ève** avec encodeur gelé ;
- les **courbes BER** (Bob/Ève vs SNR) et l’analyse du **plancher d’Ève**.


## 🔁 Reproductibilité – points d’attention

- **PNR des pilotes = 30 dB** pour **toutes** les expériences.
- Canaux **MIMO Rayleigh i.i.d.** + **AWGN**.
- Normalisation de puissance en sortie d’encodeur : \(x \leftarrow \sqrt{2N_A}\,x/\|x\|_2\).
- Entraînement : `AdamW` + `OneCycleLR` (paramètres dans les scripts).  
- Les **seeds** et tailles de lot peuvent légèrement changer les chiffres ; les tendances sont robustes (genou ~10 dB, séparation Bob/Ève, plancher d’Ève ~0,40).

---

## 📚 Références clés

- A. D. Wyner, “The Wire-Tap Channel,” *Bell System Technical Journal*, 1975.  
- I. Csiszár & J. Körner, “Broadcast Channels with Confidential Messages,” *IEEE Trans. IT*, 1978.  
- S. Goel & R. Negi, “Guaranteeing Secrecy using Artificial Noise,” *IEEE Trans. WC*, 2008.  
- L. Dong, Z. Han, A. P. Petropulu, H. V. Poor, “Cooperative Jamming for Wireless Physical Layer Security,” *IEEE SPL*, 2010.  
- C.-H. Lin, C.-C. Wu, K.-F. Chen, T.-S. Lee, “A VAE-Based Secure Transceiver Design Using Deep Learning,” 2020.

---

## 🤝 Remerciements

Merci aux encadrants et à l’équipe de l’U2IS – ENSTA pour leur accompagnement et les échanges techniques qui ont permis d’aboutir à ces résultats.
