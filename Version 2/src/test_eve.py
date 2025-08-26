import csv
import math
import torch
import torch.nn as nn
import os
from torch.optim.lr_scheduler import OneCycleLR
import pandas as pd
import matplotlib.pyplot as plt


# ─── hiperparámetros globales ─────────────────────────────────────────────────
N_A, N_B, N_E = 2, 4, 4
INPUT_DIM = 2 * N_A
# SNR_EdB = 40
SNR_BdB = 20
max_epochs = 4000
batch_sz = 1024
lr = 5e-4
device = 'cpu'

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ─── utilidades de canal ──────────────────────────────────────────────────────
def mimo_rayleigh(x, snr_db, H, na=N_A):
    B = x.size(0)
    y_c = torch.einsum('bijc,bjc->bic', H, x.view(B, na, 2))
    p_signal = 1
    snr_lin = 10**(snr_db/10)
    sigma = math.sqrt(p_signal / (2 * snr_lin))
    y_c += sigma * torch.randn_like(y_c)
    return y_c.reshape(B, -1)

def generate_channel_rayleigh(batch_sz, n, na, device='cpu'):
    return torch.randn(batch_sz, n, na, 2, device=device) / math.sqrt(2)

def fc(in_f, out_f):
    return nn.Sequential(
        nn.Linear(in_f, out_f),
        nn.BatchNorm1d(out_f),
        nn.Tanh()
    )

# Mapeo de constelación QPSK Gray → índice natural de bits
gray2bin = torch.tensor([0, 1, 3, 2], device=device, dtype=torch.long)

def compute_ber(sym_decoded, sym_true):
    dec = gray2bin[sym_decoded.view(-1)]
    tru = gray2bin[sym_true.view(-1)]
    b0_dec, b1_dec = dec // 2, dec % 2
    b0_tru, b1_tru = tru // 2, tru % 2
    errors = (b0_dec != b0_tru).sum() + (b1_dec != b1_tru).sum()
    return errors.item() / (2 * dec.numel())

# ─── definición de redes ──────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        in_dim = INPUT_DIM + 2*N_B*N_A
        self.body = nn.Sequential(
            fc(in_dim, 64),
            fc(64, 32),
            fc(32, 16),
            fc(16, 8),
            fc(8, 2*N_A)
        )
        self.out = nn.Linear(2*N_A, 4*N_A)

    def forward(self, m, H):
        B = m.size(0)
        H_flat = H.view(B, -1)
        x_in = torch.cat([m, H_flat], dim=1)
        h = self.body(x_in)
        mu, logv = self.out(h).chunk(2, dim=-1)
        std = torch.exp(0.5 * logv)
        x = mu + std * torch.randn_like(std)
        x = x / x.norm(dim=1, keepdim=True) * math.sqrt(INPUT_DIM)
        return x

class Decoder(nn.Module):
    def __init__(self, first):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(first),
            fc(first, 16),
            fc(16, 32),
            fc(32, 64),
            fc(64, 128),
            fc(128, 256),
            nn.Linear(256, INPUT_DIM)
        )

    def forward(self, y):
        return self.net(y)

# ─── entrenamiento solo para Eve ───────────────────────────────────────────────

def train_only_eve(SNR_EdB, history):
    enc = Encoder().to(device)
    dec = Decoder(2*N_E + 2*N_A*N_E + 1).to(device)

    # Carga el encoder y decoder entrenados previos
    ckpt = torch.load("checkpoints/full_checkpoint.pth", map_location=device)
    enc.load_state_dict(ckpt['enc_state_dict'])
    dec.load_state_dict(ckpt['dec_state_dict'])

    # Congela el encoder - Eve no lo reentrena
    for p in enc.parameters():
        p.requires_grad = False

    opt_dec = torch.optim.AdamW(dec.parameters(), lr=lr, weight_decay=1e-6)
    scheduler_dec = OneCycleLR(opt_dec, max_lr=lr, total_steps=max_epochs, pct_start=0.05, anneal_strategy='cos')
    mse = nn.MSELoss()


    for epoch in range(1, max_epochs+1):
        phase = torch.randint(0, 4, (batch_sz, N_A), device=device, dtype=torch.long)
        m = torch.stack([torch.cos(math.pi/2*phase), torch.sin(math.pi/2*phase)], -1).reshape(batch_sz, -1)
        m = m / math.sqrt(2)

        
        # Canal de Eve
        H_E = generate_channel_rayleigh(batch_sz, N_E, N_A, device)
        H_B = generate_channel_rayleigh(batch_sz, N_B, N_A, device)
        with torch.no_grad():
            x = enc(m, H_B)
        y_E = mimo_rayleigh(x, SNR_EdB, H_E)

        # Estimación de canal
        snr_pilot_lin = 10**(30/10)
        sigma_p = math.sqrt(1/(2*snr_pilot_lin))
        H_hat_E = H_E + sigma_p * torch.randn_like(H_E)

        # Decodificación de Eve
        y_E_flat = y_E.reshape(batch_sz, -1)
        H_E_hat_flat = H_hat_E.reshape(batch_sz, -1)
        pnr_E = torch.full((batch_sz,1), 10**(SNR_EdB/10), device=device)
        decoder_in_E = torch.cat([y_E_flat, H_E_hat_flat, pnr_E], dim=1)
        x_hat_E = dec(decoder_in_E)

        # Pérdida y actualización sólo de decoder
        L = mse(x_hat_E, m)
        opt_dec.zero_grad()
        L.backward()
        opt_dec.step()
        scheduler_dec.step()

        # Métrica BER_E
        with torch.no_grad():
            ph_E = torch.atan2(x_hat_E[:,1::2], x_hat_E[:,0::2])
            sym_E = ((ph_E % (2*math.pi)) / (math.pi/2)).round() % 4
            sym_E = sym_E.to(torch.long)
            ber_E = compute_ber(sym_E, phase)
        history.append((SNR_EdB, epoch, ber_E))

        print(f"Epoch {epoch:4d} - BER_E: {ber_E:.4f}")
    
    fn = f"results/eve_ber_SNR{SNR_EdB}.csv"
    with open(fn,'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch','ber_E'])
        writer.writerows(history)

    print(f"[+] Guardado history en {fn}\n")
    return history

if __name__ == '__main__':
    snr_list = [8,16,20,32]
    results = []

    for snr in snr_list:
        print(f"--> Training Eve @ SNR={snr} dB")
        train_only_eve(snr, results)

    # vuelca todo en un CSV único
    fn = "results/eve_ber_all.csv"
    with open(fn,'w',newline='') as f:
        w = csv.writer(f)
        w.writerow(['snr','epoch','ber'])
        w.writerows(results)
    print(f"[+] Salvado historial completo en {fn}")
    
    fn = "results/eve_ber_all.csv"
    df = pd.read_csv(fn)
    df['snr'] = pd.to_numeric(df['snr'], errors='coerce')

    # ---------- Ventana 1: RSB = 8 dB (o el más cercano si no existe) ----------
    target1 = 8
    subset1 = df[df['snr'] == target1]
    label1 = f"RSB = {target1} dB"
    if subset1.empty:
        closest1 = df.loc[(df['snr'] - target1).abs().idxmin(), 'snr']
        subset1 = df[df['snr'] == closest1]
        label1 = f"RSB = {closest1:g} dB (proche de {target1} dB)"

    fig1 = plt.figure()
    try:
        fig1.canvas.manager.set_window_title(label1)
    except Exception:
        pass
    plt.plot(subset1['epoch'], subset1['ber'], label=label1)
    plt.xlabel("Époque")
    plt.ylabel("Taux d’erreur binaire (BER) d’Eve")
    plt.title(f"Taux d’erreur binaire (BER) d’Eve en fonction de l’époque – {label1}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # ---------- Ventana 2: RSB = 40 dB (o el más cercano si no existe) ----------
    target2 = 40
    subset2 = df[df['snr'] == target2]
    label2 = f"RSB = {target2} dB"
    if subset2.empty:
        closest2 = df.loc[(df['snr'] - target2).abs().idxmin(), 'snr']
        subset2 = df[df['snr'] == closest2]
        label2 = f"RSB = {closest2:g} dB (proche de {target2} dB)"

    fig2 = plt.figure()
    try:
        fig2.canvas.manager.set_window_title(label2)
    except Exception:
        pass
    plt.plot(subset2['epoch'], subset2['ber'], label=label2)
    plt.xlabel("Époque")
    plt.ylabel("Taux d’erreur binaire (BER) d’Eve")
    plt.title(f"Taux d’erreur binaire (BER) d’Eve en fonction de l’époque – {label2}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # ---------- Ventana 3: Todas las curvas ----------
    fig3 = plt.figure()
    try:
        fig3.canvas.manager.set_window_title("Tous les RSB")
    except Exception:
        pass
    for snr, grp in df.groupby('snr'):
        plt.plot(grp['epoch'], grp['ber'], label=f"RSB = {snr:g} dB")
    plt.xlabel("Époque")
    plt.ylabel("Taux d’erreur binaire (BER) d’Eve")
    plt.title("Taux d’erreur binaire (BER) d’Eve en fonction de l’époque pour différents RSB")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # mostrar todas las ventanas
    plt.show()