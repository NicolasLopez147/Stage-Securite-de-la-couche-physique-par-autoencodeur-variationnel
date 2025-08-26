import math
import torch
import torch.nn as nn
import os
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import trange
import pandas as pd

# ─── hiperparámetros globales ─────────────────────────────────────────────────
N_A, N_B = 2, 4
INPUT_DIM = 2 * N_A
# SNR_BdB = 20
ber_threshold = 0.03
max_epochs = 40000
batch_sz = 1024
lr = 5e-4
device = 'cpu'

os.makedirs("checkpoints", exist_ok=True)

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
    """
    sym_*: tensores con valores 0,1,2,3 (el índice 'phase').
    Retorna BER usando mapeo Gray→natural de bits (00,01,11,10).
    """
    # Pasamos de Gray a binario natural
    dec = gray2bin[sym_decoded.view(-1)]
    tru = gray2bin[sym_true.view(-1)]
    # Extraemos bits
    b0_dec, b1_dec = dec // 2, dec % 2
    b0_tru, b1_tru = tru // 2, tru % 2
    # Contamos errores
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

# ─── entrenamiento solo para Bob ───────────────────────────────────────────────
def train_only_bob(SNR_BdB):
    enc = Encoder().to(device)
    dec = Decoder(2*N_B + 2*N_A*N_B + 1).to(device)

    opt_enc = torch.optim.AdamW(enc.parameters(), lr=lr, weight_decay=1e-6)
    opt_dec = torch.optim.AdamW(dec.parameters(), lr=lr, weight_decay=1e-6)
    scheduler_enc = OneCycleLR(opt_enc, max_lr=lr, total_steps=max_epochs, pct_start=0.05, anneal_strategy='cos')
    scheduler_dec = OneCycleLR(opt_dec, max_lr=lr, total_steps=max_epochs, pct_start=0.05, anneal_strategy='cos')
    mse = nn.MSELoss()

    for epoch in range(1, max_epochs+1):
        phase = torch.randint(0, 4, (batch_sz, N_A), device=device)
        m = torch.stack([torch.cos(math.pi/2*phase), torch.sin(math.pi/2*phase)], -1).reshape(batch_sz, -1)
        m = m / math.sqrt(2)

        H_B = generate_channel_rayleigh(batch_sz, N_B, N_A, device)
        x = enc(m, H_B)
        y_B = mimo_rayleigh(x, SNR_BdB, H_B)

        snr_pilot_lin = 10**(30/10)
        sigma_p = math.sqrt(1/(2*snr_pilot_lin))
        H_hat_B = H_B + sigma_p * torch.randn_like(H_B)

        y_B_flat = y_B.reshape(batch_sz, -1)
        H_B_hat_flat = H_hat_B.reshape(batch_sz, -1)
        pnr_B = torch.full((batch_sz,1), 10**(SNR_BdB/10), device=device)
        decoder_in = torch.cat([y_B_flat, H_B_hat_flat, pnr_B], dim=1)
        x_hat_B = dec(decoder_in)

        L = mse(x_hat_B, m)

        opt_enc.zero_grad()
        opt_dec.zero_grad()
        L.backward()
        opt_enc.step()
        scheduler_enc.step()
        opt_dec.step()
        scheduler_dec.step()

        with torch.no_grad():
            ph_B = torch.atan2(x_hat_B[:,1::2], x_hat_B[:,0::2])
            sym_B = ((ph_B % (2*math.pi)) / (math.pi/2)).round() % 4
            sym_B = sym_B.to(torch.long)
            ber_B = compute_ber(sym_B,phase)

        print(f"Epoch {epoch:4d} - BER_B: {ber_B:.4f} , snr_B: {SNR_BdB}dB")

        # if ber_B < ber_threshold:
        #     filename = f"checkpoints/base_encoder_snr{str(SNR_BdB).replace('.', 'p')}dB.pth"

        #     torch.save({
        #         'enc_state_dict': enc.state_dict(),
        #         'dec_state_dict': dec.state_dict(),
        #         'opt_enc_state_dict': opt_enc.state_dict(),
        #         'opt_dec_state_dict': opt_dec.state_dict(),
        #         'epoch': epoch,
        #     }, filename)
        #     print("Modelo base guardado con BER_B aceptable.")
        #     break
    filename = f"checkpoints/base_encoder_snr{str(SNR_BdB).replace('.', 'p')}dB.pth"

    torch.save({
        'enc_state_dict': enc.state_dict(),
        'dec_state_dict': dec.state_dict(),
        'opt_enc_state_dict': opt_enc.state_dict(),
        'opt_dec_state_dict': opt_dec.state_dict(),
        'epoch': epoch,
    }, filename)
    print("Modelo base guardado con BER_B aceptable.")
    return ber_B, epoch 

if __name__ == '__main__':
    results = []
    snr_list = [10,20]
    for snr in snr_list:
        final_ber, epochs_used = train_only_bob(snr)
        results.append({
        'SNR': snr,
        'epochs_used': epochs_used,
        'final_ber': final_ber
    })
    print(f"SNR {snr} - Epochs: {epochs_used}, BER_B: {final_ber:.4f}")

    # convertir a DataFrame y guardar
    df = pd.DataFrame(results)
    print(df)