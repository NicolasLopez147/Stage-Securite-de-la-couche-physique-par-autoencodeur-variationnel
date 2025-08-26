import math
import torch
import torch.nn as nn
import pandas as pd
from tqdm import trange
from torch.optim.lr_scheduler import OneCycleLR

# ─── hiperparámetros globales ─────────────────────────────────────────────────
N_A, N_B       = 2, 4
INPUT_DIM      = 2 * N_A
# SNR_BdB, SNR_EdB = 20, 8
alpha, beta    = 0.1, 0.5

# parámetros de experimentación
ber_threshold  = 0.001
max_epochs     = 40000
batch_sz       = 1024
lr             = 5e-4
device         = 'cpu'
N_experiments  = 1  # número de corridas independientes

# ─── utilidades de canal ──────────────────────────────────────────────────────

def equalize_mmse(y_flat, H, sigma2, eps=1e-12):
    B = y_flat.size(0)
    y_c = y_flat.view(B, N_B, 2)[...,0] + 1j*y_flat.view(B,N_B,2)[...,1]
    H_c = H[...,0] + 1j*H[...,1]

    HhH = torch.matmul(H_c.conj().transpose(-2,-1), H_c)       # B,N_A,N_A
    Rn  = sigma2 * torch.eye(N_A, device=y_flat.device)
    W   = torch.linalg.solve(HhH + Rn + eps, H_c.conj().transpose(-2,-1))
    x_eq = torch.matmul(W, y_c.unsqueeze(-1)).squeeze(-1)
    return torch.view_as_real(x_eq).reshape(B, -1)

def mimo_rayleigh(x, snr_db, na=N_A, nb=N_B):
    B = x.size(0)
    H = torch.randn(B, nb, na, 2, device=x.device) / math.sqrt(2)
    y_c = torch.einsum('bijc,bjc->bic', H, x.view(B, na, 2))
    p_signal = 1
    snr_lin   = 10**(snr_db/10)
    sigma     = math.sqrt(p_signal / (2*snr_lin))
    y_c += sigma * torch.randn_like(y_c)
    return y_c.reshape(B, -1), H


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

def fc(in_f, out_f):
    return nn.Sequential(
        nn.Linear(in_f, out_f),
        nn.BatchNorm1d(out_f),
        nn.Tanh()
    )

# ─── definición de redes ──────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            fc(INPUT_DIM, 64),
            fc(64, 32),
            fc(32, 16),
            fc(16, 8),
            fc(8, 2*N_A)
        )
        self.out  = nn.Linear(2*N_A, 4*N_A)
    def forward(self, m):
        h        = self.body(m)
        mu, logv = self.out(h).chunk(2, dim=-1)
        std      = torch.exp(0.5 * logv)
        x        = mu + std * torch.randn_like(std)
        x        = x / x.norm(dim=1, keepdim=True) * math.sqrt(INPUT_DIM)
        return x, mu, logv

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

# ─── función de entrenamiento con early-stopping por BER ─────────────────────
def train_run(snr):

    SNR_BdB, SNR_EdB = snr, snr
    # instanciar redes y optimizadores
    enc   = Encoder().to(device)
    dec_B = Decoder(2*N_B + 2*N_A*N_B + 1).to(device)
    opt_enc  = torch.optim.AdamW(enc.parameters(), lr=lr , weight_decay=1e-6) 
    opt_decB = torch.optim.AdamW(dec_B.parameters(), lr=lr, weight_decay=1e-6)
    scheduler_enc = OneCycleLR(opt_enc, max_lr=lr, total_steps=max_epochs, pct_start=0.05, anneal_strategy='cos')
    scheduler_decB = OneCycleLR(opt_decB, max_lr=lr, total_steps=max_epochs, pct_start=0.05, anneal_strategy='cos')

    mse      = nn.MSELoss()

    for epoch in range(1, max_epochs+1):
        # 1) generar lote QPSK
        phase = torch.randint(0, 4, (batch_sz, N_A), device=device)
        m = torch.stack([torch.cos(math.pi/2*phase),
                     torch.sin(math.pi/2*phase)], -1).reshape(batch_sz, -1)
        m = m / math.sqrt(2)  # normalizar potenc

        # 2) encode, canal Rayleigh
        x, mu, logv = enc(m)
        y_B, H_B    = mimo_rayleigh(x, SNR_BdB)
        y_E, H_E    = mimo_rayleigh(x, SNR_EdB)


        sigma2_E  = 1/(2*10**(SNR_EdB/10)) 
        y_E_eq = equalize_mmse(y_E, H_E, sigma2_E) 

        # 3) estimación imperfecta de H
        snr_pilot_lin = 10**(30/10)
        sigma_p       = math.sqrt(1/(2*snr_pilot_lin))
        H_hat         = H_B + sigma_p * torch.randn_like(H_B)

        # 4) preparar entrada decoder (iCSI-SVAE)
        y_B_flat     = y_B.reshape(batch_sz, -1)
        H_hat_flat   = H_hat.reshape(batch_sz, -1)
        pnr          = torch.full((batch_sz,1), 10**(SNR_BdB/10), device=device)
        decoder_in   = torch.cat([y_B_flat, H_hat_flat, pnr], dim=1)
        x_hat        = dec_B(decoder_in)

        L = (mse(x_hat, m))

        # 6) backward & step
        opt_enc.zero_grad()
        opt_decB.zero_grad()
        L.backward()
        opt_enc.step()
        scheduler_enc.step()
        opt_decB.step()
        scheduler_decB.step()

        # 7) calcular BER_B y early-stop si corresponde
        with torch.no_grad():
            true_bits = phase.view(-1).to(torch.int64)
            ph_B = torch.atan2(x_hat[:,1::2], x_hat[:,0::2])
            sym_B = ((ph_B % (2*math.pi)) / (math.pi/2)).round() % 4
            sym_B = sym_B.view(-1).to(torch.int64)

            ph_E = torch.atan2(y_E_eq[:,1::2], y_E_eq[:,0::2])
            sym_E = ((ph_E % (2*math.pi)) / (math.pi/2)).round() % 4
            sym_E = sym_E.view(-1).to(torch.int64)

            ber_B = compute_ber(sym_B, true_bits)
            ber_E = compute_ber(sym_E, true_bits)

            print(f"E {epoch:4d} BER_B={ber_B:.4f} BER_E={ber_E:.4f}")

        if ber_B < ber_threshold:
            return epoch, ber_B, ber_E

    # si no llegó al threshold:
    return max_epochs, ber_B, ber_E

# ─── loop de experimentos y registro de resultados ──────────────────────────
results = []
snr_list = [0 , 5 , 10 , 20]
for snr in snr_list:
    epochs_used, final_ber , final_ber_e = train_run(snr)
    results.append({
        'SNR': snr,
        'epochs_used': epochs_used,
        'final_ber': final_ber,
        "final_ber_e": final_ber_e
    })
    print(f"SNR {snr} - Epochs: {epochs_used}, BER_B: {final_ber:.4f}, BER_E: {final_ber_e:.4f}")

# convertir a DataFrame y guardar
df = pd.DataFrame(results)
print(df)
