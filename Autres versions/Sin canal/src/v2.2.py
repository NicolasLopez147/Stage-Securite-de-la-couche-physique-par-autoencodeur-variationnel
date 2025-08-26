import math
import torch
import torch.nn as nn
import pandas as pd
from tqdm import trange
from torch.optim.lr_scheduler import OneCycleLR
import os
os.makedirs("checkpoints", exist_ok=True)

# ─── hiperparámetros globales ─────────────────────────────────────────────────
N_A, N_B, N_E = 2, 4 , 4
INPUT_DIM = 2 * N_A
SNR_BdB, SNR_EdB = 20, 8
alpha, beta = 0.1, 0.5
lambda_adv = 0.0001

# parámetros de experimentación
ser_threshold  = 0.03
max_epochs     = 80000
batch_sz       = 1024
lr             = 5e-4
device         = 'cpu'
N_experiments  = 1  # número de corridas independientes

# ─── utilidades de canal ──────────────────────────────────────────────────────

def mimo_rayleigh(x, snr_db, H, na=N_A ):
    B = x.size(0)
    y_c = torch.einsum('bijc,bjc->bic', H, x.view(B, na, 2))
    p_signal = 1
    snr_lin   = 10**(snr_db/10)
    sigma     = math.sqrt(p_signal / (2*snr_lin))
    y_c += sigma * torch.randn_like(y_c)
    return y_c.reshape(B, -1)

def generate_channel_rayleigh(batch_sz, n, na, device='cpu'):
    H = torch.randn(batch_sz, n, na, 2, device = device) / math.sqrt(2)
    return H

def kl_div(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def corr_penalty(m, c):
    mc, cc = m - m.mean(0, True), c - c.mean(0, True)
    rho    = (mc.T @ cc) / (mc.size(0)-1)
    rho   /= mc.std(0, False).unsqueeze(1) * cc.std(0, False).unsqueeze(0) + 1e-9
    return -0.5 * torch.log1p(-rho.pow(2) + 1e-9).mean()

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
        self.in_dim = INPUT_DIM + 2*N_B*N_A  
        self.body = nn.Sequential(
            fc(self.in_dim, 64),
            fc(64, 32),
            fc(32, 16),
            fc(16, 8),
            fc(8, 2*N_A)
        )
        self.out  = nn.Linear(2*N_A, 4*N_A)
    def forward(self, m, H_AB):
        B = m.size(0)
        H_flat = H_AB.view(B, -1)
        x_in    = torch.cat([m, H_flat], dim=1)
        h       = self.body(x_in)
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

# ─── función de entrenamiento con early-stopping por SER ─────────────────────
def train_run():
    # instanciar redes y optimizadores
    enc   = Encoder().to(device)
    dec = Decoder(2*N_B).to(device)

    opt_enc  = torch.optim.AdamW(enc.parameters(), lr=lr , weight_decay=1e-6) 
    opt_dec = torch.optim.AdamW(dec.parameters(), lr=lr, weight_decay=1e-6)
    mse      = nn.MSELoss()

    scheduler_enc = OneCycleLR(opt_enc, max_lr=lr, total_steps=max_epochs, pct_start=0.05, anneal_strategy='cos')
    scheduler_dec = OneCycleLR(opt_dec, max_lr=lr, total_steps=max_epochs, pct_start=0.05, anneal_strategy='cos')


    if os.path.exists('checkpoints/full_checkpoint.pth'):

        ckpt = torch.load("checkpoints/full_checkpoint.pth", map_location=device)
        enc.load_state_dict(ckpt['enc_state_dict'])
        dec.load_state_dict(ckpt['dec_state_dict'])

        opt_enc.load_state_dict(ckpt['opt_enc_state_dict'])
        opt_dec.load_state_dict(ckpt['opt_dec_state_dict'])

        

    flag = True
    count = 50000
    for epoch in range(1, max_epochs+1):
        # 1) generar lote QPSK
        phase = torch.randint(0, 4, (batch_sz, N_A), device=device)
        m = torch.stack([torch.cos(math.pi/2*phase),
                     torch.sin(math.pi/2*phase)], -1).reshape(batch_sz, -1)
        m = m / math.sqrt(2)  # normalizar potenc
        H_B = generate_channel_rayleigh(batch_sz, N_B, N_A, device=device)
        H_E = generate_channel_rayleigh(batch_sz, N_E, N_A, device=device)
        # 2) encode, canal Rayleigh
        x, mu, logv = enc(m, H_B)
        y_B = mimo_rayleigh(x, SNR_BdB, H_B)
        y_E = mimo_rayleigh(x, SNR_EdB, H_E)

        # 2) decodificación "ciega" sin canal
        y_B_flat = y_B.reshape(batch_sz, -1)    # (batch_sz, 16)
        x_hat_B  = dec(y_B_flat)

        y_E_flat = y_E.reshape(batch_sz, -1)
        x_hat_E  = dec(y_E_flat)

        # 5) pérdida con warm-up sobre α,β
        #    (aquí uso warm_epochs=2000 como ejemplo)
        # warm_epochs = 2000
        # if epoch <= warm_epochs:
        #     a, b = 0.0, 0.0
        # else:
        #     ramp = min((epoch-warm_epochs)/warm_epochs, 1.0)
        #     a, b = alpha * ramp, beta * ramp

        # L = mse(x_hat, m) + a * kl_div(mu, logv) + b * corr_penalty(m, x)
        L_B = (mse(x_hat_B, m))
        L_E = -(mse(x_hat_E, m))

        if flag:
            L = L_B
        else:
            L = L_B + L_E * lambda_adv
        # L = L_B

        # 6) backward & step
        opt_enc.zero_grad()
        opt_dec.zero_grad()
        L.backward()
        opt_enc.step()
        scheduler_enc.step()
        opt_dec.step()
        scheduler_dec.step()

        # 7) calcular SER_B y early-stop si corresponde
        with torch.no_grad():
            ph_B  = torch.atan2(x_hat_B[:,1::2], x_hat_B[:,0::2])
            sym_B = ((ph_B % (2*math.pi)) / (math.pi/2)).round() % 4
            ser_B = (sym_B != phase).float().mean().item()

            ph_E  = torch.atan2(x_hat_E[:,1::2], x_hat_E[:,0::2])
            sym_E = ((ph_E%(2*math.pi))/(math.pi/2)).round()%4
            ser_E = (sym_E != phase).float().mean().item()

            print(f"E {epoch:4d} SER_B={ser_B:.4f} SER_E={ser_E:.4f}")

        if ser_B < ser_threshold:
            flag = False
            torch.save({
                'enc_state_dict': enc.state_dict(),
                'dec_state_dict': dec.state_dict(),
                'opt_enc_state_dict': opt_enc.state_dict(),
                'opt_dec_state_dict': opt_dec.state_dict(),
                'epoch': epoch,
            }, "checkpoints/full_checkpoint.pth")
            
        if not flag:
            count -= 1
        if count <= 0:
            return epoch, ser_B, ser_E

    # si no llegó al threshold:
    return max_epochs, ser_B, ser_E

# ─── loop de experimentos y registro de resultados ──────────────────────────
results = []
for run in trange(1, N_experiments+1, desc="Experimentos"):
    epochs_used, final_ser , final_ser_e = train_run()
    results.append({
        'run': run,
        'epochs_used': epochs_used,
        'final_ser': final_ser,
        "final_ser_e": final_ser_e
    })
    print(f"Run {run} - Epochs: {epochs_used}, SER_B: {final_ser:.4f}, SER_E: {final_ser_e:.4f}")

# convertir a DataFrame y guardar
df = pd.DataFrame(results)
print(df)

# Aquí el decoder recibe la entrada de y_B, aqui no hay H_B_hat.