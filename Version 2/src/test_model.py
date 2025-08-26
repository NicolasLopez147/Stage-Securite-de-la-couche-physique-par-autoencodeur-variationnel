import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ─── parámetros del sistema ──────────────────────────────────────────────────────
N_A, N_B, N_E = 4, 2, 2
INPUT_DIM = 2 * N_A
SNR_BdB = 20
SNR_EdB_range = range(0, 31, 2)
batch_sz = 1024
device = 'cpu'
avg_over = 200  # número de épocas por cada SNR_E para promediar

# ─── funciones auxiliares ───────────────────────────────────────────────────────
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

# ─── modelos cargados ───────────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        in_dim = INPUT_DIM + 2*N_B*N_A
        self.body = nn.Sequential(
            fc(in_dim, 64), fc(64, 32), fc(32, 16), fc(16, 8), fc(8, 2*N_A)
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
            fc(first, 16), fc(16, 32), fc(32, 64), fc(64, 128), fc(128, 256),
            nn.Linear(256, INPUT_DIM)
        )

    def forward(self, y):
        return self.net(y)

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



# ─── cargar modelos entrenados ──────────────────────────────────────────────────
enc = Encoder().to(device)
dec = Decoder(2*N_B + 2*N_A*N_B + 1).to(device)
ckpt = torch.load("checkpoints/full_checkpoint.pth", map_location=device)
enc.load_state_dict(ckpt['enc_state_dict'])
dec.load_state_dict(ckpt['dec_state_dict'])

# ─── evaluación en distintos SNR_E ──────────────────────────────────────────────
ber_B_all = []
ber_E_all = []

for SNR_EdB in SNR_EdB_range:
    total_ber_B, total_ber_E = 0.0, 0.0

    for _ in range(avg_over):
        with torch.no_grad():
            phase = torch.randint(0, 4, (batch_sz, N_A), device=device)
            m = torch.stack([torch.cos(math.pi/2*phase), torch.sin(math.pi/2*phase)], -1).reshape(batch_sz, -1)
            m = m / math.sqrt(2)

            H_B = generate_channel_rayleigh(batch_sz, N_B, N_A, device)
            H_E = generate_channel_rayleigh(batch_sz, N_E, N_A, device)

            x = enc(m, H_B)
            y_B = mimo_rayleigh(x, SNR_BdB, H_B)
            y_E = mimo_rayleigh(x, SNR_EdB, H_E)

            snr_pilot_lin = 10**(30/10)
            sigma_p = math.sqrt(1/(2*snr_pilot_lin))
            H_hat_B = H_B + sigma_p * torch.randn_like(H_B)
            H_hat_E = H_E + sigma_p * torch.randn_like(H_E)

            y_B_flat = y_B.reshape(batch_sz, -1)
            H_B_hat_flat = H_hat_B.reshape(batch_sz, -1)
            pnr_B = torch.full((batch_sz,1), 10**(SNR_BdB/10), device=device)
            decoder_in_B = torch.cat([y_B_flat, H_B_hat_flat, pnr_B], dim=1)
            x_hat_B = dec(decoder_in_B)

            y_E_flat = y_E.reshape(batch_sz, -1)
            H_E_hat_flat = H_hat_E.reshape(batch_sz, -1)
            pnr_E = torch.full((batch_sz,1), 10**(SNR_EdB/10), device=device)
            decoder_in_E = torch.cat([y_E_flat, H_E_hat_flat, pnr_E], dim=1)
            x_hat_E = dec(decoder_in_E)

            true_bits = phase.view(-1).long()
            ph_B = torch.atan2(x_hat_B[:,1::2], x_hat_B[:,0::2])
            sym_B = ((ph_B % (2*math.pi)) / (math.pi/2)).round() % 4
            sym_B = sym_B.view(-1).long()

            ph_E = torch.atan2(x_hat_E[:,1::2], x_hat_E[:,0::2])
            sym_E = ((ph_E % (2*math.pi)) / (math.pi/2)).round() % 4
            sym_E = sym_E.view(-1).long()

            ber_B = compute_ber(sym_B, true_bits)
            ber_E = compute_ber(sym_E, true_bits)

            
            total_ber_B += ber_B
            total_ber_E += ber_E

    print(f"SNR_EdB: {SNR_EdB}, BER_B: {total_ber_B / avg_over:.4f}, BER_E: {total_ber_E / avg_over:.4f}")
    ber_B_all.append(total_ber_B / avg_over)
    ber_E_all.append(total_ber_E / avg_over)

# ─── graficar resultados ───────────────────────────────────────────────────────
plt.plot(SNR_EdB_range, ber_E_all, label='BER Eve', marker='o', color='red')
plt.plot(SNR_EdB_range, ber_B_all, label='BER Bob', marker='x', color='blue')
plt.xlabel('SNR de Eve (dB)')
plt.ylabel('BER')
plt.title('Robustez del sistema frente a SNR variable de Eve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
