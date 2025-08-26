import math, torch, torch.nn as nn

# ─── hiperparámetros ─────────────────────────────────────────────────────────
N_A, N_B = 2, 4
INPUT_DIM = 2 * N_A
SNR_BdB, SNR_EdB = 20, 8
alpha, beta = 0.1, 0.5
epochs, batch_sz = 3000, 2048
lr, device = 1e-3, 'cpu'


# # tabla de conversión: phase → bits
# bit_lut = {
#     0: torch.tensor([0, 0], device=device),
#     1: torch.tensor([0, 1], device=device),
#     2: torch.tensor([1, 1], device=device),
#     3: torch.tensor([1, 0], device=device),
# }

# def phase_to_bits(ph_tensor):
#     """Recibe tensor (B,N_A) con valores 0-3 y devuelve (B,2*N_A) bits"""
#     bits = torch.stack([bit_lut[int(p.item())] for p in ph_tensor.flatten()])
#     return bits.view(ph_tensor.size(0), -1)      # (B, 2*N_A)

# ─── utilidades de canal ──────────────────────────────────────────────────────
def mimo_rayleigh(x, snr_db, na=N_A, nb=N_B):
    B = x.size(0) # número de lotes
    H = torch.randn(B, nb, na, 2, device=x.device) / math.sqrt(2) # Rayleigh

    # producto complejo
    y_c = torch.einsum('bijc,bjc->bic', H, x.view(B, na, 2)) # Multiplicacon matricial 

    # potencia media de la señal por lote
    # p_signal = y_c.pow(2).sum(-1).mean()          # potencia compleja (B,NB)
    p_signal = 1
    snr_lin  = 10**(snr_db/10)
    sigma    = math.sqrt(p_signal / (2*snr_lin))  # var por componente Re/Im
    y_c += sigma * torch.randn_like(y_c) # ruido complejo

    return y_c.reshape(B, -1), H


def kl_div(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def corr_penalty(m, c):
    mc, cc = m - m.mean(0, True), c - c.mean(0, True)
    rho = (mc.T @ cc)/(mc.size(0)-1)
    rho /= mc.std(0, False).unsqueeze(1)*cc.std(0, False).unsqueeze(0)+1e-9
    return -0.5*torch.log1p(-rho.pow(2)+1e-9).mean()

def fc(in_f, out_f):
    return nn.Sequential(
        nn.Linear(in_f, out_f), # capa lineal
        nn.BatchNorm1d(out_f), # normalización por lotes
        nn.Tanh() # función de activación tangente hiperbólica
    )
# Ecualizador de la pseudo-inversa
def equalize(y_flat, H):
    """
    ecualiza y = H x + n  usando la pseudo-inversa H†.
    y_flat : (B, 2*N_B)  vector real [Re1,Im1, … ReNb,ImNb]
    H      : (B, N_B, N_A, 2)  matriz compleja Rayleigh
    devuelve: (B, 2*N_A)  señal igualada
    """
    B = y_flat.size(0)

    # --- convertir a forma compleja ---
    y_c = y_flat.view(B, N_B, 2)[...,0] + 1j*y_flat.view(B, N_B, 2)[...,1]
    H_c = H[...,0] + 1j*H[...,1]           # (B, N_B, N_A) complejo

    # --- pseudo-inversa por lote ---
    W = torch.linalg.pinv(H_c)             # (B, N_A, N_B)
    x_eq_c = (W @ y_c.unsqueeze(-1)).squeeze(-1)  # (B, N_A) complejo

    # --- volver a real concatenado ---
    x_eq = torch.view_as_real(x_eq_c)      # (B, N_A, 2)
    return x_eq.reshape(B, -1)             # (B, 2*N_A)

def equalize_mmse(y_flat, H, sigma2, eps=1e-12):
    B = y_flat.size(0)
    y_c = y_flat.view(B, N_B, 2)[...,0] + 1j*y_flat.view(B,N_B,2)[...,1]
    H_c = H[...,0] + 1j*H[...,1]

    HhH = torch.matmul(H_c.conj().transpose(-2,-1), H_c)       # B,N_A,N_A
    Rn  = sigma2 * torch.eye(N_A, device=y_flat.device)
    W   = torch.linalg.solve(HhH + Rn + eps, H_c.conj().transpose(-2,-1))
    x_eq = torch.matmul(W, y_c.unsqueeze(-1)).squeeze(-1)
    return torch.view_as_real(x_eq).reshape(B, -1)



# ─── redes ────────────────────────────────────────────────────────────────────
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
        h  = self.body(m) # Crea un vector de características a partir de la entrada m ( Pasa por las capas ocultas)
        mu, logvar = self.out(h).chunk(2, dim=-1) # Divide el vector de características en dos partes: mu y logvar 
        std  = torch.exp(0.5 * logvar) # Calcula la desviación estándar a partir de logvar
        x    = mu + std * torch.randn_like(std) # reparametrización
        x = x / x.norm(dim=1, keepdim=True) * math.sqrt(INPUT_DIM) # Normalización de potencia L2
        return x, mu, logvar   # x es lo que enviamos al canal, mu y logvar son los parámetros que se usaran para calcular la divergencia KL y la penalización de correlación

class Decoder(nn.Module):
    def __init__(self, first):
        super().__init__()
        self.first = first 
        self.net = nn.Sequential(
            nn.BatchNorm1d(self.first),
            fc(self.first, 16),
            fc(16, 32),
            fc(32, 64),
            fc(64, 128),
            fc(128, 256),
            nn.Linear(256, INPUT_DIM)
        )
    def forward(self, y): 
        return self.net(y)
# class Decoder(nn.Module):
#     def __init__(self, in_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, 32),
#             nn.ReLU(),
#             nn.Linear(32, 64),
#             nn.ReLU(),
#             nn.Linear(64, 128),
#             nn.ReLU(),
#             nn.Linear(128, INPUT_DIM)      # ***Sin Tanh aquí***
#         )
#     def forward(self, y):
#         return self.net(y)


enc, dec_B = Encoder().to(device), Decoder(2*N_A).to(device)
# dec_E = Decoder(2*N_B).to(device)  # Eve también decodifica
# dec_E.eval()                               # evita BN en modo learn
# for p in dec_E.parameters():
#     p.requires_grad_(False) 

# ─── optimizadores y pérdidas ────────────────────────────────────────────────
opt_enc  = torch.optim.Adam(enc.parameters(), lr=lr , weight_decay=1e-5) # Optimizador para el codificador (ajustar los pesos) usando Adaptive Moment Estimation
opt_decB = torch.optim.Adam(dec_B.parameters(), lr=lr, weight_decay=1e-5) # Optimizador para el decodificador de Bob (ajusar los pesos) usando Adaptive Moment Estimation
mse = nn.MSELoss()

# ─── entrenamiento ───────────────────────────────────────────────────────────

for epoch in range(1, epochs+1):
    # 1. lote de símbolos QPSK I/Q
    phase = torch.randint(0, 4, (batch_sz, N_A), device=device)
    m = torch.stack([torch.cos(math.pi/2*phase),
                     torch.sin(math.pi/2*phase)], -1).reshape(batch_sz, -1)
    m = m / math.sqrt(2)  # normalizar potencia
    # 2. codificar
    x, mu, logv = enc(m)

    # 3. canal rayleigh para Bob
    y_B, H_B = mimo_rayleigh(x, SNR_BdB)
    y_E, H_E = mimo_rayleigh(x, SNR_EdB)  # canal para Eve (no usado)

    # 3.1 ecualizar para Bob
    # y_B_eq = equalize(y_B, H_B)  # (B, 2*N_A)

    snr_lin_B = (10**(-SNR_BdB/10))  
    y_B_eq = equalize_mmse(y_B, H_B, snr_lin_B)
    y_E_eq = equalize_mmse(y_E, H_E, snr_lin_B) 


    # 4. decodificar y pérdidas
    x_hat = dec_B(y_B_eq)


    warm  = 5000
    if epoch/ warm  <= 1:
        t  = 0
    else:
        t = 1                 # nº de épocas de calentamiento
    L = (mse(x_hat, m)
        + t*alpha*kl_div(mu, logv)
        + t*beta *corr_penalty(m, x))

    # 5. optimizar minimizar la perdida
    opt_enc.zero_grad() # Limpia los gradientes
    opt_decB.zero_grad()
    L.backward()        # Calcula los gradientes
    opt_enc.step()      # Actualiza los pesos
    opt_decB.step()

    if epoch % 500 == 0:
        with torch.no_grad():
            # símbolo estimado → fase discreta
            ph_B  = torch.atan2(x_hat[:,1::2], x_hat[:,0::2])
            sym_B = ((ph_B%(2*math.pi))/(math.pi/2)).round()%4
            ser_B = (sym_B != phase).float().mean().item()

            # x_hat_E = dec_E(y_E).detach()  # Eve no entrena
            # ph_E = torch.atan2(x_hat_E[:,1::2], x_hat_E[:,0::2])
            ph_E  = torch.atan2(y_E_eq[:,1::2], y_E_eq[:,0::2])
            sym_E = ((ph_E%(2*math.pi))/(math.pi/2)).round()%4
            ser_E = (sym_E != phase).float().mean().item()

            # bits_true = phase_to_bits(phase)              # (B, 4)
            # bits_hat  = phase_to_bits(sym_B)
            # ber_B = (bits_true != bits_hat).float().mean().item()

        print(f'E {epoch:4d}  loss={L.item():.3f}  SER_B={ser_B} SER_E={ser_E}') # Signal Error Rate (SER)

