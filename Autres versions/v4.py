import math
import torch
import torch.nn as nn

# ─── SAM Wrapper ─────────────────────────────────────────────────────────────
from torch.optim.optimizer import Optimizer

class SAM(Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups   = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        # norma de gradientes
        grad_norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2) for g in self.param_groups for p in g["params"]
                if p.grad is not None
            ]), p=2
        )
        scale = self.param_groups[0]["rho"] / (grad_norm + 1e-12)
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale
                p.add_(e_w)                # perturb
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for g in self.param_groups:
            for p in g["params"]:
                e_w = self.state[p].get("e_w")
                if e_w is not None:
                    p.sub_(e_w)            # restore
        self.base_optimizer.step()    # actual update
        if zero_grad:
            self.zero_grad()

    def step(self, closure):
        assert closure is not None, "SAM needs forward-only closure"
        # 1) forward-only
        loss = closure()
        # 2) backward con create_graph + retain_graph
        self.base_optimizer.zero_grad()
        loss.backward(retain_graph=True, create_graph=True)
        # 3) perturb params
        self.first_step(zero_grad=True)
        # 4) forward-only de nuevo
        loss2 = closure()
        # 5) backward normal
        loss2.backward()
        # 6) restore & base_optimizer.step()
        self.second_step(zero_grad=True)
        return loss2


# ─── Hiperparámetros y Device ────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando device:", device)

N_A, N_B = 2, 4
INPUT_DIM = 2 * N_A
SNR_BdB, SNR_EdB = 20, 8
alpha, beta = 0.1, 0.5
epochs, batch_sz = 100000, 512
lr = 1e-3

# ─── Redes (Encoder & Decoder) ──────────────────────────────────────────────
def fc(in_f, out_f):
    return nn.Sequential(
        nn.Linear(in_f, out_f),
        nn.BatchNorm1d(out_f),
        nn.Tanh()
    )

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
        self.out = nn.Linear(2*N_A, 4*N_A)

    def forward(self, m):
        h = self.body(m)
        mu, logvar = self.out(h).chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        x = mu + std * torch.randn_like(std)
        x = x / x.norm(dim=1, keepdim=True) * math.sqrt(INPUT_DIM)
        return x, mu, logvar

class Decoder(nn.Module):
    def __init__(self, inp_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(inp_dim),
            fc(inp_dim, 16),
            fc(16, 32),
            fc(32, 64),
            fc(64, 128),
            fc(128, 256),
            nn.Linear(256, INPUT_DIM)
        )
    def forward(self, y):
        return self.net(y)

enc   = Encoder().to(device)
dec_B = Decoder(2*N_B + 2*N_B*N_A).to(device)

# ─── Pérdidas y Utilidades ─────────────────────────────────────────────────
mse = nn.MSELoss()

def kl_div(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def corr_penalty(m, c):
    mc, cc = m - m.mean(0, True), c - c.mean(0, True)
    rho = (mc.T @ cc) / (mc.size(0)-1)
    rho /= mc.std(0, False).unsqueeze(1) * cc.std(0, False).unsqueeze(0) + 1e-9
    return -0.5 * torch.log1p(-rho.pow(2) + 1e-9).mean()

def mimo_rayleigh(x, snr_db):
    B = x.size(0)
    H = torch.randn(B, N_B, N_A, 2, device=x.device) / math.sqrt(2)
    y_c = torch.einsum('bijc,bjc->bic', H, x.view(B, N_A, 2))
    sigma = math.sqrt(1/(2*10**(snr_db/10)))
    y_c += sigma * torch.randn_like(y_c)
    return y_c.reshape(B, -1), H

# ─── Instanciación de SAM ──────────────────────────────────────────────────
# Envuelve ambos modelos en un solo optimizador SAM
optimizer = SAM(
    list(enc.parameters()) + list(dec_B.parameters()),
    torch.optim.AdamW,
    rho=0.05,
    lr=lr,
    weight_decay=1e-4
)

# ─── Loop de entrenamiento ─────────────────────────────────────────────────
for epoch in range(1, epochs+1):
    # 1) Genera tu batch y el canal _una sola vez_:
    phase = torch.randint(0, 4, (batch_sz, N_A), device=device)
    m     = torch.stack([torch.cos(math.pi/2*phase),
                         torch.sin(math.pi/2*phase)], -1).view(batch_sz, -1) / math.sqrt(2)

    # Encoder “sin gradientes” solo para mu y logvar previos al canal
    # NO lo uses en closure, lo recalcularemos allí.
    
    # Muestreamos el canal y el ruido UNA VEZ para que sea idéntico en ambas llamadas:
    with torch.no_grad():
        x0 = torch.zeros_like(m)  # placeholder
        # Crea un H fijo y un ruido fijo:
        H_B       = torch.randn(batch_sz, N_B, N_A, 2, device=device) / math.sqrt(2)
        noise = torch.randn(batch_sz, N_B, 2, device=device)  # ruido fijo
        # Definimos una función que aplica canal+ruido de forma determinista:
        def channel(x):
            y_c = torch.einsum('bijc,bjc->bic', H_B, x.view(batch_sz, N_A, 2))
            sigma = math.sqrt(1/(2*10**(SNR_BdB/10)))
            return (y_c + sigma*noise).reshape(batch_sz, -1)

        # Aplícalo sobre un x dummy para forzar tamaño; no se usa el resultado aquí:
        _ = channel(x0)

    # Ahora definimos el closure que recalcula todo:
    def closure():
        optimizer.zero_grad()
        # 1) forward encoder
        x, mu, logvar = enc(m)
        # 2) forward canal (fijo H_B y ruido fijo)
        y_B = channel(x)
        # 3) construye input al decoder
        inp = torch.cat([y_B, H_B.view(batch_sz, -1)], dim=1)
        # 4) forward decoder + pérdida
        x_hat = dec_B(inp)
        L = mse(x_hat, m)
             
        return L

    # Y por fin llamamos a SAM:
    loss = optimizer.step(closure)

    if epoch % 500 == 0:
        with torch.no_grad():
            # Recalcula x_hat igual que en closure
            x, _, _ = enc(m)
            y_B     = channel(x)
            inp     = torch.cat([y_B, H_B.view(batch_sz, -1)], dim=1)
            x_hat   = dec_B(inp)
            ph_B    = torch.atan2(x_hat[:,1::2], x_hat[:,0::2])
            sym_B   = ((ph_B % (2*math.pi))/(math.pi/2)).round() % 4
            ser_B   = (sym_B != phase).float().mean().item()
        print(f"E {epoch:4d} loss={loss.item():.4f} SER_B={ser_B:.4f}")
