"""Kern: entrollte Netze (Skalar-Netz und LISTA), Vorwärts- und Rückwärtsrechnung von Hand, Adam mit Minibatches, Trainingsverteilungen, ISTA und FISTA, Vergleichslöser (Tikhonov, L1 exakt),
Posterior-Mittel und die Lösung der kleinsten Quadrate des linearen Netzes. Nur numpy, kein Streamlit, kein scipy.

Jede Funktion ist eine kleine, einzeln prüfbare Einheit (DEMO-PLAYBOOK §1b): `tests/test_model.py` rechnet sie an der 3 x 3-Mini-Instanz von Hand und `tests/test_oracle_ent.py` gegen
den Differenzenquotienten, ISTA (Stück 13), die exakte Lösung der kleinsten Quadrate und das Posterior-Mittel (Stück 7).

Ein entrolltes ISTA mit K Schichten (Start 0, Zeilenkonvention: B (m, n) Daten, X (m, n) Schätzungen):
    Skalar:  X_{k+1} = soft(X_k - gamma_k (X_k A^T - B) A, theta_k)                 Parameter gamma_k, theta_k (2 K)
    LISTA:   X_{k+1} = soft(B We^T + X_k S^T, theta_k)                              Parameter We, S (gemeinsam), theta_k (2 n^2 + K)
Start der Parameter = ISTA (gamma = t, We = t A^T, S = I - t A^T A, theta = t lambda, t = 1 / |A|_2^2). Verlust: mittlerer relativer quadratischer Fehler |X_K - X|^2 / |X|^2.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

import ent_constants as C
from inverse_generator import SplitMix64, grid, operator

_TINY = 1e-300


def soft(v, t):
    """Weiche Schwelle sign(v) max(|v| - t, 0): die Prox-Abbildung von t |.|_1."""
    v = np.asarray(v, dtype=float)
    return np.sign(v) * np.maximum(np.abs(v) - t, 0.0)


@lru_cache(maxsize=4)
def _op(n: int, w: float) -> np.ndarray:
    a = operator(n, w)
    a.flags.writeable = False
    return a


def op(n: int = C.N, w: float = C.W) -> np.ndarray:
    """Der Operator des Generators, zwischengespeichert."""
    return _op(int(n), round(float(w), 6))


def step_size(a: np.ndarray) -> float:
    """t = 1 / |A|_2^2."""
    return 1.0 / float(np.linalg.norm(a, 2) ** 2)


# ----------------------------------------------------------------------------- Trainingsverteilungen

def exp_kernel(s: np.ndarray, ell: float) -> np.ndarray:
    """Korrelationsmatrix exp(-|s_i - s_j| / ell) (Stück 8)."""
    s = np.asarray(s, dtype=float)
    return np.exp(-np.abs(s[:, None] - s[None, :]) / ell)


def prior_cov(s: np.ndarray) -> np.ndarray:
    """Prior-Kovarianz tau^2 C + 1e-10 I (der Zuschlag hält die Cholesky-Zerlegung stabil)."""
    return C.PRIOR_TAU ** 2 * exp_kernel(s, C.PRIOR_ELL) + 1e-10 * np.eye(len(s))


def _spikes(r: np.ndarray, n: int) -> np.ndarray:
    x = np.zeros(n)
    for j in range(C.SPIKE_COUNT[0] + int(r[0] * C.SPIKE_COUNT[1])):
        x[int(r[1 + 2 * j] * n) % n] += C.SPIKE_HEIGHT[0] + C.SPIKE_HEIGHT[1] * r[2 + 2 * j]
    return x


def _blocks(r: np.ndarray, n: int) -> np.ndarray:
    x = np.zeros(n)
    for j in range(C.BLOCK_COUNT[0] + int(r[0] * C.BLOCK_COUNT[1])):
        w = C.BLOCK_WIDTH[0] + int(r[1 + 3 * j] * C.BLOCK_WIDTH[1])
        left = int(r[2 + 3 * j] * (n - w))
        x[left:left + w] += C.BLOCK_HEIGHT[0] + C.BLOCK_HEIGHT[1] * r[3 + 3 * j]
    return x


def _waves(r: np.ndarray, n: int) -> np.ndarray:
    s = grid(n)
    per = C.WAVE_PERIODS[0] + int(r[0] * C.WAVE_PERIODS[1])
    lo, hi = C.WAVE_RANGE
    return np.abs((per * s + r[1]) % 1.0 - 0.5) * 2.0 * ((s > lo) & (s < hi))


def _phantom(r: np.ndarray, n: int) -> np.ndarray:
    s = grid(n)
    x = np.zeros(n)
    for j in range(2):
        c = C.PHANTOM_CENTER[0] + C.PHANTOM_CENTER[1] * r[3 * j]
        wd = C.PHANTOM_WIDTH[0] + C.PHANTOM_WIDTH[1] * r[3 * j + 1]
        h = C.PHANTOM_HEIGHT[0] + C.PHANTOM_HEIGHT[1] * r[3 * j + 2]
        x += h * np.exp(-((s - c) ** 2) / (2.0 * wd * wd))
    left = C.PHANTOM_BOX_LEFT[0] + C.PHANTOM_BOX_LEFT[1] * r[6]
    return x + (C.PHANTOM_BOX_HEIGHT[0] + C.PHANTOM_BOX_HEIGHT[1] * r[7]) * ((s >= left) & (s < left + C.PHANTOM_BOX_WIDTH))


_DRAW = {"spikes": _spikes, "blocks": _blocks, "waves": _waves, "phantom": _phantom}


def draw_signals(world: str, count: int, seed: int, n: int = C.N) -> np.ndarray:
    """(count, n) zufällige Signale der Welt aus dem festen Strom `seed`. spikes: 2 bis 4 Spitzen, Orte gleichverteilt, Höhe U(1; 2,5); blocks: 2 bis 4 Blöcke, Breite 3 bis 10 Zellen, Höhe
    U(0,3; 1,2); waves: Dreieckswelle mit 3 bis 5 Perioden und zufälliger Phase; phantom: zwei Hügel und ein Kasten; gauss: Ziehungen aus dem Prior (Stück 8); all: je ein Viertel der vier
    Welten (die ersten count mod 4 Welten bekommen ein Signal mehr)."""
    if world == "all":
        sizes = [count // 4 + (1 if i < count % 4 else 0) for i in range(4)]
        return np.vstack([draw_signals(w, sz, seed + i + 1, n) for i, (w, sz) in enumerate(zip(C.TEST_WORLDS, sizes))])
    rng = SplitMix64(seed)
    if world == "gauss":
        return rng.normal(count * n).reshape(count, n) @ np.linalg.cholesky(prior_cov(grid(n))).T
    u = rng.uniform(count * C.DRAW_WIDTH).reshape(count, C.DRAW_WIDTH)
    return np.array([_DRAW[world](u[i], n) for i in range(count)])


def make_data(signals: np.ndarray, noise: float, seed: int, a: np.ndarray | None = None):
    """(B, sigma): Daten B = X A^T + e, absolutes Rauschen je Signal sigma_i = noise mal Effektivwert der exakten Daten."""
    a = op() if a is None else a
    clean = signals @ a.T
    sig = noise * np.sqrt(np.mean(clean ** 2, axis=1))
    e = SplitMix64(seed).normal(clean.size).reshape(clean.shape) * sig[:, None]
    return clean + e, sig


def rel_err(x: np.ndarray, xt: np.ndarray) -> np.ndarray:
    """Relativer Fehler |x - xt| / |xt| je Zeile."""
    return np.linalg.norm(np.asarray(x) - xt, axis=-1) / np.linalg.norm(xt, axis=-1)


# ----------------------------------------------------------------------------- Netze

def init_params(kind: str, k_layers: int, lam0: float, a: np.ndarray | None = None) -> dict:
    """Start = ISTA: Skalar gamma = t, theta = t lam0; LISTA We = t A^T, S = I - t A^T A, theta = t lam0."""
    a = op() if a is None else a
    t = step_size(a)
    th = np.full(k_layers, t * lam0)
    if kind == "scalar":
        return {"gam": np.full(k_layers, t), "th": th}
    return {"We": t * a.T.copy(), "Sm": np.eye(a.shape[1]) - t * (a.T @ a), "th": th}


def forward(p: dict, b: np.ndarray, kind: str, a: np.ndarray | None = None):
    """(X_K, Zwischenspeicher) für Blöcke b (m, n); K ist die Länge von p['th']."""
    a = op() if a is None else a
    x = np.zeros_like(b)
    cache = []
    for k in range(len(p["th"])):
        if kind == "scalar":
            r = x @ a.T - b
            z = x - p["gam"][k] * (r @ a)
            cache.append((x, r, z))
        else:
            z = b @ p["We"].T + x @ p["Sm"].T
            cache.append((x, None, z))
        x = soft(z, p["th"][k])
    return x, cache


def apply(p: dict, b: np.ndarray, kind: str, a: np.ndarray | None = None) -> np.ndarray:
    """Die Schätzungen X_K des Netzes."""
    return forward(p, np.atleast_2d(b), kind, a)[0]


def loss_and_grad(p: dict, b: np.ndarray, xt: np.ndarray, kind: str, weights: np.ndarray, a: np.ndarray | None = None):
    """Verlust sum_i w_i |X_K,i - X_i|^2 / m und seine Gradienten nach allen Parametern (Rückwärtsrechnung von Hand durch die K Schichten; Maske = Zahl der Eingänge über der Schwelle)."""
    a = op() if a is None else a
    gm = a.T @ a
    xk, cache = forward(p, b, kind, a)
    diff = xk - xt
    loss = float(np.sum(weights * np.sum(diff ** 2, axis=1)) / len(b))
    g = 2.0 * weights[:, None] * diff / len(b)
    k_layers = len(p["th"])
    grads = {"th": np.zeros(k_layers)}
    if kind == "scalar":
        grads["gam"] = np.zeros(k_layers)
    else:
        grads["We"], grads["Sm"] = np.zeros_like(p["We"]), np.zeros_like(p["Sm"])
    for k in range(k_layers - 1, -1, -1):
        x, r, z = cache[k]
        dz = g * (np.abs(z) > p["th"][k])
        grads["th"][k] = -float(np.sum(np.sign(z) * dz))
        if kind == "scalar":
            grads["gam"][k] = -float(np.sum(dz * (r @ a)))
            g = dz - p["gam"][k] * (dz @ gm)
        else:
            grads["We"] += dz.T @ b
            grads["Sm"] += dz.T @ x
            g = dz @ p["Sm"]
    return loss, grads


def batch_order(count: int, seed: int) -> np.ndarray:
    """Zufällige Reihenfolge der Trainingssignale aus dem Strom (Rangordnung gleichverteilter Werte, nicht numpy.random)."""
    return np.argsort(SplitMix64(seed).uniform(count))


def train(kind: str, k_layers: int, b: np.ndarray, xt: np.ndarray, lam0: float, steps: int, seed: int = 0, val: tuple | None = None, a: np.ndarray | None = None):
    """Adam mit Minibatches (C.BATCH, ohne Zurücklegen je Epoche, Reihenfolge aus SplitMix64) und Kosinus-Lernrate bis C.COSINE_FLOOR; Start = ISTA mit lam0. Gibt (Parameter, Verlauf) zurück;
    der Verlauf hat C.LOG_POINTS + 1 Einträge {step, train, val} (Median des relativen Fehlers auf allen Trainingssignalen und auf `val` = (b, x), sonst nan)."""
    a = op() if a is None else a
    p = init_params(kind, k_layers, lam0, a)
    weights = 1.0 / np.sum(xt ** 2, axis=1)
    m1 = {k: np.zeros_like(v) for k, v in p.items()}
    m2 = {k: np.zeros_like(v) for k, v in p.items()}
    batch = min(C.BATCH, len(b))
    order, pos, epoch = batch_order(len(b), seed), 0, 0
    marks = {round(steps * i / C.LOG_POINTS) for i in range(C.LOG_POINTS + 1)}
    hist = []

    def log(step):
        hist.append({"step": step, "train": float(np.median(rel_err(apply(p, b, kind, a), xt))),
                     "val": float(np.median(rel_err(apply(p, val[0], kind, a), val[1]))) if val is not None else math.nan})

    log(0)
    for t in range(1, steps + 1):
        if pos + batch > len(b):
            epoch += 1
            order, pos = batch_order(len(b), seed + epoch), 0
        idx = order[pos:pos + batch]
        pos += batch
        scale = C.COSINE_FLOOR + (1.0 - C.COSINE_FLOOR) * 0.5 * (1.0 + math.cos(math.pi * t / steps))
        _, gr = loss_and_grad(p, b[idx], xt[idx], kind, weights[idx], a)
        for k in p:
            rate = C.LR[kind] * (1.0 if k in ("gam", "th") else C.LR_MATRIX_FACTOR) * scale
            m1[k] = C.ADAM_B1 * m1[k] + (1.0 - C.ADAM_B1) * gr[k]
            m2[k] = C.ADAM_B2 * m2[k] + (1.0 - C.ADAM_B2) * gr[k] ** 2
            p[k] = p[k] - rate * (m1[k] / (1.0 - C.ADAM_B1 ** t)) / (np.sqrt(m2[k] / (1.0 - C.ADAM_B2 ** t)) + C.ADAM_EPS)
        p["th"] = np.maximum(p["th"], 0.0)
        if t in marks:
            log(t)
    return p, hist


# ----------------------------------------------------------------------------- Vergleichsverfahren

def ista(b: np.ndarray, lam: float, k_steps: int, accel: bool = False, a: np.ndarray | None = None) -> np.ndarray:
    """ISTA (accel: FISTA) für Blöcke b (m, n), Schrittweite t, Schwelle t lam, Start 0, k_steps Schritte."""
    a = op() if a is None else a
    t = step_size(a)
    b = np.atleast_2d(np.asarray(b, dtype=float))
    x = np.zeros_like(b)
    y, mom = x.copy(), 1.0
    for _ in range(k_steps):
        xn = soft(y - t * ((y @ a.T - b) @ a), t * lam)
        if accel:
            mn = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * mom * mom))
            y = xn + ((mom - 1.0) / mn) * (xn - x)
            mom = mn
        else:
            y = xn
        x = xn
    return x


def tikhonov_svd(a: np.ndarray, b: np.ndarray, lams) -> np.ndarray:
    """Tikhonov-Lösungen x = V diag(s / (s^2 + lam^2)) U^T b für Blöcke b (m, n) und alle lam: Form (len(lams), m, n)."""
    u, sv, vt = np.linalg.svd(a)
    b = np.atleast_2d(b)
    return np.array([((b @ u) * (sv / (sv ** 2 + lam ** 2))) @ vt for lam in lams])


def lasso(m: np.ndarray, y: np.ndarray, lam: float, maxit: int = C.LASSO_MAXIT, tol: float = 1e-10) -> np.ndarray:
    """Exakte Lösung von min 1/2 |m w - y|^2 + lam |w|_1 per Aktivmenge (wie in Stück 6, 12 und 13)."""
    p = m.shape[1]
    mtm, mty = m.T @ m, m.T @ y
    w = np.zeros(p)
    active = np.zeros(p, dtype=bool)
    sgn = np.zeros(p)
    for _ in range(maxit):
        idx = np.flatnonzero(active)
        if idx.size:
            s = sgn[idx]
            sol = np.linalg.solve(mtm[idx][:, idx], mty[idx] - lam * s)
            bad = sol * s < 0.0
            if bad.any():
                w_old = w[idx]
                with np.errstate(divide="ignore", invalid="ignore"):
                    t = np.where(bad, w_old / (w_old - sol), np.inf)
                j = int(np.argmin(t))
                w[idx[j]] = 0.0
                active[idx[j]] = False
                continue
            w[idx] = sol
        g = mtm @ w - mty
        viol = np.where(active, 0.0, np.abs(g) - lam)
        j = int(np.argmax(viol))
        if viol[j] <= tol * max(lam, _TINY):
            return w
        active[j] = True
        sgn[j] = -np.sign(g[j])
        w[j] = 0.0
    return w


def posterior_gain(a: np.ndarray, sigma: float, p: np.ndarray) -> np.ndarray:
    """P A^T (A P A^T + sigma^2 I)^-1: das Posterior-Mittel ist gain @ b (Stück 7)."""
    return p @ a.T @ np.linalg.inv(a @ p @ a.T + sigma ** 2 * np.eye(a.shape[0]))


def lstsq_linear(b: np.ndarray, xt: np.ndarray, ridge: float = 1e-12) -> np.ndarray:
    """Die Matrix We des linearen Netzes (K = 1, theta = 0), die den gewichteten Verlust sum_i |We b_i - x_i|^2 / |x_i|^2 minimiert (exakt, normale Gleichungen): X = B We^T."""
    w = 1.0 / np.sum(xt ** 2, axis=1)
    return np.linalg.solve((b * w[:, None]).T @ b + ridge * np.eye(b.shape[1]), (b * w[:, None]).T @ xt).T
