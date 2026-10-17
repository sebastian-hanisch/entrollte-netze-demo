"""Auswertung: Formatierung, Training und Analyse einer Einstellung, die vier Experimente über feste Ströme und die Verdikte. Nur numpy, kein Streamlit."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import ent_constants as C
import ent_model as model
from inverse_generator import grid

_SUPER = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


# ----------------------------------------------------------------------------- Formatierung

def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def sci(x: float, digits: int = 1) -> str:
    """Zahl in Zehnerpotenz-Schreibweise, z. B. 2,5·10⁹."""
    if x == 0.0 or not math.isfinite(x):
        return de(x, digits) if math.isfinite(x) else "unendlich"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10.0 ** e
    if round(abs(m), digits) >= 10.0:
        m, e = m / 10.0, e + 1
    return f"{de(m, digits)}·10{str(e).translate(_SUPER)}"


def err_pct(err: float) -> str:
    """Fehler als Prozent der Signalnorm: ab 100 Prozent ohne, ab 1 Prozent mit einer, ab 0,01 Prozent mit zwei Stellen, darunter als Zehnerpotenz;
    ab dem Zehnfachen der Signalnorm als Vielfaches."""
    if not math.isfinite(err):
        return "unendlich"
    if err >= 10.0:
        return f"{sci(err)}-fach"
    x = 100.0 * err
    if x < 1e-9:
        return "0 %"
    if x >= 99.5:
        return f"{de(x, 0)} %"
    if x >= 0.95:
        return f"{de(x, 1)} %"
    if x >= 0.01:
        return f"{de(x, 2)} %"
    return f"{sci(x)} %"


def noise_label(v: float) -> str:
    """Rauschstufe als Prozent des Effektivwerts, z. B. 0,1 %."""
    return de(100.0 * v, 4).rstrip("0").rstrip(",") + " %" if v > 0.0 else "0 %"


def _by(rows: list, **kw) -> dict:
    return next(r for r in rows if all(r[k] == v for k, v in kw.items()))


# ----------------------------------------------------------------------------- Vergleichsverfahren (Orakel-λ auf den Testsignalen)

def _best_lambda(fn, b: np.ndarray, x: np.ndarray):
    """(Median-Fehler, λ) des besten λ aus C.LAMBDA_GRID für das Verfahren fn(b, λ) -> Schätzungen."""
    errs = [float(np.median(model.rel_err(fn(b, lam), x))) for lam in C.LAMBDA_GRID]
    k = int(np.argmin(errs))
    return errs[k], float(C.LAMBDA_GRID[k])


def best_ista(b: np.ndarray, x: np.ndarray, k_steps: int, accel: bool = False):
    """Bestes λ (Orakel auf (b, x)) für ISTA oder FISTA mit k_steps Schritten: (Median-Fehler, λ)."""
    return _best_lambda(lambda bb, lam: model.ista(bb, lam, k_steps, accel), b, x)


def best_tikhonov(b: np.ndarray, x: np.ndarray):
    a = model.op()
    errs = [float(np.median(model.rel_err(xs, x))) for xs in model.tikhonov_svd(a, b, C.LAMBDA_GRID)]
    k = int(np.argmin(errs))
    return errs[k], float(C.LAMBDA_GRID[k])


def best_l1(b: np.ndarray, x: np.ndarray):
    """Exaktes L1 (Aktivmenge) beim besten λ des Gitters, Median über die Zeilen von (b, x): (Fehler, λ, Schätzung der ersten Zeile)."""
    a = model.op()
    sols = [np.array([model.lasso(a, bb, lam) for bb in b]) for lam in C.LAMBDA_GRID]
    errs = [float(np.median(model.rel_err(s, x))) for s in sols]
    k = int(np.argmin(errs))
    return errs[k], float(C.LAMBDA_GRID[k]), sols[k][0]


# ----------------------------------------------------------------------------- Einstellung, Training, Analyse

@dataclass(frozen=True)
class Settings:
    kind: str = "lista"
    k: int = C.DEFAULT_K
    train_world: str = "spikes"
    test_world: str = "spikes"
    noise: float = C.DEFAULT_NOISE
    n_train: int = C.DEFAULT_N_TRAIN
    steps: int = C.DEFAULT_STEPS
    seed: int = C.DEFAULT_SEED


def train_data(world: str, count: int, noise: float, seed: int):
    """(X, B) Trainingssignale und Daten aus den festen Strömen der Einstellung."""
    x = model.draw_signals(world, count, seed + C.SEED_TRAIN)
    return x, model.make_data(x, noise, seed + C.SEED_TRAIN_NOISE)[0]


def test_data(world: str, count: int, noise: float, seed: int):
    """(X, B) Testsignale und Daten (andere Ströme als beim Training)."""
    x = model.draw_signals(world, count, seed + C.SEED_TEST)
    return x, model.make_data(x, noise, seed + C.SEED_TEST_NOISE)[0]


def train_network(kind: str, k_layers: int, world: str, count: int, noise: float, steps: int, seed: int, val: tuple | None = None):
    """Trainiert ein Netz auf zufälligen Signalen der Welt: Start = ISTA mit dem besten λ auf den Trainingsdaten. Gibt (Parameter, Verlauf, λ des Starts) zurück."""
    x, b = train_data(world, count, noise, seed)
    _, lam0 = best_ista(b, x, k_layers)
    p, hist = model.train(kind, k_layers, b, x, lam0, steps, seed, val=val)
    return p, hist, lam0


@dataclass
class Analysis:
    settings: Settings
    s: np.ndarray
    params: dict
    hist: list                # Verlauf des Trainings: step, train, val
    lam0: float               # λ des ISTA-Starts
    t: float                  # Schrittweite 1 / |A|²
    x: np.ndarray             # Beispiel-Testsignal (das erste)
    b: np.ndarray
    x_net: np.ndarray
    x_ista: np.ndarray
    x_fista: np.ndarray
    x_l1: np.ndarray
    x_tik: np.ndarray
    err_net: float            # Median-Fehler auf den Testsignalen
    err_train: float          # Median-Fehler auf den Trainingssignalen
    err_ista: float           # ISTA mit k Schritten, bestes λ auf den Testsignalen
    err_fista: float
    err_l1: float             # exaktes L1 (BASE_COUNT Testsignale)
    err_tik: float
    lam_ista: float
    lam_fista: float
    lam_l1: float
    gam_max: float            # größte gelernte Schrittweite durch t (nur Skalar-Netz, sonst nan)
    th_zero: bool             # alle Schwellen abgeschaltet
    train_noise: float


def analyse(settings: Settings) -> Analysis:
    st = settings
    a = model.op()
    t = model.step_size(a)
    xv, bv = test_data(st.test_world, C.TEST_COUNT, st.noise, st.seed)
    p, hist, lam0 = train_network(st.kind, st.k, st.train_world, st.n_train, st.noise, st.steps, st.seed, val=(bv, xv))
    xt, bt = train_data(st.train_world, st.n_train, st.noise, st.seed)
    out = model.apply(p, bv, st.kind)
    e_i, lam_i = best_ista(bv, xv, st.k)
    e_f, lam_f = best_ista(bv, xv, st.k, True)
    e_t, lam_t = best_tikhonov(bv, xv)
    e_l, lam_l, x_l1 = best_l1(bv[:C.BASE_COUNT], xv[:C.BASE_COUNT])
    xs_t = model.tikhonov_svd(a, bv[:1], [lam_t])[0, 0]
    return Analysis(
        settings=st, s=grid(C.N), params=p, hist=hist, lam0=lam0, t=t, x=xv[0], b=bv[0], x_net=out[0], x_ista=model.ista(bv[:1], lam_i, st.k)[0], x_fista=model.ista(bv[:1], lam_f, st.k, True)[0],
        x_l1=x_l1, x_tik=xs_t, err_net=float(np.median(model.rel_err(out, xv))), err_train=float(np.median(model.rel_err(model.apply(p, bt, st.kind), xt))), err_ista=e_i, err_fista=e_f,
        err_l1=e_l, err_tik=e_t, lam_ista=lam_i, lam_fista=lam_f, lam_l1=lam_l, gam_max=float(p["gam"].max() / t) if "gam" in p else math.nan, th_zero=bool((p["th"] <= C.ZERO_TH).all()),
        train_noise=st.noise,
    )


# ----------------------------------------------------------------------------- Experiment 1: was das Training lernt

def posterior_experiment() -> dict:
    """Prior-Ziehungen (dichte Signale, Stück 8) als Wahrheit, Rauschen 1 %: Posterior-Mittel (exakt), Tikhonov, ISTA und FISTA (50 Schritte), exaktes L1 und die Netze mit POST_K Schichten.
    Dazu die exakte Lösung der kleinsten Quadrate des linearen Netzes (K = 1, θ = 0) bei POST_SIZES Trainingssignalen: Abstand zur Matrix des Posterior-Mittels und Test-Fehler."""
    a = model.op()
    p_cov = model.prior_cov(grid(C.N))
    seed = C.SEED_BASE
    xv, bv = test_data("gauss", 500, 1e-2, seed)
    sig = model.make_data(xv, 1e-2, seed + C.SEED_TEST_NOISE)[1]
    gain = model.posterior_gain(a, float(np.sqrt(np.mean(sig ** 2))), p_cov)
    pm = float(np.median(model.rel_err(bv @ gain.T, xv)))
    ls = []
    for size in C.POST_SIZES:
        x, b = train_data("gauss", size, 1e-2, seed)
        we = model.lstsq_linear(b, x)
        ls.append({"size": size, "dist": float(np.linalg.norm(we - gain) / np.linalg.norm(gain)), "err": float(np.median(model.rel_err(bv @ we.T, xv)))})
    nets = []
    for kind in C.KINDS:
        p, hist, lam0 = train_network(kind, C.POST_K, "gauss", C.DEFAULT_N_TRAIN, 1e-2, C.EXP_STEPS, seed)
        nets.append({"kind": kind, "err": float(np.median(model.rel_err(model.apply(p, bv, kind), xv))), "th": [float(v) for v in p["th"]], "th_zero": bool((p["th"] <= C.ZERO_TH).all())})
    return {"pm": pm, "ls": ls, "nets": nets, "tik": best_tikhonov(bv, xv)[0], "l1": best_l1(bv[:C.BASE_COUNT], xv[:C.BASE_COUNT])[0], "ista": best_ista(bv, xv, 50)[0],
            "fista": best_ista(bv, xv, 50, True)[0], "k": C.POST_K}


# ----------------------------------------------------------------------------- Experiment 2: Schichten statt Schritte

def _equivalent_steps(err_net: float, b: np.ndarray, x: np.ndarray, k_layers: int, accel: bool):
    """Kleinste Schrittzahl aus k mal EQUIV_FACTORS, bei der ISTA (accel: FISTA) mit bestem λ höchstens den Fehler des Netzes hat; None, wenn keine der Schrittzahlen reicht."""
    for f in C.EQUIV_FACTORS:
        if best_ista(b, x, f * k_layers, accel)[0] <= err_net:
            return f * k_layers
    return None


def layers_experiment() -> dict:
    """Spitzen-Welt, Rauschen 1 %: Netze mit LAYER_KS Schichten gegen ISTA und FISTA mit denselben Schichten und die äquivalente Schrittzahl (kleinste Zahl aus K mal EQUIV_FACTORS, bei der
    ISTA beziehungsweise FISTA den Fehler des Netzes erreicht); Grenzen: exaktes L1 und Tikhonov; die gelernten Parameter des Skalar-Netzes bei der größten Schichtzahl."""
    seed = C.SEED_BASE
    xv, bv = test_data("spikes", C.TEST_COUNT, 1e-2, seed)
    rows = []
    learned = None
    t = model.step_size(model.op())
    for k in C.LAYER_KS:
        row = {"k": k, "ista": best_ista(bv, xv, k)[0], "fista": best_ista(bv, xv, k, True)[0]}
        for kind in C.KINDS:
            p, hist, lam0 = train_network(kind, k, "spikes", C.DEFAULT_N_TRAIN, 1e-2, C.EXP_STEPS, seed)
            err = float(np.median(model.rel_err(model.apply(p, bv, kind), xv)))
            row[kind] = err
            row[kind + "_equiv_ista"] = _equivalent_steps(err, bv, xv, k, False)
            row[kind + "_equiv_fista"] = _equivalent_steps(err, bv, xv, k, True)
            if kind == "scalar" and k == C.LAYER_KS[-1]:
                learned = {"k": k, "gam": [float(g / t) for g in p["gam"]], "lam": [float(th / t) for th in p["th"]], "lam0": lam0}
        rows.append(row)
    return {"rows": rows, "l1": best_l1(bv[:C.BASE_COUNT], xv[:C.BASE_COUNT])[0], "tik": best_tikhonov(bv, xv)[0], "learned": learned}


# ----------------------------------------------------------------------------- Experiment 3: Verteilungs- und Rauschwechsel

def shift_experiment() -> dict:
    """K = EXP_K Schichten, Rauschen 1 %. Teil 1: LISTA trainiert auf SHIFT_WORLDS, getestet auf TEST_WORLDS (Median-Fehler); dazu das Skalar-Netz auf Spitzen trainiert und ISTA und
    Tikhonov (bestes λ je Testwelt). Teil 2: LISTA auf Spitzen mit Trainingsrauschen 0,1 / 1 / 10 %, getestet bei jedem."""
    seed = C.SEED_BASE
    tests = {w: test_data(w, C.TEST_COUNT, 1e-2, seed) for w in C.TEST_WORLDS}
    rows = []
    for tw in C.SHIFT_WORLDS:
        p, _, _ = train_network("lista", C.EXP_K, tw, C.DEFAULT_N_TRAIN, 1e-2, C.EXP_STEPS, seed)
        rows.append({"kind": "lista", "train": tw, "errs": {w: float(np.median(model.rel_err(model.apply(p, b, "lista"), x))) for w, (x, b) in tests.items()}})
    p, _, _ = train_network("scalar", C.EXP_K, "spikes", C.DEFAULT_N_TRAIN, 1e-2, C.EXP_STEPS, seed)
    rows.append({"kind": "scalar", "train": "spikes", "errs": {w: float(np.median(model.rel_err(model.apply(p, b, "scalar"), x))) for w, (x, b) in tests.items()}})
    base = {"ista": {w: best_ista(b, x, C.EXP_K)[0] for w, (x, b) in tests.items()}, "tik": {w: best_tikhonov(b, x)[0] for w, (x, b) in tests.items()}}
    noise_rows = []
    for tn in (C.SHIFT_NOISES[0], 1e-2, C.SHIFT_NOISES[1]):
        p, _, _ = train_network("lista", C.EXP_K, "spikes", C.DEFAULT_N_TRAIN, tn, C.EXP_STEPS, seed)
        errs = {}
        for te in (C.SHIFT_NOISES[0], 1e-2, C.SHIFT_NOISES[1]):
            x, b = test_data("spikes", C.TEST_COUNT, te, seed)
            errs[te] = float(np.median(model.rel_err(model.apply(p, b, "lista"), x)))
        noise_rows.append({"train": tn, "errs": errs})
    return {"rows": rows, "base": base, "noise_rows": noise_rows, "k": C.EXP_K}


# ----------------------------------------------------------------------------- Experiment 4: Überanpassung

def overfit_experiment() -> dict:
    """K = EXP_K Schichten, Spitzen, Rauschen 1 %: Trainingsfehler und Testfehler (Median) bei OVERFIT_SIZES Trainingssignalen für LISTA; das Skalar-Netz bei der kleinsten und größten Zahl."""
    seed = C.SEED_BASE
    xv, bv = test_data("spikes", C.TEST_COUNT, 1e-2, seed)
    rows = []
    for kind, sizes in (("lista", C.OVERFIT_SIZES), ("scalar", (C.OVERFIT_SIZES[0], C.OVERFIT_SIZES[-1]))):
        for size in sizes:
            p, _, _ = train_network(kind, C.EXP_K, "spikes", size, 1e-2, C.EXP_STEPS, seed)
            x, b = train_data("spikes", size, 1e-2, seed)
            n_par = int(sum(v.size for v in p.values()))
            rows.append({"kind": kind, "size": size, "train": float(np.median(model.rel_err(model.apply(p, b, kind), x))), "test": float(np.median(model.rel_err(model.apply(p, bv, kind), xv))), "params": n_par})
    return {"rows": rows, "ista": best_ista(bv, xv, C.EXP_K)[0], "k": C.EXP_K}


# ----------------------------------------------------------------------------- Verdikte: (Art, Text)

def verdict_instance(a: Analysis) -> tuple[str, str]:
    """Lohnt das Netz? Testfehler gegen das bessere von ISTA und FISTA mit derselben Schichtzahl (Schwelle GAIN); dazu Hinweise: exaktes L1 besser, Überanpassung, Verteilungswechsel,
    abgeschaltete Schwellen, gelernte Schrittweiten über 2 / L."""
    st = a.settings
    base = min(a.err_ista, a.err_fista)
    lead = (f"{C.KIND_SHORT[st.kind]}, {st.k} Schichten, Trainingswelt {C.WORLD_SHORT[st.train_world]} ({st.n_train} Signale, {st.steps} Schritte), Testwelt {C.WORLD_SHORT[st.test_world]}, "
            f"Rauschen {noise_label(st.noise)}: Testfehler {err_pct(a.err_net)} (Trainingsfehler {err_pct(a.err_train)}; ISTA {err_pct(a.err_ista)}, FISTA {err_pct(a.err_fista)} mit {st.k} Schritten; "
            f"exaktes L1 {err_pct(a.err_l1)}, Tikhonov {err_pct(a.err_tik)}).")
    if a.err_net <= C.GAIN * base:
        kind, text = "success", f"{lead} Das Netz lohnt: der Testfehler liegt unter 80 % des besseren von ISTA und FISTA mit derselben Schichtzahl."
    elif a.err_net <= base:
        kind, text = "info", f"{lead} Das Netz ist kaum besser als das bessere von ISTA und FISTA mit derselben Schichtzahl."
    else:
        kind, text = "warning", f"{lead} Das Netz lohnt nicht: ISTA oder FISTA mit derselben Schichtzahl sind besser."
    if a.err_l1 < a.err_net:
        text += f" Das exakte L1 ist besser ({err_pct(a.err_l1)}): das Netz holt die Iteration ein, nicht die Lösung."
    if a.err_train <= C.OVERFIT * a.err_net:
        text += f" Überanpassung: der Trainingsfehler ({err_pct(a.err_train)}) ist höchstens halb so groß wie der Testfehler."
    if st.train_world != st.test_world:
        text += f" Verteilungswechsel: Trainingswelt {C.WORLD_SHORT[st.train_world]}, Testwelt {C.WORLD_SHORT[st.test_world]}."
    if a.th_zero:
        text += " Alle Schwellen sind abgeschaltet: das Netz ist linear und schätzt das Posterior-Mittel der Trainingsverteilung."
    if a.gam_max > C.STEP_OVER:
        text += f" Die größte gelernte Schrittweite ist das {de(a.gam_max, 1)}-Fache von 1/‖A‖² (über dem Doppelten): das ist keine konvergente Iteration mehr."
    return kind, text


def verdict_posterior(res: dict) -> tuple[str, str]:
    ls = res["ls"]
    lista = _by(res["nets"], kind="lista")
    scal = _by(res["nets"], kind="scalar")
    return "info", (f"Bei Prior-Ziehungen ist das Posterior-Mittel mit {err_pct(res['pm'])} nicht zu unterbieten (Tikhonov {err_pct(res['tik'])}, exaktes L1 {err_pct(res['l1'])}). Die exakte Lösung der kleinsten "
                    f"Quadrate des linearen Netzes (K = 1, θ = 0) nähert sich ihm mit der Zahl der Trainingssignale: Matrixabstand {de(ls[0]['dist'], 2)} / {de(ls[1]['dist'], 2)} / {de(ls[2]['dist'], 2)} "
                    f"und Testfehler {err_pct(ls[0]['err'])} / {err_pct(ls[1]['err'])} / {err_pct(ls[2]['err'])} bei {ls[0]['size']} / {ls[1]['size']} / {ls[2]['size']} Signalen. LISTA mit {res['k']} Schichten "
                    f"erreicht {err_pct(lista['err'])}{' und schaltet alle Schwellen ab (linear)' if lista['th_zero'] else ''}, das Skalar-Netz {err_pct(scal['err'])}; FISTA mit 50 Schritten {err_pct(res['fista'])}.")


def _steps_text(v) -> str:
    return "mehr als 100 K" if v is None else str(v)


def verdict_layers(res: dict) -> tuple[str, str]:
    rows = res["rows"]
    r2, r10 = _by(rows, k=2), _by(rows, k=10)
    lr = res["learned"]
    return "warning", (f"In der Spitzen-Welt (Rauschen 1 %) erreicht LISTA mit {' / '.join(str(r['k']) for r in rows)} Schichten {' / '.join(err_pct(r['lista']) for r in rows)}, das Skalar-Netz "
                       f"{' / '.join(err_pct(r['scalar']) for r in rows)}; ISTA mit denselben Schichten {' / '.join(err_pct(r['ista']) for r in rows)}, FISTA {' / '.join(err_pct(r['fista']) for r in rows)}. "
                       f"Zehn Schichten LISTA entsprechen {_steps_text(r10['lista_equiv_fista'])} FISTA- oder {_steps_text(r10['lista_equiv_ista'])} ISTA-Schritten (Zwei Schichten: {_steps_text(r2['lista_equiv_fista'])} "
                       f"und {_steps_text(r2['lista_equiv_ista'])}). Das exakte L1 liegt bei {err_pct(res['l1'])}, Tikhonov bei {err_pct(res['tik'])}: das Netz holt die Iteration ein, nicht die Lösung. Das Skalar-Netz "
                       f"mit {lr['k']} Schichten lernt Schrittweiten von {de(min(lr['gam']), 1)} bis {de(max(lr['gam']), 1)} mal 1/‖A‖² (Summe {de(sum(lr['gam']), 1)} gegen {lr['k']} bei ISTA).")


def verdict_shift(res: dict) -> tuple[str, str]:
    sp = _by(res["rows"], kind="lista", train="spikes")["errs"]
    bl = _by(res["rows"], kind="lista", train="blocks")["errs"]
    al = _by(res["rows"], kind="lista", train="all")["errs"]
    sc = _by(res["rows"], kind="scalar", train="spikes")["errs"]
    nz = res["noise_rows"]
    return "warning", (f"LISTA mit {res['k']} Schichten, auf Spitzen trainiert: Spitzen {err_pct(sp['spikes'])}, Blöcke {err_pct(sp['blocks'])}, Wellen {err_pct(sp['waves'])}, Phantom {err_pct(sp['phantom'])}; "
                       f"auf Blöcken trainiert: Blöcke {err_pct(bl['blocks'])}, Spitzen {err_pct(bl['spikes'])}; auf allen vier gemischt: Spitzen {err_pct(al['spikes'])}, Blöcke {err_pct(al['blocks'])}. Das Skalar-Netz "
                       f"von den Spitzen erzeugt auf Blöcken {err_pct(sc['blocks'])}. Ohne Training: ISTA {err_pct(res['base']['ista']['blocks'])} und Tikhonov {err_pct(res['base']['tik']['blocks'])} auf Blöcken. "
                       f"Das Rauschniveau des Trainings spielt kaum eine Rolle: Test bei 1 % nach Training bei {noise_label(nz[0]['train'])} / {noise_label(nz[1]['train'])} / {noise_label(nz[2]['train'])}: "
                       f"{' / '.join(err_pct(r['errs'][1e-2]) for r in nz)}.")


def verdict_overfit(res: dict) -> tuple[str, str]:
    rows = [r for r in res["rows"] if r["kind"] == "lista"]
    sc = [r for r in res["rows"] if r["kind"] == "scalar"]
    return "warning", (f"LISTA ({rows[0]['params']} Parameter) mit {' / '.join(str(r['size']) for r in rows)} Trainingssignalen: Trainingsfehler {' / '.join(err_pct(r['train']) for r in rows)}, "
                       f"Testfehler {' / '.join(err_pct(r['test']) for r in rows)}. Das Skalar-Netz ({sc[0]['params']} Parameter) überanpasst nicht ({err_pct(sc[0]['train'])} / {err_pct(sc[0]['test'])} bei "
                       f"{sc[0]['size']} Signalen), lernt aber auch weniger (ISTA mit denselben Schichten: {err_pct(res['ista'])}).")
