"""Plotly-Figuren und die Tabelle der Instanz. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import ent_constants as C
from ent_evaluation import Analysis, de, err_pct, noise_label, sci


def _fixed(fig: go.Figure, height: int = 360, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _ref_line(fig: go.Figure, value: float, line: dict, text: str, position: str, axis: str) -> None:
    """Referenzlinie mit Beschriftung auf einer logarithmischen Achse: die Linie liegt auf dem Datenwert, die Beschriftung braucht log10 des Werts
    (add_hline und add_vline setzen beide auf denselben Wert, die Beschriftung sitzt dann falsch und dehnt die Achse)."""
    if not value > 0.0:
        return
    pos = math.log10(value)
    top, left = "top" in position, "left" in position
    if axis == "y":
        fig.add_hline(y=value, line=line)
        fig.add_annotation(xref="paper", yref="y", x=0 if left else 1, y=pos, text=text, showarrow=False, xanchor="left" if left else "right", yanchor="bottom" if top else "top")
    else:
        fig.add_vline(x=value, line=line)
        fig.add_annotation(xref="x", yref="paper", x=pos, y=1 if top else 0, text=text, showarrow=False, xanchor="right" if left else "left", yanchor="top" if top else "bottom")


def build_table(a: Analysis, rows: int = 40) -> pd.DataFrame:
    k = min(rows, C.N)
    return pd.DataFrame({"Punkt": [f"{i + 1}" for i in range(k)], "Ort s": a.s[:k], "Daten b": a.b[:k], "Wahrheit x": a.x[:k], "Netz": a.x_net[:k], "ISTA": a.x_ista[:k], "FISTA": a.x_fista[:k],
                         "L1 exakt": a.x_l1[:k], "Tikhonov": a.x_tik[:k]})


def style_table(df: pd.DataFrame):
    return df.style.format({c: (lambda v: de(v, 4)) for c in df.columns if c != "Punkt"})


def build_recon_chart(a: Analysis) -> go.Figure:
    """Das erste Testsignal: Wahrheit, Netz, ISTA und FISTA mit denselben Schichten, exaktes L1; die Achse ist fest (je Testwelt)."""
    lo, hi = C.SHOW_RANGE[a.settings.test_world]
    k = a.settings.k
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a.s, y=a.x, mode="lines", name="Wahrheit", line=dict(color=C.COLOR_TRUTH, width=3), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a.s, y=a.x_l1, mode="lines", name="L1 exakt", line=dict(color=C.COLOR_L1, width=1.5, dash="dot"), hovertemplate="s %{x:.3f}<br>x̂ %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=a.s, y=a.x_ista, mode="lines", name=f"ISTA ({k} Schritte)", line=dict(color=C.COLOR_ISTA, width=1.5, dash="dash"), hovertemplate="s %{x:.3f}<br>x̂ %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=a.s, y=a.x_fista, mode="lines", name=f"FISTA ({k} Schritte)", line=dict(color=C.COLOR_FISTA, width=1.5, dash="dash"), hovertemplate="s %{x:.3f}<br>x̂ %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=a.s, y=a.x_net, mode="lines", name=f"Netz ({k} Schichten)", line=dict(color=C.COLOR_NET, width=2.5), hovertemplate="s %{x:.3f}<br>x̂ %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title="Ort s")
    fig.update_yaxes(title="Wert", range=[lo, hi])
    return _fixed(fig, 400)


def build_training_chart(a: Analysis) -> go.Figure:
    """Fehler auf den Trainings- und den Testsignalen über die Trainingsschritte, mit ISTA, FISTA und exaktem L1 als Linien."""
    steps = [h["step"] for h in a.hist]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=[h["train"] for h in a.hist], mode="lines+markers", name="Training", line=dict(color=C.COLOR_NET, width=3), marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=steps, y=[h["val"] for h in a.hist], mode="lines+markers", name="Test", line=dict(color=C.COLOR_L1, width=2, dash="dash"), marker=dict(size=6)))
    _ref_line(fig, a.err_ista, dict(color=C.COLOR_ISTA, width=1.5, dash="dash"), "ISTA (gleiche Schichten)", "top left", "y")
    _ref_line(fig, a.err_fista, dict(color=C.COLOR_FISTA, width=1.5, dash="dash"), "FISTA", "bottom left", "y")
    _ref_line(fig, a.err_l1, dict(color=C.COLOR_L1, width=1.5, dash="dot"), "L1 exakt", "bottom right", "y")
    fig.update_xaxes(title="Trainingsschritt")
    fig.update_yaxes(title="Median-Fehler", type="log", exponentformat="power")
    return _fixed(fig, 340)


def build_param_chart(a: Analysis) -> go.Figure:
    """Gelernte Schwellen λ_k = θ_k / t je Schicht (und beim Skalar-Netz die Schrittweiten γ_k / t); die Linien zeigen ISTA (γ / t = 1, λ konstant)."""
    p, t = a.params, a.t
    layers = list(range(1, len(p["th"]) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=layers, y=p["th"] / t, mode="lines+markers", name="Schwelle λ_k = θ_k / t", line=dict(color=C.COLOR_NET, width=3), marker=dict(size=7)))
    _ref_line(fig, a.lam0, dict(color=C.COLOR_NET, width=1.5, dash="dot"), "ISTA: λ", "top left", "y")
    if "gam" in p:
        fig.add_trace(go.Scatter(x=layers, y=p["gam"] / t, mode="lines+markers", name="Schrittweite γ_k / t", line=dict(color=C.COLOR_L1, width=3), marker=dict(size=7)))
        _ref_line(fig, 1.0, dict(color=C.COLOR_L1, width=1.5, dash="dot"), "ISTA: 1", "bottom left", "y")
        _ref_line(fig, C.STEP_OVER, dict(color=C.COLOR_ISTA, width=1, dash="dash"), "2 / L", "top right", "y")
    fig.update_xaxes(title="Schicht", tickvals=layers)
    fig.update_yaxes(title="Wert", type="log", exponentformat="power")
    return _fixed(fig, 340)


def build_matrix_chart(a: Analysis) -> go.Figure | None:
    """Bilder der gelernten Matrizen W_e und S (LISTA); beim Skalar-Netz gibt es keine."""
    if "We" not in a.params:
        return None
    fig = make_subplots(rows=1, cols=2, subplot_titles=("W_e (Eingang b)", "S (Rückführung von x)"))
    for col, key in ((1, "We"), (2, "Sm")):
        z = a.params[key]
        lim = float(np.abs(z).max())
        fig.add_trace(go.Heatmap(z=z, zmin=-lim, zmax=lim, colorscale="RdBu", reversescale=True, showscale=(col == 2), hovertemplate="Zeile %{y}, Spalte %{x}<br>%{z:.3f}<extra></extra>"), row=1, col=col)
    fig.update_yaxes(autorange="reversed")
    return _fixed(fig, 340, showlegend=False)


# ----------------------------------------------------------------------------- Experimente

def build_posterior_chart(res: dict) -> go.Figure:
    """Fehler bei Prior-Ziehungen: Posterior-Mittel (Untergrenze), kleinste Quadrate (K = 1, θ = 0, größte Zahl Trainingssignale), die Netze, FISTA, Tikhonov und exaktes L1."""
    lista = next(r for r in res["nets"] if r["kind"] == "lista")
    scal = next(r for r in res["nets"] if r["kind"] == "scalar")
    labels = ["Posterior-Mittel", f"kleinste Quadrate ({res['ls'][-1]['size']} Signale)", f"LISTA ({res['k']} Schichten)", f"Skalar-Netz ({res['k']})", "FISTA (50)", "Tikhonov", "L1 exakt"]
    vals = [res["pm"], res["ls"][-1]["err"], lista["err"], scal["err"], res["fista"], res["tik"], res["l1"]]
    fig = go.Figure(go.Bar(x=labels, y=[100 * v for v in vals], marker_color=["#222222", "#2F6B65", C.COLOR_NET, "#8fb3d9", C.COLOR_FISTA, C.COLOR_ISTA, C.COLOR_L1], text=[err_pct(v) for v in vals], textposition="outside"))
    fig.update_xaxes(title="Prior-Ziehungen als Wahrheit, Rauschen 1 %", type="category")
    fig.update_yaxes(title="Median-Fehler (%)", range=[0, 75])
    return _fixed(fig, 360, showlegend=False)


def build_layers_chart(res: dict) -> go.Figure:
    """Fehler über die Schichtzahl: LISTA, Skalar-Netz, ISTA und FISTA mit derselben Zahl Schritte; die Linie ist das exakte L1."""
    ks = [r["k"] for r in res["rows"]]
    fig = go.Figure()
    for key, name, color, dash in (("lista", "LISTA", C.COLOR_NET, "solid"), ("scalar", "Skalar-Netz", "#8fb3d9", "solid"), ("ista", "ISTA", C.COLOR_ISTA, "dash"), ("fista", "FISTA", C.COLOR_FISTA, "dash")):
        fig.add_trace(go.Scatter(x=ks, y=[100 * r[key] for r in res["rows"]], mode="lines+markers", name=name, line=dict(color=color, width=3, dash=dash), marker=dict(size=8)))
    fig.add_hline(y=100 * res["l1"], line=dict(color=C.COLOR_L1, width=1.5, dash="dot"), annotation_text="L1 exakt", annotation_position="top right")
    fig.update_xaxes(title="Schichten beziehungsweise Schritte K", tickvals=ks, type="category")
    fig.update_yaxes(title="Median-Fehler (%), Spitzen, Rauschen 1 %", range=[0, 90])
    return _fixed(fig, 340)


def build_shift_chart(res: dict) -> go.Figure:
    """Fehler je Testwelt: LISTA nach Trainingswelt, dazu ISTA und Tikhonov (bestes λ je Testwelt)."""
    worlds = list(C.TEST_WORLDS)
    labels = [C.WORLD_SHORT[w] for w in worlds]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[100 * res["base"]["tik"][w] for w in worlds], name="Tikhonov", marker_color=C.COLOR_TIK))
    fig.add_trace(go.Bar(x=labels, y=[100 * res["base"]["ista"][w] for w in worlds], name="ISTA (gleiche Schichten)", marker_color=C.COLOR_ISTA))
    for r, color in zip([r for r in res["rows"] if r["kind"] == "lista"], ("#1f4e79", "#4c78a8", "#8fb3d9", "#c6dbef")):
        fig.add_trace(go.Bar(x=labels, y=[100 * r["errs"][w] for w in worlds], name=f"LISTA, Training auf {C.WORLD_SHORT[r['train']]}", marker_color=color))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Testwelt", type="category")
    fig.update_yaxes(title="Median-Fehler (%)", range=[0, 100])
    return _fixed(fig, 380)


def build_noise_chart(res: dict) -> go.Figure:
    """Fehler auf Spitzen nach Trainingsrauschen und Testrauschen."""
    tests = sorted(res["noise_rows"][0]["errs"])
    fig = go.Figure()
    for r, color in zip(res["noise_rows"], ("#8fb3d9", "#4c78a8", "#1f4e79")):
        fig.add_trace(go.Bar(x=[noise_label(t) for t in tests], y=[100 * r["errs"][t] for t in tests], name=f"Training bei {noise_label(r['train'])}", marker_color=color))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Rauschen im Test", type="category")
    fig.update_yaxes(title="Median-Fehler (%)", range=[0, 40])
    return _fixed(fig, 340)


def build_overfit_chart(res: dict) -> go.Figure:
    """Trainings- und Testfehler über die Zahl der Trainingssignale: LISTA (Linien) und Skalar-Netz (Punkte)."""
    lista = [r for r in res["rows"] if r["kind"] == "lista"]
    scal = [r for r in res["rows"] if r["kind"] == "scalar"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r["size"] for r in lista], y=[100 * r["train"] for r in lista], mode="lines+markers", name="LISTA, Training", line=dict(color=C.COLOR_NET, width=3), marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=[r["size"] for r in lista], y=[100 * r["test"] for r in lista], mode="lines+markers", name="LISTA, Test", line=dict(color=C.COLOR_L1, width=3, dash="dash"), marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=[r["size"] for r in scal], y=[100 * r["train"] for r in scal], mode="markers", name="Skalar-Netz, Training", marker=dict(size=11, color=C.COLOR_NET, symbol="diamond-open")))
    fig.add_trace(go.Scatter(x=[r["size"] for r in scal], y=[100 * r["test"] for r in scal], mode="markers", name="Skalar-Netz, Test", marker=dict(size=11, color=C.COLOR_L1, symbol="x")))
    fig.update_xaxes(title="Zahl der Trainingssignale", type="log", tickvals=[r["size"] for r in lista])
    fig.update_yaxes(title="Median-Fehler (%), Spitzen, Rauschen 1 %", range=[0, 100])
    return _fixed(fig, 340)
