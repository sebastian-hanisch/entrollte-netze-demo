"""PDF-Export der aktuellen Instanz (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import ent_constants as C
from ent_evaluation import Analysis, de, err_pct, noise_label, sci

_REPLACE = {"→": "->", "–": "-", "—": "-", "σ": "sigma", "λ": "lambda", "θ": "theta", "γ": "gamma", "Γ": "Gamma", "×": "x", "τ": "tau", "−": "-", "€": "EUR", "≥": ">=", "·": "*", "‖": "|", "⁻": "-",
            "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}
MAX_ROWS = 40


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Entrollte Netze"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"{C.KIND_SHORT[s.kind]} mit {s.k} Schichten, Training auf {C.WORLD_SHORT[s.train_world]} ({s.n_train} Signale, {s.steps} Schritte), Test auf {C.WORLD_SHORT[s.test_world]}, "
                            f"Rauschen {noise_label(s.noise)} des Effektivwerts, Seed {s.seed}. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    lines = [
        f"Testfehler (Median) {err_pct(a.err_net)}, Trainingsfehler {err_pct(a.err_train)}.",
        f"ISTA mit {s.k} Schritten {err_pct(a.err_ista)} (lambda {sci(a.lam_ista)}), FISTA {err_pct(a.err_fista)} (lambda {sci(a.lam_fista)}), exaktes L1 {err_pct(a.err_l1)} (lambda {sci(a.lam_l1)}), "
        f"Tikhonov {err_pct(a.err_tik)}.",
        "Gelernte Schwellen lambda_k = theta_k / t: " + ", ".join(sci(v) for v in a.params["th"] / a.t) + ".",
    ]
    if "gam" in a.params:
        lines.append("Gelernte Schrittweiten gamma_k / t: " + ", ".join(de(v, 2) for v in a.params["gam"] / a.t) + " (ISTA: 1).")
    for line in lines:
        pdf.multi_cell(0, 5, _s(line), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    head, widths = ["Punkt", "Ort s", "Wahrheit x", "Netz", "ISTA", "FISTA", "L1 exakt"], [14, 22, 28, 28, 28, 28, 28]
    pdf.set_font("Helvetica", "B", 9)
    for h, w in zip(head, widths):
        pdf.cell(w, 6, _s(h), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for i in range(min(C.N, MAX_ROWS)):
        row = [f"{i + 1}", de(a.s[i], 3), de(a.x[i], 3), de(a.x_net[i], 3), de(a.x_ista[i], 3), de(a.x_fista[i], 3), de(a.x_l1[i], 3)]
        for v, w in zip(row, widths):
            pdf.cell(w, 5.5, _s(v), border=1)
        pdf.ln()
    if C.N > MAX_ROWS:
        pdf.ln(1)
        pdf.cell(0, 5, _s(f"... {C.N - MAX_ROWS} weitere Zeilen nicht gezeigt."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Entrollte Netze: ISTA mit gelernten Parametern und fester Schichtzahl. Quelle: entrollte-netze-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
