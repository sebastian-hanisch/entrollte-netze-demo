"""Entrollte Netze: ISTA mit gelernten Parametern - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Vierzehntes Stück der Linie Inverse Probleme und Regularisierung. Auf der Welt aus Stück 1 bis 13 (lineare Entfaltung, Gauß-Unschärfe) wird ISTA mit K Schichten entrollt und trainiert:
Skalar-Netz (Schrittweite und Schwelle je Schicht) und LISTA (Matrizen W_e und S, Schwelle je Schicht). Gefragt wird, was das Training lernt, was es gegen ISTA und FISTA bringt und was es kostet.
Kern: ent_model.py (Vorwärts- und Rückwärtsrechnung von Hand, Adam), Generator: inverse_generator.py (wortgleich in allen Repos der Linie).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import ent_constants as C
from ent_evaluation import (
    Settings,
    analyse,
    de,
    err_pct,
    layers_experiment,
    noise_label,
    overfit_experiment,
    posterior_experiment,
    sci,
    shift_experiment,
    verdict_instance,
    verdict_layers,
    verdict_overfit,
    verdict_posterior,
    verdict_shift,
)
from ent_pdf_export import build_pdf
from ent_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from ent_visualization import (
    build_layers_chart,
    build_matrix_chart,
    build_noise_chart,
    build_overfit_chart,
    build_param_chart,
    build_posterior_chart,
    build_recon_chart,
    build_shift_chart,
    build_table,
    build_training_chart,
    style_table,
)

st.set_page_config(page_title="Entrollte Netze – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner="Netz trainiert (je nach Schichtzahl 2 bis 20 Sekunden) …")
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner="Experiment trainiert mehrere Netze über feste Zufallsströme (30 bis 60 Sekunden) …")
def _posterior():
    return posterior_experiment()


@st.cache_data(show_spinner="Experiment trainiert mehrere Netze über feste Zufallsströme (30 bis 60 Sekunden) …")
def _layers():
    return layers_experiment()


@st.cache_data(show_spinner="Experiment trainiert mehrere Netze über feste Zufallsströme (30 bis 60 Sekunden) …")
def _shift():
    return shift_experiment()


@st.cache_data(show_spinner="Experiment trainiert mehrere Netze über feste Zufallsströme (30 bis 60 Sekunden) …")
def _overfit():
    return overfit_experiment()


def verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


st.title("🧠 Entrollte Netze: ISTA mit gelernten Parametern")
st.markdown(
    """
**Ein entrolltes Netz ist ISTA mit gelernten Parametern und fester Schichtzahl: es gewinnt Schritte, aber nur für die Verteilung, auf der es trainiert wurde.** Dies ist das vierzehnte Stück der Linie
**Inverse Probleme und Regularisierung** und baut auf **Plug-and-Play**, **Sparsity und Total Variation** und **Bayes-Sicht: MAP** auf: dieselbe Welt (lineare Entfaltung, Gauß-Unschärfe), aber die
Schrittweiten, Schwellen und Matrizen von ISTA werden **trainiert**. Gefragt wird, **was das Training lernt** (das Posterior-Mittel der Trainingsverteilung, nicht das MAP), **was es bringt** (Schichten statt
Schritte, aber nicht die exakte Lösung) und **was es kostet** (Überanpassung, Verteilungswechsel, keine konvergente Iteration mehr). Die Welt ist simuliert; die Wahrheit ist bekannt und dient nur als Maßstab.

Weiter unten: vier Experimente (**🔬**, je mit eigenem Rechenknopf), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("Wie entrollte Netze funktionieren", expanded=True):
    st.markdown(
        r"""
1. **ISTA entrollen.** Eine Schicht ist ein ISTA-Schritt $x \leftarrow \operatorname{soft}\bigl(x - \gamma A^\top(Ax - b), \theta\bigr)$; $K$ Schichten sind $K$ Schritte, aber jede Schicht hat eigene Parameter.
2. **Zwei Netze.** Das **Skalar-Netz** lernt je Schicht Schrittweite $\gamma_k$ und Schwelle $\theta_k$ ($2K$ Parameter); **LISTA** lernt dazu die Matrizen $W_e$ und $S$ ($2n^2 + K$ Parameter): $x \leftarrow \operatorname{soft}(W_e b + S x, \theta_k)$.
3. **Training.** Zufällige Signale der Trainingswelt, Daten mit Rauschen, Verlust = mittlerer relativer quadratischer Fehler; die Gradienten werden **von Hand** durch die $K$ Schichten zurückgerechnet (geprüft gegen den Differenzenquotienten), Adam mit Minibatches. Start der Parameter ist ISTA.
4. **Was gelernt wird.** Der quadratische Verlust belohnt den bedingten Mittelwert: das Netz nähert das **Posterior-Mittel der Trainingsverteilung** an, nicht das MAP und nicht die L1-Lösung.
5. **Der Preis.** Gelernt gilt nur dort, wo trainiert wurde: Überanpassung bei wenigen Signalen, Verteilungswechsel bei anderer Welt, und die gelernte Iteration konvergiert nicht mehr als Löser.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden (danach „Netz trainieren“):")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Netz**")
    kind = st.selectbox("Welches Netz?", C.KINDS, key="kind_select", format_func=lambda v: C.KIND_LABELS[v],
                        help="Das Skalar-Netz lernt je Schicht zwei Zahlen, LISTA zusätzlich zwei Matrizen (4618 Parameter bei zehn Schichten).")
    k = st.select_slider("Schichten K", options=C.K_CHOICES, key="k_select", help="Zahl der entrollten ISTA-Schritte; mehr Schichten kosten mehr Trainingszeit (zehn Schichten rund 8 Sekunden).")
    st.markdown("**Welt**")
    train_world = st.selectbox("Trainingswelt", C.TRAIN_WORLDS, key="train_world_select", format_func=lambda v: C.WORLD_LABELS[v], help="Aus welcher Verteilung die Trainingssignale gezogen werden.")
    test_world = st.selectbox("Testwelt", C.TEST_WORLDS, key="test_world_select", format_func=lambda v: C.WORLD_LABELS[v],
                              help="Auf welchen neuen Signalen der Fehler gemessen wird; ungleich der Trainingswelt ist ein Verteilungswechsel.")
    noise = st.select_slider("Rauschen (relativ)", options=C.NOISE_CHOICES, key="noise_select", format_func=noise_label,
                             help="Standardabweichung des weißen Rauschens relativ zum Effektivwert der exakten Daten (Training und Test).")
    st.markdown("**Training**")
    n_train = st.select_slider("Trainingssignale", options=C.N_TRAIN_CHOICES, key="n_train_select", help="Wie viele Signale das Netz sieht; bei wenigen überanpasst LISTA.")
    steps = st.select_slider("Trainingsschritte", options=C.STEPS_CHOICES, key="steps_select", help="Schritte von Adam (Minibatch 256).")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Legt die Trainings- und Testsignale, das Rauschen und die Reihenfolge der Minibatches fest.")
    st.button("🎲 Neu würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"kind_select": kind, "k_select": int(k), "train_world_select": train_world, "test_world_select": test_world, "noise_select": float(noise), "n_train_select": int(n_train),
                   "steps_select": int(steps), "seed_input": int(seed)})

settings = Settings(kind, int(k), train_world, test_world, float(noise), int(n_train), int(steps), int(seed))

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
if st.button("🧠 Netz trainieren", key="train_button", help="Trainiert das Netz mit den gewählten Einstellungen (einmal je Einstellung, danach zwischengespeichert)."):
    st.session_state["trained"] = (settings, _analysis(settings))
trained = st.session_state.get("trained")
a = trained[1] if trained and trained[0] == settings else None

if a is None:
    st.info("Für diese Einstellungen ist noch kein Netz trainiert: „Netz trainieren“ drücken (zehn Schichten mit 1000 Schritten brauchen rund 8 Sekunden).")
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Testfehler des Netzes", err_pct(a.err_net), delta=f"Training {err_pct(a.err_train)}", delta_color="off", help="Median des relativen Fehlers auf 100 neuen Signalen der Testwelt; darunter der Fehler auf den Trainingssignalen.")
    c2.metric(f"ISTA, {settings.k} Schritte", err_pct(a.err_ista), delta=f"FISTA {err_pct(a.err_fista)}", delta_color="off",
              help="Dieselbe Zahl Schritte ohne Training, bestes λ (Orakel auf den Testsignalen); darunter FISTA.")
    c3.metric("L1 exakt", err_pct(a.err_l1), delta=f"Tikhonov {err_pct(a.err_tik)}", delta_color="off",
              help="Exakte L1-Lösung (Aktivmenge, Stück 6) beim besten λ, Median über 30 Testsignale: der Grenzwert von ISTA bei unendlich vielen Schritten; darunter Tikhonov.")
    c4.metric("Größte Schrittweite" if a.gam_max == a.gam_max else "Schwellen", f"{de(a.gam_max, 1)} / L" if a.gam_max == a.gam_max else ("abgeschaltet" if a.th_zero else "gelernt"),
              delta="ISTA: 1 / L, konvergent bis 2 / L" if a.gam_max == a.gam_max else f"Start-λ {sci(a.lam0)}", delta_color="off",
              help="Beim Skalar-Netz die größte gelernte Schrittweite im Vielfachen von 1/‖A‖²; ISTA konvergiert nur bis zum Doppelten. Bei LISTA, ob alle Schwellen abgeschaltet sind.")
    st.caption(f"Trainingsverteilung: {C.WORLD_LABELS[settings.train_world]}, {settings.n_train} Signale; Vergleichsverfahren mit dem besten λ (Orakel) auf den Testsignalen: ISTA λ = {sci(a.lam_ista)}, "
               f"FISTA λ = {sci(a.lam_fista)}, L1 λ = {sci(a.lam_l1)}.")

    tab_rec, tab_tr, tab_par, tab_mat, tab_tab = st.tabs(["Rekonstruktion", "Training", "Parameter", "Matrizen", "Tabelle"])
    with tab_rec:
        st.plotly_chart(build_recon_chart(a), width="stretch", key="recon_chart")
        st.caption("Das erste Testsignal: grün die Wahrheit, blau das Netz, grau gestrichelt ISTA, orange gestrichelt FISTA (beide mit derselben Schrittzahl wie das Netz Schichten hat), rot gepunktet das exakte L1.")
    with tab_tr:
        st.plotly_chart(build_training_chart(a), width="stretch", key="training_chart")
        st.caption("Median-Fehler auf den Trainings- und den Testsignalen über die Trainingsschritte (logarithmisch); die Linien sind ISTA, FISTA und das exakte L1. Klafft die Lücke zwischen Training und Test, überanpasst das Netz.")
    with tab_par:
        st.plotly_chart(build_param_chart(a), width="stretch", key="param_chart")
        st.caption("Gelernte Schwellen (und beim Skalar-Netz Schrittweiten) je Schicht, logarithmisch; gepunktet der Start (ISTA). Schrittweiten über 2/L sind für ISTA nicht mehr konvergent.")
    with tab_mat:
        fig = build_matrix_chart(a)
        if fig is None:
            st.info("Das Skalar-Netz hat keine Matrizen; die Matrizen W_e und S gibt es bei LISTA.")
        else:
            st.plotly_chart(fig, width="stretch", key="matrix_chart")
            st.caption("Gelernte Matrizen von LISTA (Start: W_e = t Aᵀ, S = I − t AᵀA). Rot positiv, blau negativ.")
    with tab_tab:
        st.dataframe(style_table(build_table(a)), hide_index=True, width="stretch", height=560)
        st.caption("Die ersten Punkte des ersten Testsignals.")

    if st.button("📄 PDF der aktuellen Instanz erstellen", key="pdf_make"):
        st.session_state["pdf_made"] = (settings, build_pdf(a))
    made = st.session_state.get("pdf_made")
    if made and made[0] == settings:
        st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="entrollte-netze.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Lohnt das Netz?")
if a is None:
    st.info("Das Urteil erscheint, sobald ein Netz mit diesen Einstellungen trainiert ist.")
else:
    verdict(verdict_instance(a))
st.caption("Das Urteil vergleicht den Testfehler mit dem besseren von ISTA und FISTA mit derselben Schichtzahl (lohnt: unter 80 %) und nennt, ob das exakte L1 besser ist (das Netz holt die Iteration ein, nicht die Lösung), "
           "ob das Netz überanpasst, ob die Welten sich unterscheiden, ob die Schwellen abgeschaltet sind und ob gelernte Schrittweiten über 2/L liegen.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------


def experiment(key, run, label):
    """Rechenknopf je Experiment: das Ergebnis liegt im Zwischenspeicher und bleibt in der Sitzung."""
    if st.button(label, key=f"{key}_run"):
        st.session_state[key] = run()
    return st.session_state.get(key)


st.subheader("🔬 Experiment 1: Was das Training lernt")
res1 = experiment("exp1", _posterior, "🧪 Experiment 1 rechnen (rund 20 Sekunden)")
st.caption("Prior-Ziehungen (dichte Signale, Stück 8) als Wahrheit, Rauschen 1 %: das Posterior-Mittel (Stück 7) ist die Untergrenze jedes Schätzers. Das lineare Netz (K = 1, θ = 0) hat eine exakte Lösung der kleinsten "
           "Quadrate; sie nähert sich dem Posterior-Mittel mit der Zahl der Trainingssignale.")
if res1 is None:
    st.info("Noch nicht gerechnet: der Knopf trainiert zwei Netze und löst die kleinsten Quadrate für drei Größen.")
else:
    st.plotly_chart(build_posterior_chart(res1), width="stretch", key="posterior_chart")
    verdict(verdict_posterior(res1))
    st.dataframe([{"Trainingssignale": r["size"], "Abstand der Matrix zum Posterior-Mittel": de(r["dist"], 3), "Testfehler": err_pct(r["err"])} for r in res1["ls"]], hide_index=True, width="stretch")
    st.dataframe([{"Netz": C.KIND_SHORT[r["kind"]], "Schichten": res1["k"], "Testfehler": err_pct(r["err"]), "alle Schwellen abgeschaltet": "ja" if r["th_zero"] else "nein",
                   "Schwellen θ_k": ", ".join(sci(v) for v in r["th"])} for r in res1["nets"]], hide_index=True, width="stretch")
    st.caption("Der quadratische Verlust lernt den bedingten Mittelwert: bei dichten Signalen schaltet LISTA die Schwellen ab und wird linear. Unter das Posterior-Mittel kommt kein Netz; das exakte L1 ist hier weit schlechter, "
               "weil die Welt nicht dünn besetzt ist.")

st.subheader("🔬 Experiment 2: Schichten statt Schritte")
res2 = experiment("exp2", _layers, "🧪 Experiment 2 rechnen (rund 50 Sekunden)")
st.caption("Spitzen-Welt, Rauschen 1 %: Netze mit 2, 5 und 10 Schichten gegen ISTA und FISTA mit derselben Zahl Schritte und die Schrittzahl, die ISTA und FISTA brauchen, um den Fehler des Netzes zu erreichen.")
if res2 is None:
    st.info("Noch nicht gerechnet: der Knopf trainiert sechs Netze.")
else:
    st.plotly_chart(build_layers_chart(res2), width="stretch", key="layers_chart")
    verdict(verdict_layers(res2))
    st.dataframe([{"Schichten K": r["k"], "LISTA": err_pct(r["lista"]), "Skalar-Netz": err_pct(r["scalar"]), "ISTA (K Schritte)": err_pct(r["ista"]), "FISTA (K Schritte)": err_pct(r["fista"]),
                   "LISTA entspricht FISTA-Schritten": "mehr als 100 K" if r["lista_equiv_fista"] is None else r["lista_equiv_fista"],
                   "LISTA entspricht ISTA-Schritten": "mehr als 100 K" if r["lista_equiv_ista"] is None else r["lista_equiv_ista"]} for r in res2["rows"]], hide_index=True, width="stretch")
    lr = res2["learned"]
    st.dataframe([{"Schicht": i + 1, "Schrittweite γ_k / t": de(g, 2), "Schwelle λ_k = θ_k / t": sci(lam)} for i, (g, lam) in enumerate(zip(lr["gam"], lr["lam"]))], hide_index=True, width="stretch")
    st.caption("Das Netz holt die Iteration ein, nicht die Lösung: das exakte L1 liegt weit darunter. Das Skalar-Netz lernt Schrittweiten weit über dem Doppelten von 1/L (zweite Tabelle): die gelernte Iteration ist kein "
               "konvergenter Löser mehr, sondern eine Folge von Schritten, die zusammen auf die Trainingsverteilung passt.")

st.subheader("🔬 Experiment 3: Verteilungs- und Rauschwechsel")
res3 = experiment("exp3", _shift, "🧪 Experiment 3 rechnen (rund 40 Sekunden)")
st.caption(f"LISTA mit {C.EXP_K} Schichten, auf je einer Welt (und auf allen gemischt) trainiert, auf jeder Welt getestet, dazu das Skalar-Netz von den Spitzen; unten das Trainingsrauschen.")
if res3 is None:
    st.info("Noch nicht gerechnet: der Knopf trainiert sieben Netze.")
else:
    cs1, cs2 = st.columns(2)
    with cs1:
        st.plotly_chart(build_shift_chart(res3), width="stretch", key="shift_chart")
    with cs2:
        st.plotly_chart(build_noise_chart(res3), width="stretch", key="noise_chart")
    verdict(verdict_shift(res3))
    st.dataframe([{"Netz": C.KIND_SHORT[r["kind"]], "Training auf": C.WORLD_SHORT[r["train"]], **{f"Test {C.WORLD_SHORT[w]}": err_pct(e) for w, e in r["errs"].items()}} for r in res3["rows"]] +
                 [{"Netz": "ISTA", "Training auf": "–", **{f"Test {C.WORLD_SHORT[w]}": err_pct(e) for w, e in res3["base"]["ista"].items()}},
                  {"Netz": "Tikhonov", "Training auf": "–", **{f"Test {C.WORLD_SHORT[w]}": err_pct(e) for w, e in res3["base"]["tik"].items()}}], hide_index=True, width="stretch")
    st.caption("Das Netz ist der Prior seiner Trainingsverteilung: auf der eigenen Welt gewinnt es, auf einer anderen verliert es gegen Tikhonov, und das Skalar-Netz kann dort sogar explodieren (Schrittweiten über 2/L). "
               "Ein Netz auf allen Welten ist ein Kompromiss. Das Rauschniveau des Trainings spielt bei so wenigen Schichten kaum eine Rolle: die Schichtzahl begrenzt, nicht das Rauschen.")

st.subheader("🔬 Experiment 4: Überanpassung")
res4 = experiment("exp4", _overfit, "🧪 Experiment 4 rechnen (rund 30 Sekunden)")
st.caption(f"{C.EXP_K} Schichten, Spitzen, Rauschen 1 %: Trainings- und Testfehler bei 20 bis 2000 Trainingssignalen; LISTA (viele Parameter) gegen das Skalar-Netz (wenige).")
if res4 is None:
    st.info("Noch nicht gerechnet: der Knopf trainiert sechs Netze.")
else:
    st.plotly_chart(build_overfit_chart(res4), width="stretch", key="overfit_chart")
    verdict(verdict_overfit(res4))
    st.dataframe([{"Netz": C.KIND_SHORT[r["kind"]], "Parameter": r["params"], "Trainingssignale": r["size"], "Trainingsfehler": err_pct(r["train"]), "Testfehler": err_pct(r["test"])} for r in res4["rows"]],
                 hide_index=True, width="stretch")
    st.caption("Mit wenigen Signalen lernt LISTA die Trainingssignale auswendig (Trainingsfehler null, Testfehler hoch). Das Skalar-Netz kann das nicht, weil es nur zwei Zahlen je Schicht hat, lernt aber auch weniger.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Das Netz löst das Problem** | Es holt die Iteration ein, nicht die Lösung: das exakte L1 ist bei Spitzen weit besser als jedes Netz mit wenigen Schichten. | Stück 16 (Verfahrensvergleich) |
| **Das Netz gilt überall** | Außerhalb der Trainingsverteilung verliert es gegen Tikhonov und ISTA (Experiment 3). | Stück 15 (Diffusions-Prior) |
| **Mehr Parameter helfen** | Bei wenigen Trainingssignalen lernt LISTA sie auswendig (Experiment 4). | – |
| **Es ist ein Löser** | Die gelernten Schrittweiten liegen weit über 2/L; die Iteration konvergiert nicht, sie ist auf K Schichten abgestimmt. | – |
| **Ein Training genügt** | Jede Zahl ist ein Training mit festem Seed; Mehrfachläufe, andere Optimierer und Gewichtsabfall sind nicht gebaut. | – |
| **Operator und Welten** | Eine Toeplitz-Faltung (w = 0,03, n = 48), von Hand gebaute Trainingsverteilungen, eine Dimension. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** $b = A x + e$, $e \sim \mathcal N(0, \sigma^2 I)$, $A$ die Gauß-Faltungsmatrix ($n = 48$, $w = 0{,}03$), $t = 1/\|A\|_2^2$.

**Netze.** Skalar-Netz: $x_{k+1} = \operatorname{soft}\bigl(x_k - \gamma_k A^\top(A x_k - b), \theta_k\bigr)$. LISTA: $x_{k+1} = \operatorname{soft}(W_e b + S x_k, \theta_k)$ mit gemeinsamen $W_e$, $S$. Start: $x_0 = 0$ und die Parameter von ISTA
($\gamma_k = t$, $W_e = tA^\top$, $S = I - tA^\top A$, $\theta_k = t\lambda$).

**Verlust.** $\mathcal L = \frac1m\sum_i \|x_{K,i} - x_i\|^2/\|x_i\|^2$. **Rückwärtsrechnung.** Mit $z_k$ dem Eingang der Weichschwelle und $d_k = g_k \odot \mathbf 1_{|z_k| > \theta_k}$ gilt
$\partial\mathcal L/\partial\theta_k = -\sum \operatorname{sign}(z_k)\odot d_k$; Skalar: $\partial\mathcal L/\partial\gamma_k = -\sum d_k \odot (A^\top(A x_k - b))$, $g_{k-1} = d_k - \gamma_k A^\top A\, d_k$; LISTA:
$\partial\mathcal L/\partial W_e = \sum d_k^\top b$ (über alle Schichten), $\partial\mathcal L/\partial S = \sum d_k^\top x_k$, $g_{k-1} = d_k S$.

**Posterior-Mittel.** Bei quadratischem Verlust ist der beste Schätzer der bedingte Mittelwert; für den Gauß-Prior $x \sim \mathcal N(0, P)$ ist das $P A^\top(A P A^\top + \sigma^2 I)^{-1} b$ (Stück 7), eine lineare Abbildung: das Netz mit
$K = 1$, $\theta = 0$ lernt $W_e$ durch kleinste Quadrate und trifft sie im Grenzwert vieler Trainingssignale.

**Literatur.** Das Beispiel ist erfunden. Entrollte Netze: K. Gregor und Y. LeCun (ICML 2010, LISTA); Konvergenz und Kopplung der Gewichte: X. Chen, J. Liu, Z. Wang, W. Yin (NeurIPS 2018); Überblick: V. Monga, Y. Li, Y. Eldar
(IEEE Signal Processing Magazine 2021); Adam: D. Kingma und J. Ba (2015) (alles aus dem Gedächtnis, nicht geprüft; die Zahlen hier sind gemessen, nicht der Literatur entnommen).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
