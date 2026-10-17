# Entrollte Netze: ISTA mit gelernten Parametern – Streamlit-Demo

Vierzehntes Stück der Linie **Inverse Probleme und Regularisierung** in der „Konzepte“-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine Learning). Es baut auf [plug-and-play-demo](https://github.com/sebastian-hanisch/plug-and-play-demo), [sparsity-tv-demo](https://github.com/sebastian-hanisch/sparsity-tv-demo), [bayes-map-demo](https://github.com/sebastian-hanisch/bayes-map-demo) und [iterativ-demo](https://github.com/sebastian-hanisch/iterativ-demo) auf und nutzt dieselbe Welt: eine lineare Entfaltung (n = 48) mit Gauß-Unschärfe (w = 0,03) und weißem Rauschen. Die Wahrheit ist bekannt, weil die Signale selbst die Wahrheit sind; sie dient hier nur als Maßstab.

## Kernfrage

Was passiert, wenn man ISTA **entrollt** (K Schichten, jede mit eigenen Parametern) und **trainiert**? Was lernt das Training (das **Posterior-Mittel** der Trainingsverteilung, nicht das MAP), was bringt es gegen ISTA und FISTA (**Schichten statt Schritte**, aber nicht die exakte Lösung), und was kostet es (**Überanpassung**, **Verteilungswechsel**, keine konvergente Iteration mehr)?

## Modell

Generator `inverse_generator.py`, wortgleich in allen Stücken der Linie (Phantom, Gitter, Operator, Rauschen):

- **Daten:** b = A x + e mit der Gauß-Faltungsmatrix A (n = 48, w = 0,03), e ~ N(0, σ² I), σ je Signal = Rauschstufe mal Effektivwert der exakten Daten (0,1 / 1 / 10 %), Daten ohne Gitterfehler (das Modell stimmt).
- **Skalar-Netz:** x_{k+1} = soft(x_k − γ_k Aᵀ(A x_k − b), θ_k), 2 K Parameter. **LISTA:** x_{k+1} = soft(W_e b + S x_k, θ_k) mit gemeinsamen W_e, S, 2 n² + K Parameter (4618 bei K = 10). Start x_0 = 0, Start der Parameter = ISTA (γ = t, W_e = t Aᵀ, S = I − t AᵀA, θ = t λ mit dem besten λ auf den Trainingsdaten, t = 1/‖A‖²).
- **Training:** Verlust = mittlerer relativer quadratischer Fehler; die Gradienten werden **von Hand** durch die K Schichten zurückgerechnet (kein Autodiff; gegen den Differenzenquotienten geprüft), Adam mit Minibatches (256), Kosinus-Lernrate, 1000 Schritte; Daten und Reihenfolge der Minibatches aus festen SplitMix64-Strömen.
- **Trainingsverteilungen** (zufällige Signale): Spitzen (2 bis 4, Höhe 1 bis 2,5), Blöcke (2 bis 4, Breite 3 bis 10 Zellen), Dreieckswellen (3 bis 5 Perioden), Phantome (zwei Hügel und ein Kasten), alle vier gemischt, und Prior-Ziehungen (Stück 8, nur Experiment 1).
- **Vergleich:** ISTA und FISTA mit derselben Schrittzahl wie das Netz Schichten hat, exaktes L1 (Aktivmenge, Stück 6) und Tikhonov, jeweils mit dem besten λ des Gitters (Orakel) auf den Testsignalen.

Alles ist erfunden; die Wahrheit sind die simulierten Signale.

## Methodik

- **Verdikt:** „lohnt“, wenn der Testfehler höchstens 80 % des besseren von ISTA und FISTA mit derselben Schichtzahl beträgt; „kaum besser“ bis 100 %; sonst „lohnt nicht“. Dazu Hinweise: das exakte L1 ist besser (das Netz holt die Iteration ein, nicht die Lösung), Überanpassung (Trainingsfehler höchstens halb so groß wie der Testfehler), Verteilungswechsel, abgeschaltete Schwellen, gelernte Schrittweiten über 2/L.
- **Experimente** laufen über feste Zufallsströme (SplitMix64, Seeds ab 1000), **ein Training je Zelle**, je mit eigenem Rechenknopf (30 bis 60 Sekunden); ein einzelnes Netz („Netz trainieren“) braucht 2 bis 20 Sekunden.
- **Gleichstände und letzte Stellen** werden nicht festgelegt; Zahlen ohne Training (ISTA, FISTA, L1, Tikhonov, Posterior-Mittel, kleinste Quadrate) sind eng gepinnt, trainierte Zahlen mit weiter Toleranz und als Ungleichungen.

## Befunde (gemessen, keine Behauptungen)

Spitzen-Welt, Rauschen 1 %, 100 Testsignale (Median), wenn nicht anders genannt. Jede Zahl steht in `tests/test_claims.py` oder `tests/test_presets.py`.

| Frage | Befund | Test |
|---|---|---|
| Lernt das Netz das Posterior-Mittel (Experiment 1, Prior-Ziehungen, K = 5)? | **Ja, im Grenzwert, und nie darunter:** Posterior-Mittel **20,6 %** (Tikhonov 20,7 %, exaktes L1 66,0 %); die exakte Lösung der kleinsten Quadrate des linearen Netzes (K = 1, θ = 0) hat bei 100 / 1000 / 10000 Trainingssignalen den Matrixabstand 1,75 / 0,42 / 0,20 und den Testfehler 28,7 / 21,0 / **20,8 %**. LISTA mit 5 Schichten erreicht 24,0 % und **schaltet alle Schwellen ab** (linear), das Skalar-Netz 27,1 %. | `test_the_posterior_mean_is_the_floor_…`, `test_the_networks_cannot_beat_…` |
| Schichten statt Schritte (Experiment 2)? | **Ja:** LISTA mit 2 / 5 / 10 Schichten **66,9 / 27,6 / 16,2 %**, Skalar-Netz 64,2 / 52,8 / 45,7 %, ISTA 83,1 / 78,3 / 73,5 %, FISTA 83,1 / 76,3 / 67,0 %. Zehn Schichten LISTA entsprechen **50 FISTA- oder 300 ISTA-Schritten**, zwei Schichten 20 und 60. | `test_layers_beat_steps_but_not_the_exact_solution` |
| Holt das Netz die Lösung ein? | **Nein, nur die Iteration:** das exakte L1 hat **1,0 %**, Tikhonov 50,6 %. Das Orakel ist nicht die Obergrenze, es ist die Messlatte. | `test_layers_beat_steps_…` |
| Ist die gelernte Iteration noch ein Löser? | **Nein:** das Skalar-Netz (K = 10) lernt Schrittweiten von 0,4 bis 13,1 mal 1/‖A‖², Summe 46,8 gegen 10 bei ISTA; Schrittweiten über 2/L sind für ISTA nicht konvergent. | `test_the_scalar_network_learns_steps_far_above_two_over_l` |
| Gilt das Netz außerhalb seiner Trainingswelt (Experiment 3, K = 5)? | **Nein:** LISTA auf Spitzen trainiert: Spitzen **27,6 %**, Blöcke **50,7 %**, Wellen 44,6 %, Phantom 45,6 % (Tikhonov 50,6 / 17,9 / 9,4 / 9,9 %, ISTA 78,3 / 25,5 / 11,7 / 12,9 %); auf Blöcken trainiert: Blöcke 19,2 %, Spitzen 69,0 %; auf Wellen: Wellen **5,1 %** (Tikhonov 9,4 %); auf allen vier gemischt: Spitzen 59,8 %, Blöcke 19,2 %. Das **Skalar-Netz** von den Spitzen erzeugt auf Blöcken **122 %**. | `test_a_network_is_the_prior_of_its_training_distribution` |
| Muss das Trainingsrauschen passen? | **Kaum:** Test bei 1 % nach Training bei 0,1 / 1 / 10 %: 27,6 / 27,6 / 28,8 % (bei so wenigen Schichten begrenzt die Schichtzahl, nicht das Rauschen). | `test_the_training_noise_hardly_matters_at_few_layers` |
| Überanpassung (Experiment 4, K = 5)? | **LISTA (4613 Parameter)** mit 20 / 100 / 500 / 2000 Trainingssignalen: Trainingsfehler **0 % / 0,01 % / 19,1 % / 25,7 %**, Testfehler **106 % / 94,1 % / 44,2 % / 27,6 %**. Das **Skalar-Netz (10 Parameter)** überanpasst nicht (47,5 / 49,8 % bei 20 Signalen), lernt aber auch weniger (ISTA: 78,3 %). | `test_lista_memorizes_few_signals_…` |
| Einzelstichproben (Presets, K = 10, 1000 Schritte) | **Standard** (LISTA, Spitzen): Testfehler **20,0 %** gegen 73,4 % (ISTA) und 66,9 % (FISTA), exaktes L1 0,9 %. **Skalar-Netz:** 44,4 %, größte Schrittweite 14,9/L. **Verteilungswechsel** (Spitzen → Blöcke): 59,1 %, ISTA 22,2 %, Tikhonov 18,1 %. **20 Trainingssignale:** Trainingsfehler 0 %, Testfehler 92,8 %. **Wellen:** 4,9 % gegen 11,4 % (ISTA) und 10,0 % (Tikhonov). **Alle Welten:** 48,0 %. **Zwei Schichten:** 66,7 % gegen 83,1 %. | `tests/test_presets.py` |
| Mini-Instanz von Hand | n = 3, A symmetrisch (Eigenwerte 1 / 0,5 / 0,1), b = (1,3, 1,4, 1,3), x = (1, 2, 1): Skalar-Netz mit γ = 1, θ = 0,1 gibt (1,23, 1,24, 1,23), Verlust 0,1139, dL/dθ = 0,1, dL/dγ = −0,135533; LISTA dL/dW_e[0, 0] = 0,099667; ISTA nach zwei Schritten (1,2267, 1,2466, 1,2267). | `tests/test_model.py` |

## Befunde und Korrekturen gegenüber dem Plan

- **Das Orakel ist nicht die Obergrenze.** Die Gliederung nannte „ISTA mit voller Iteration“ als Orakel; gemessen liegt das exakte L1 (1,0 %) bei Spitzen unter jedem Netz mit wenigen Schichten (LISTA mit zehn Schichten 16,2 %). Das Netz holt die **Iteration** ein (50 FISTA-Schritte), nicht die **Lösung**; die zweite Messlatte ist das Posterior-Mittel (Stück 7).
- **„MMSE-artig“ ist exakt prüfbar, aber nur für die Trainingsverteilung:** Training mit quadratischem Verlust auf Prior-Ziehungen gibt das Posterior-Mittel (Test 20,8 gegen 20,6 %), und die Schwellen gehen auf null; bei Spitzen lernt das Netz einen nichtlinearen Schwellen-Schätzer.
- **Entrollt heißt schneller, nicht genauso gut und nicht mehr konvergent:** Faktor 5 gegen FISTA und 30 gegen ISTA bei zehn Schichten, aber nur auf der Trainingsverteilung, und mit Schrittweiten bis 13/L.
- **Rauschen im Training hat bei wenigen Schichten kaum Einfluss;** die Schichtzahl begrenzt die Genauigkeit.

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Das Netz löst das Problem** | Es holt die Iteration ein, nicht die Lösung: das exakte L1 ist bei Spitzen weit besser als jedes Netz mit wenigen Schichten. | Stück 16 (Verfahrensvergleich) |
| **Das Netz gilt überall** | Außerhalb der Trainingsverteilung verliert es gegen Tikhonov und ISTA (Experiment 3). | Stück 15 (Diffusions-Prior) |
| **Mehr Parameter helfen** | Bei wenigen Trainingssignalen lernt LISTA sie auswendig (Experiment 4). | – |
| **Es ist ein Löser** | Die gelernten Schrittweiten liegen weit über 2/L; die Iteration konvergiert nicht, sie ist auf K Schichten abgestimmt. | – |
| **Ein Training genügt** | Jede Zahl ist ein Training mit festem Seed; Mehrfachläufe, andere Optimierer und Gewichtsabfall sind nicht gebaut. | – |
| **Operator und Welten** | Eine Toeplitz-Faltung (w = 0,03, n = 48), von Hand gebaute Trainingsverteilungen, eine Dimension. | – |

Die Zahlen gelten für diese erfundene Welt und zeigen nichts über echte Messungen.

## Bezug zu den Nachbarn im Portfolio

- [plug-and-play-demo](https://github.com/sebastian-hanisch/plug-and-play-demo): Stück 13; dort der Denoiser als Prior mit Stärke und Schrittzahl von Hand, hier gelernt (das Skalar-Netz mit Schwellen je Schicht ist PnP mit trainierten Parametern).
- [sparsity-tv-demo](https://github.com/sebastian-hanisch/sparsity-tv-demo) und [iterativ-demo](https://github.com/sebastian-hanisch/iterativ-demo): Stück 6 und 5; ISTA, FISTA, das exakte L1 als Messlatte, die Schrittzahl als Regularisierung.
- [bayes-map-demo](https://github.com/sebastian-hanisch/bayes-map-demo): Stück 7; das Posterior-Mittel ist das Orakel des linearen Netzes.
- [mlp-backprop-demo](https://github.com/sebastian-hanisch/mlp-backprop-demo), [gradient-descent-demo](https://github.com/sebastian-hanisch/gradient-descent-demo): Backpropagation und Gradientenverfahren werden hier **benutzt**, nicht erklärt; neu ist nur die Rückwärtsrechnung durch K Schichten einer Iteration.

## Tests

Mini-Instanz von Hand (Weichschwelle, Schrittweite, Start gleich ISTA, Vorwärtsrechnung, Verlust und Gradienten für beide Netze, Spiegelsymmetrie, Zufallsströme und Signalverteilungen, ISTA und FISTA, Tikhonov, Lasso, kleinste Quadrate), Orakel (Rückwärtsrechnung gegen den Differenzenquotienten, Start gleich ISTA bei jeder Schichtzahl, ISTA gegen das exakte L1 auf einer gut gestellten Matrix, kleinste Quadrate gegen das Posterior-Mittel, Training senkt den Fehler und hält die Schwellen nicht negativ, Reproduzierbarkeit, Überanpassung, Verteilungswechsel, Bereiche der Verteilungen), Generator, Auswertung (jede Schwelle des Urteils auf der Kante), jede README-Zahl in `test_claims.py`, AppTest (Presets, Regler, Netze und Welten, Experimentknöpfe, PDF, Permalink), Footer, Presets und Laufzeit-Importe.

```bash
python -m pytest tests -q
```

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `inverse_generator.py` | gemeinsamer Generator der Linie (SplitMix64, Phantom, Gitter, Operator) |
| `ent_model.py` | Netze, Vorwärts- und Rückwärtsrechnung, Adam, Trainingsverteilungen, ISTA und FISTA, Tikhonov, L1-Aktivmenge, kleinste Quadrate |
| `ent_evaluation.py` | Training und Analyse, Experimente, Verdikte |
| `ent_visualization.py`, `ent_pdf_export.py` | Figuren, PDF |
| `ent_constants.py`, `ent_presets.py` | Regler, Presets, Permalink |
| `tests/` | Mini-Instanz, Orakel, Claims, App |

## Bewusst nicht umgesetzt

- **Autodiff-Rahmen und GPU:** die Rückwärtsrechnung von Hand ist der Lerninhalt.
- **Schicht-eigene Matrizen, Gewichtskopplung (ALISTA), tiefere Netze:** nur LISTA mit gemeinsamen Matrizen.
- **Mehrfachläufe, andere Optimierer, Gewichtsabfall, Frühabbruch auf Testdaten:** ein Training je Zelle.
- **Trainierte Denoiser, Rauschniveau pro Schicht, Scores:** Stück 15.
- **Ehrliche Daten, mehrere Operatoren, 2D-Bilder:** b = A x + e, ein Operator, eine Dimension.
- **Praxisdaten:** nicht gebaut; alle Zahlen sind erzeugt.

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python 3.12.

Gebaut mit Streamlit, Plotly, numpy, pandas und fpdf2.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
