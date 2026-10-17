"""Konstanten: Regler, feste Annahmen, Training, Experimente, Farben und Presets."""

# Welt: die lineare Entfaltung aus Stück 1 bis 13 (n = 48 Gitterpunkte, Gauß-Unschärfe w = 0,03), Daten b = A x + e (das Modell stimmt).
N = 48
W = 0.03                                                         # Breite der Unschärfe (Anteil der Länge 1), fest
NOISE_CHOICES = (1e-3, 1e-2, 1e-1)                               # Rauschen relativ zum Effektivwert der exakten Daten
DEFAULT_NOISE = 1e-2
KINDS = ("scalar", "lista")
KIND_LABELS = {"scalar": "Skalar-Netz (Schrittweite und Schwelle je Schicht)", "lista": "LISTA (Matrizen W_e und S, Schwelle je Schicht)"}
KIND_SHORT = {"scalar": "Skalar-Netz", "lista": "LISTA"}
K_CHOICES = (2, 5, 10, 20)
DEFAULT_K = 10
TRAIN_WORLDS = ("spikes", "blocks", "waves", "phantom", "all")
TEST_WORLDS = ("spikes", "blocks", "waves", "phantom")
WORLD_LABELS = {"spikes": "zufällige Spitzen (2 bis 4)", "blocks": "zufällige Blöcke (2 bis 4)", "waves": "Dreieckswellen (3 bis 5 Perioden)", "phantom": "zufällige Phantome (zwei Hügel, ein Kasten)",
                "all": "alle vier gemischt", "gauss": "Prior-Ziehungen (Stück 8)"}
WORLD_SHORT = {"spikes": "Spitzen", "blocks": "Blöcke", "waves": "Wellen", "phantom": "Phantom", "all": "alle", "gauss": "Prior-Ziehungen"}
N_TRAIN_CHOICES = (20, 100, 500, 2000)
DEFAULT_N_TRAIN = 2000
STEPS_CHOICES = (300, 1000)
DEFAULT_STEPS = 1000
SEED_MAX, DEFAULT_SEED = 999999, 11

# Signalverteilungen: je Signal DRAW_WIDTH gleichverteilte Zahlen aus dem Strom
DRAW_WIDTH = 16
SPIKE_COUNT = (2, 3)                                             # 2 + Ganzzahlanteil von u * 3: 2 bis 4 Spitzen
SPIKE_HEIGHT = (1.0, 1.5)                                        # Höhe 1 + 1,5 u
BLOCK_COUNT = (2, 3)                                             # 2 bis 4 Blöcke
BLOCK_WIDTH = (3, 8)                                             # Breite 3 + Ganzzahlanteil von u * 8 (3 bis 10 Zellen)
BLOCK_HEIGHT = (0.3, 0.9)                                        # Höhe 0,3 + 0,9 u
WAVE_PERIODS = (3, 3)                                            # 3 bis 5 Perioden, zufällige Phase
WAVE_RANGE = (0.1, 0.9)
PHANTOM_CENTER, PHANTOM_WIDTH, PHANTOM_HEIGHT = (0.2, 0.6), (0.03, 0.03), (0.5, 0.5)   # Hügel: Mitte, Breite, Höhe = Anfang + Spanne * u
PHANTOM_BOX_LEFT, PHANTOM_BOX_HEIGHT, PHANTOM_BOX_WIDTH = (0.1, 0.7), (0.3, 0.4), 0.12

# Training: Adam mit Minibatches und Kosinus-Lernrate
PRIOR_TAU, PRIOR_ELL = 0.5, 0.12                                 # Gauß-Prior wie in Stück 8
BATCH = 256
LR = {"scalar": 0.1, "lista": 0.01}                              # Lernrate; die Matrizen von LISTA bekommen LR_MATRIX_FACTOR davon
LR_MATRIX_FACTOR = 0.1
COSINE_FLOOR = 0.05                                              # die Lernrate fällt kosinusförmig auf 5 %
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-12
LOG_POINTS = 10                                                  # Auswertungen des Verlaufs (Training und Test) je Training
TEST_COUNT = 100                                                 # Testsignale je Auswertung
BASE_COUNT = 30                                                  # Testsignale für das exakte L1 (teuer)
LAMBDA_GRID = tuple(10.0 ** (-6.0 + 0.5 * k) for k in range(13))  # λ-Gitter von ISTA, FISTA, L1 und Tikhonov (1e-6 bis 1)
LASSO_MAXIT = 5000
SEED_TRAIN, SEED_TRAIN_NOISE, SEED_TEST, SEED_TEST_NOISE = 1, 2, 3, 4   # Offsets zum Seed der Einstellung

# Verdikt-Schwellen
GAIN = 0.8                                                       # das Netz „lohnt“: Testfehler höchstens 80 % des besseren von ISTA und FISTA mit derselben Schichtzahl
OVERFIT = 0.5                                                    # Überanpassung: Trainingsfehler höchstens halb so groß wie der Testfehler
STEP_OVER = 2.0                                                  # gelernte Schrittweite über 2 / L: keine konvergente Iteration mehr
ZERO_TH = 1e-12                                                  # Schwelle gilt als abgeschaltet

# Experimente: feste Ströme (Seeds ab SEED_BASE), ein Training je Zelle.
SEED_BASE = 1000
EXP_STEPS = 1000
EXP_K = 5                                                        # Schichtzahl der Experimente 3 und 4 (kürzeres Training)
POST_K = 5                                                       # Experiment 1: Netze mit dieser Schichtzahl
POST_SIZES = (100, 1000, 10000)                                  # Experiment 1: Zahl der Trainingssignale der Lösung der kleinsten Quadrate
LAYER_KS = (2, 5, 10)                                            # Experiment 2
EQUIV_FACTORS = (1, 2, 5, 10, 30, 100)                           # Experiment 2: ISTA und FISTA mit Faktor mal K Schritten
SHIFT_WORLDS = ("spikes", "blocks", "waves", "all")              # Experiment 3: Trainingswelten (Testwelten: TEST_WORLDS)
SHIFT_NOISES = (1e-3, 1e-1)                                      # Experiment 3: Trainingsrauschen außer 1 %
OVERFIT_SIZES = (20, 100, 500, 2000)                             # Experiment 4

COLOR_TRUTH = "#2F6B65"
COLOR_NET = "#4c78a8"
COLOR_ISTA = "#8a8a8a"
COLOR_FISTA = "#D68A2E"
COLOR_L1 = "#c0392b"
COLOR_TIK = "#222222"
SHOW_RANGE = {"spikes": (-0.6, 3.0), "blocks": (-0.4, 1.9), "waves": (-0.4, 1.9), "phantom": (-0.4, 1.9)}   # feste Achse der Rekonstruktion je Testwelt

_BASE = {"kind": "lista", "k": DEFAULT_K, "train_world": "spikes", "test_world": "spikes", "noise": DEFAULT_NOISE, "n_train": DEFAULT_N_TRAIN, "steps": DEFAULT_STEPS, "seed": DEFAULT_SEED}
PRESETS = {
    "Standard": dict(_BASE),
    "Skalar-Netz": dict(_BASE, kind="scalar"),
    "Verteilungswechsel": dict(_BASE, test_world="blocks"),
    "Wenige Signale": dict(_BASE, n_train=20),
    "Wellen": dict(_BASE, train_world="waves", test_world="waves"),
    "Alle Welten": dict(_BASE, train_world="all"),
    "Wenige Schichten": dict(_BASE, k=2),
}
PRESET_HELP = {
    "Standard": "Zehn Schichten LISTA auf zufälligen Spitzen: 20,0 % Testfehler gegen 73,4 % (ISTA) und 66,9 % (FISTA) mit zehn Schritten; das exakte L1 hat 0,9 %.",
    "Skalar-Netz": "Das Skalar-Netz (20 Parameter) lernt weniger: 44,4 % Testfehler; die größte gelernte Schrittweite ist das 14,9-Fache von 1/‖A‖².",
    "Verteilungswechsel": "Auf Spitzen trainiert, auf Blöcken getestet: 59,1 % Testfehler, schlechter als ISTA (22,2 %) und Tikhonov (18,1 %).",
    "Wenige Signale": "20 Trainingssignale: Trainingsfehler 0 %, Testfehler 92,8 %: LISTA lernt die Signale auswendig.",
    "Wellen": "Auf Dreieckswellen trainiert und getestet: 4,9 % Testfehler gegen 11,4 % (ISTA) und 10,0 % (Tikhonov).",
    "Alle Welten": "Auf allen vier Welten trainiert, auf Spitzen getestet: 48,0 % statt 20,0 % beim Training auf Spitzen allein: ein Kompromiss.",
    "Wenige Schichten": "Zwei Schichten: 66,7 % gegen 83,1 % bei ISTA und FISTA mit zwei Schritten; die Schichtzahl begrenzt die Genauigkeit.",
}
