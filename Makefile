# Makefile pour le projet Radar FMCW

# Variables
PYTHON = python3
PIP = pip3
DATA_DIR = data
OUTPUT_DIR = output
EXAMPLES_DIR = examples

# Exécutable principal
MAIN_SCRIPT = analyze_radar.py
ANIM_SCRIPT = animate_radar.py
# Cibles principales
.PHONY: all setup run-ms1 run-animation clean

# Cible par défaut
all: setup

# Installation des dépendances
setup:
	$(PIP) install -r requirements.txt

# Vérification des dépendances
requirements.txt:
	@echo "numpy>=1.20.0" > requirements.txt
	@echo "matplotlib>=3.5.0" >> requirements.txt
	@echo "scipy>=1.7.0" >> requirements.txt
	@echo "pandas>=1.3.0" >> requirements.txt
	@echo "pytest>=7.0.0" >> requirements.txt

# Exécution du script d'analyse simple
run-%:
	@echo "Analyse du fichier $(DATA_DIR)/$*.npz"
	@if [ ! -f "$(DATA_DIR)/$*.npz" ]; then \
		echo "Erreur: Le fichier $(DATA_DIR)/$*.npz n'existe pas"; \
		exit 1; \
	fi
	RADAR_DATA_FILE=$(DATA_DIR)/$*.npz $(PYTHON) $(EXAMPLES_DIR)/$(MAIN_SCRIPT)

# Animation 2D
runanim-%:
	@echo "Création d'une animation 2D pour le fichier $(DATA_DIR)/$*.npz"
	@if [ ! -f "$(DATA_DIR)/$*.npz" ]; then \
		echo "Erreur: Le fichier $(DATA_DIR)/$*.npz n'existe pas"; \
		exit 1; \
	fi
	@mkdir -p output/$*
	RADAR_DATA_FILE=$(DATA_DIR)/$*.npz $(PYTHON) $(EXAMPLES_DIR)/$(ANIM_SCRIPT) --output-dir=output/$* --view-type=2d --background-file $(DATA_DIR)/background1.npz

# Animation 3D
runanim3d-%:
	@echo "Création d'une animation 3D pour le fichier $(DATA_DIR)/$*.npz"
	@if [ ! -f "$(DATA_DIR)/$*.npz" ]; then \
		echo "Erreur: Le fichier $(DATA_DIR)/$*.npz n'existe pas"; \
		exit 1; \
	fi
	@mkdir -p output/$*
	RADAR_DATA_FILE=$(DATA_DIR)/$*.npz $(PYTHON) $(EXAMPLES_DIR)/$(ANIM_SCRIPT) --output-dir=output/$* --view-type=3d

runanimcombined-%:
	@echo "Création d'une animation combinée pour le fichier data/$*.npz"
	RADAR_DATA_FILE=data/$*.npz python3 $(EXAMPLES_DIR)/$(ANIM_SCRIPT) --output-dir=output/$* --fps=10 --dynamic-range=20 --detect-targets --view-type=combined --background-file $(DATA_DIR)/background1.npz

# Nettoyage
clean:
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf src/*/__pycache__
	rm -rf $(EXAMPLES_DIR)/__pycache__
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "*.pyd" -delete

# Aide
.PHONY: help
help:
	@echo "Commandes disponibles :"
	@echo "  make               - Installer les dépendances et exécuter l'analyse MS1-FMCW"
	@echo "  make setup         - Installer les dépendances"
	@echo "  make clean         - Nettoyer les fichiers temporaires"
	@echo "  make help          - Afficher cette aide"
	@echo "  make run-FICHIER      - Exécute l'analyse sur FICHIER.npz (dans le répertoire $(DATA_DIR))"
	@echo "  make help             - Affiche cette aide"
	@echo "  make clean            - Nettoie les fichiers générés"
	@echo ""
	@echo "Exemples:"
	@echo "  make run-MS1-FMCW     - Analyse le fichier data/MS1-FMCW.npz"
	@echo "  make run-anim-MS1-FMCW    - Crée une animation 2D du fichier data/MS1-FMCW.npz"
	@echo "  make run-anim-3d-MS1-FMCW - Crée une animation 3D du fichier data/MS1-FMCW.npz"
