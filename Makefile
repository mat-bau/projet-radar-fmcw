# Makefile pour le projet Radar FMCW

# Variables
PYTHON = python3
PIP = pip3
DATA_DIR = data
OUTPUT_DIR = output
EXAMPLES_DIR = examples

# Cibles principales
.PHONY: all setup run-ms1 run-animation clean

# Cible par défaut
all: setup run-ms1

# Installation des dépendances
setup:
	$(PIP) install -r requirements.txt

# Vérification des dépendances
requirements.txt:
	@echo "numpy>=1.20.0" > requirements.txt
	@echo "matplotlib>=3.5.0" >> requirements.txt
	@echo "scipy>=1.7.0" >> requirements.txt

# Exécution du script d'analyse simple MS1-FMCW
# Exécution du script d'analyse simple MS1-FMCW
run-ms1:
	@mkdir -p $(OUTPUT_DIR)
	PYTHONPATH=. $(PYTHON) $(EXAMPLES_DIR)/ms1_fmcw.py --data-file $(DATA_DIR)/MS1-FMCW.npz --frame 0 --output-dir $(OUTPUT_DIR)

# Exécution du script d'animation
run-animation:
	@mkdir -p $(OUTPUT_DIR)
	$(PYTHON) $(EXAMPLES_DIR)/animate_frames.py --data-file $(DATA_DIR)/MS1-FMCW.npz --start-frame 0 --num-frames 6 --output-file $(OUTPUT_DIR)/radar_animation.mp4

# Exécution d'un script de comparaison des cartes Range-Doppler
run-comparison:
	@mkdir -p $(OUTPUT_DIR)
	$(PYTHON) src/comparaison_rdm.py

# Génération d'une carte Range-Doppler multi-frame
run-multiframe:
	@mkdir -p $(OUTPUT_DIR)
	$(PYTHON) src/multiframe_range-doppler-map.py

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
	@echo "  make run-ms1       - Exécuter l'analyse de base MS1-FMCW"
	@echo "  make run-animation - Créer une animation à partir de plusieurs frames"
	@echo "  make run-comparison - Exécuter la comparaison des cartes Range-Doppler"
	@echo "  make run-multiframe - Générer une carte Range-Doppler multi-frame"
	@echo "  make run-rdm       - Générer une carte Range-Doppler simple"
	@echo "  make clean         - Nettoyer les fichiers temporaires"
	@echo "  make help          - Afficher cette aide"