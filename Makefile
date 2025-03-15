# Makefile pour le projet Radar FMCW

# Variables
PYTHON = python3
PIP = pip3
DATA_DIR = data
OUTPUT_DIR = output
EXAMPLES_DIR = examples

LABO1_DIR = $(DATA_DIR)/Labo1
LABO2_DIR = $(DATA_DIR)/Labo2

# Exécutable principal
MAIN_SCRIPT = analyze_radar.py
ANIM_SCRIPT = animate_radar.py

# c'est une fonction pour trouver le fichier par son numéro dans data
find_file_by_number = $(shell ls -1 $(1)/$(2).* 2>/dev/null | head -1 || echo "")

# Cibles principales, mettre les toutes regles sinon ca bug
.PHONY: all setup run-% runanim-% runanim3d-% runanimcombined-% runcalibration-% clean help

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
run-labo1-%:
	@echo "Recherche du fichier correspondant au numéro $* dans $(LABO1_DIR)"
	$(eval FILE := $(call find_file_by_number,$(LABO1_DIR),$*))
	@if [ -z "$(FILE)" ]; then \
		echo "Erreur: Aucun fichier commençant par $*. n'a été trouvé dans $(LABO1_DIR)"; \
		exit 1; \
	fi
	@echo "Analyse du fichier $(FILE)"
	RADAR_DATA_FILE=$(FILE) $(PYTHON) $(EXAMPLES_DIR)/$(MAIN_SCRIPT) --output-dir=$(OUTPUT_DIR)/labo1/$*

run-labo2-%:
	@echo "Recherche du fichier correspondant au numéro $* dans $(LABO2_DIR)"
	$(eval FILE := $(call find_file_by_number,$(LABO2_DIR),$*))
	@if [ -z "$(FILE)" ]; then \
		echo "Erreur: Aucun fichier commençant par $*. n'a été trouvé dans $(LABO2_DIR)"; \
		exit 1; \
	fi
	@echo "Analyse du fichier $(FILE)"
	RADAR_DATA_FILE=$(FILE) $(PYTHON) $(EXAMPLES_DIR)/$(MAIN_SCRIPT) --output-dir=$(OUTPUT_DIR)/labo2/$* --frame 20 --estimate-iq-imbalance --correct-iq

# Animation 2D
runanim-labo1-%:
	@echo "Recherche du fichier correspondant au numéro $* dans $(LABO1_DIR)"
	$(eval FILE := $(call find_file_by_number,$(LABO1_DIR),$*))
	@if [ -z "$(FILE)" ]; then \
		echo "Erreur: Aucun fichier commençant par $*. n'a été trouvé dans $(LABO1_DIR)"; \
		exit 1; \
	fi
	@echo "Animation du fichier $(FILE)"
	@mkdir -p $(OUTPUT_DIR)/labo1/$*
	RADAR_DATA_FILE=$(FILE) $(PYTHON) $(EXAMPLES_DIR)/$(ANIM_SCRIPT) --output-dir=$(OUTPUT_DIR)/labo1/$* --view-type=2d

runanim-labo2-%:
	@echo "Recherche du fichier correspondant au numéro $* dans $(LABO2_DIR)"
	$(eval FILE := $(call find_file_by_number,$(LABO2_DIR),$*))
	@if [ -z "$(FILE)" ]; then \
		echo "Erreur: Aucun fichier commençant par $*. n'a été trouvé dans $(LABO2_DIR)"; \
		exit 1; \
	fi
	@echo "Animation du fichier $(FILE)"
	@mkdir -p $(OUTPUT_DIR)/labo2/$*
	RADAR_DATA_FILE=$(FILE) $(PYTHON) $(EXAMPLES_DIR)/$(ANIM_SCRIPT) --output-dir=$(OUTPUT_DIR)/labo2/$* --view-type=2d

# Animation 3D
runanim3d-labo1-%:
	@echo "Recherche du fichier correspondant au numéro $* dans $(LABO1_DIR)"
	$(eval FILE := $(call find_file_by_number,$(LABO1_DIR),$*))
	@if [ -z "$(FILE)" ]; then \
		echo "Erreur: Aucun fichier commençant par $*. n'a été trouvé dans $(LABO1_DIR)"; \
		exit 1; \
	fi
	@echo "Création d'une animation 3D pour le fichier $(FILE)"
	@mkdir -p $(OUTPUT_DIR)/labo1/$*
	RADAR_DATA_FILE=$(FILE) $(PYTHON) $(EXAMPLES_DIR)/$(ANIM_SCRIPT) --output-dir=$(OUTPUT_DIR)/labo1/$* --view-type=3d

runanim3d-labo2-%:
	@echo "Recherche du fichier correspondant au numéro $* dans $(LABO2_DIR)"
	$(eval FILE := $(call find_file_by_number,$(LABO2_DIR),$*))
	@if [ -z "$(FILE)" ]; then \
		echo "Erreur: Aucun fichier commençant par $*. n'a été trouvé dans $(LABO2_DIR)"; \
		exit 1; \
	fi
	@echo "Création d'une animation 3D pour le fichier $(FILE)"
	@mkdir -p $(OUTPUT_DIR)/labo2/$*
	RADAR_DATA_FILE=$(FILE) $(PYTHON) $(EXAMPLES_DIR)/$(ANIM_SCRIPT) --output-dir=$(OUTPUT_DIR)/labo2/$* --view-type=3d

# Animation combinée
runanimcombined-labo1-%:
	@echo "Recherche du fichier correspondant au numéro $* dans $(LABO1_DIR)"
	$(eval FILE := $(call find_file_by_number,$(LABO1_DIR),$*))
	@if [ -z "$(FILE)" ]; then \
		echo "Erreur: Aucun fichier commençant par $*. n'a été trouvé dans $(LABO1_DIR)"; \
		exit 1; \
	fi
	@echo "Création d'une animation combinée pour le fichier $(FILE)"
	@mkdir -p $(OUTPUT_DIR)/labo1/$*
	RADAR_DATA_FILE=$(FILE) $(PYTHON) $(EXAMPLES_DIR)/$(ANIM_SCRIPT) --output-dir=$(OUTPUT_DIR)/labo1/$* --fps=10 --remove-static --dynamic-range=20 --view-type=combined

runanimcombined-labo2-%:
	@echo "Recherche du fichier correspondant au numéro $* dans $(LABO2_DIR)"
	$(eval FILE := $(call find_file_by_number,$(LABO2_DIR),$*))
	@if [ -z "$(FILE)" ]; then \
		echo "Erreur: Aucun fichier commençant par $*. n'a été trouvé dans $(LABO2_DIR)"; \
		exit 1; \
	fi
	@echo "Création d'une animation combinée pour le fichier $(FILE)"
	@mkdir -p $(OUTPUT_DIR)/labo2/$*
	RADAR_DATA_FILE="$(FILE)" $(PYTHON) $(EXAMPLES_DIR)/$(ANIM_SCRIPT) --output-dir=$(OUTPUT_DIR)/labo2/$* --fps=10 --remove-static --dynamic-range=20 --view-type=combined

# Exécution de la calibration IQ
runanimiq-labo2-%:
	@echo "Recherche du fichier correspondant au numéro $* dans $(LABO2_DIR)"
	$(eval FILE := $(call find_file_by_number,$(LABO2_DIR),$*))
	@if [ -z "$(FILE)" ]; then \
		echo "Erreur: Aucun fichier commençant par $*. n'a été trouvé dans $(LABO2_DIR)"; \
		exit 1; \
	fi
	@echo "Création d'une animation avec correction IQ pour le fichier $(FILE)"
	@mkdir -p $(OUTPUT_DIR)/labo2/$*
	RADAR_DATA_FILE=$(FILE) $(PYTHON) $(EXAMPLES_DIR)/$(ANIM_SCRIPT) --output-dir=$(OUTPUT_DIR)/labo2/$* --view-type=2d --balance-iq

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
	@echo "  make                            - Installer les dépendances et exécuter l'analyse"
	@echo "  make setup                      - Installer les dépendances"
	@echo "  make clean                      - Nettoyer les fichiers temporaires"
	@echo "  make help                       - Afficher cette aide"
	@echo ""
	@echo "Commandes pour visualiser:"
	@echo "  make run-labo1-NUM              - Analyse le fichier NUM.* dans data/Labo1"
	@echo "  make run-labo2-NUM              - Analyse le fichier NUM.* dans data/Labo2 avec correction IQ"
	@echo "  make runanim-labo1-NUM          - Crée une animation 2D du fichier NUM.* dans data/Labo1"
	@echo "  make runanim-labo2-NUM          - Crée une animation 2D du fichier NUM.* dans data/Labo2"
	@echo "  make runanim3d-labo1-NUM        - Crée une animation 3D du fichier NUM.* dans data/Labo1"
	@echo "  make runanim3d-labo2-NUM        - Crée une animation 3D du fichier NUM.* dans data/Labo2"
	@echo "  make runanimcombined-labo1-NUM  - Crée une animation combinée du fichier NUM.* dans data/Labo1"
	@echo "  make runanimcombined-labo2-NUM  - Crée une animation combinée du fichier NUM.* dans data/Labo2"
	@echo "  make runanimiq-labo2-NUM        - Crée une animation 2D avec correction IQ pour le fichier NUM.* dans data/Labo2"
	@echo ""
	@echo "Exemples:"
	@echo "  make run-labo1-1                - Analyse le fichier qui commence par 1. dans data/Labo1"
	@echo "  make runanim-labo2-4            - Crée une animation 2D du fichier qui commence par 4. dans data/Labo2"
	@echo "  make runanimcombined-labo1-2    - Crée une animation combinée du fichier qui commence par 2. dans data/Labo1"
	@echo "  make runanimiq-labo2-5          - Crée une animation avec correction IQ pour le fichier qui commence par 5. dans data/Labo2"