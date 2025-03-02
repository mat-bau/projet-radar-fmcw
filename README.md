# Radar FMCW

Ce repository contient une implémentation complète d'un système radar FMCW (Frequency Modulated Continuous Wave), incluant le traitement du signal, l'analyse des données, et la visualisation des résultats. Le code fourni permet de générer des cartes range-Doppler, d'estimer la distance et la vitesse des cibles, et d'analyser les signaux radar dans diverses conditions.

## Qu'est ce qu'un radar FMCW 

Le *radar FMCW* (Frequency-modulated Continuous Wave) est un type de radar qui émet continuellement des ondes électromagnétiques dont la fréquence varie linéairement avec le temps, généralement en forme de rampe (chirp). Contrairement aux radars à impulsions traditionnels, le radar FMCW émet un signal continu, ce qui lui confère plusieurs avantages:

### Faible puissance d'émission requise

En effet, un signal qui se propage dans l'air, diminue d'un facteur inversément proportionnel au carré de la distance parcourue ($Puissance \propto \frac{1}{d^2}$). Malheureusement, avec un radar à impulsions, on ne transmet que pendant une courte fraction de période ($DC \approx 10^{-1}$), pendant que le reste du temps est utilisé pour récupérer le signal à son retour. Donc si on regarde à la puissance moyenne envoyée par ce système sur toute la période, on est bien plus bas que si on émettait sur toute la période. 

$$P_{avg} = P_{peak} \times \frac{t_{on}}{t_{on}+t_{off}} = P_{peak} \times DC$$

Ce qui nous force à augmenter bien plus fort notre **$P_{peak}$** pour obtenir la même puissance qu'un système qui émet continuellement qui a donc un *DC = 100%*.

### Mesure simultanée de la distance et de la vitesse

Mais le fait d'émettre un signal continu n'est pas suffisant pour faire du radar un bon radar. En fait, en émettant qu'à une seule et unique fréquence, notre radar ne serait même pas capable de savoir quel écho correspond au premier ou au deuxième signal émis. C'est là qu'intervient la modulation en fréquence (Frequency-modulated). Ce radar FMCW envoie un signal tel 
* Meilleure résolution à courte portée
* Moins sensible aux interférences

Le principe de base repose sur le mélange du signal émis avec le signal reçu après réflexion sur une cible. La différence de fréquence entre ces deux signaux (fréquence de battement) est proportionnelle au temps de trajet aller-retour, donc à la distance de la cible.

## En quoi sont-ils plus utiles que les autres ?

Les radars FMCW sont extrêmement utiles dans de nombreuses applications pour plusieurs raisons:

1. **Précision à courte et moyenne portée** - Idéal pour les applications automobiles, la surveillance du trafic et les systèmes anticollision.
2. **Capacité à détecter des objets stationnaires** - Contrairement à certains radars Doppler qui ne détectent que les objets en mouvement.
3. **Résolution en distance et en vitesse** - Peut mesurer simultanément la distance et la vitesse radiale des cibles.
4. **Faible coût et taille réduite** - Les composants modernes permettent de construire des systèmes radar FMCW compacts et abordables.
5. **Robustesse environnementale** - Fonctionne dans des conditions de mauvaise visibilité (pluie, brouillard, neige) où les capteurs optiques échouent. 

## Comment mesurent-ils la vitesse ?

Les radars FMCW mesurent la vitesse des objets grâce à l'effet Doppler et à des techniques de traitement du signal avancées:

1. **Effet Doppler** - Le mouvement relatif entre le radar et la cible provoque un décalage de fréquence du signal réfléchi.
2. **Traitement par chirps multiples** - En émettant une succession de rampes de fréquence (chirps) et en analysant les variations de phase entre ces chirps, le radar peut déterminer la vitesse des cibles.
3. **Carte Range-Doppler** - En appliquant une double transformation de Fourier (2D-FFT) sur les données recueillies à partir de plusieurs chirps consécutifs:
    * La première FFT (sur chaque chirp) donne l'information de distance
    * La seconde FFT (à travers les chirps) révèle l'information de vitesse

## Structure du dépôt

radar-fmcw/
├── src/                    # Code source principal
│   ├── signal_processing/  # Algorithmes de traitement du signal
│   ├── visualization/      # Outils de visualisation
│   └── simulation/         # Simulateurs de signaux radar
├── examples/               # Exemples d'utilisation
├── data/                   # Données d'exemple et fichiers de configuration
├── docs/                   # Documentation détaillée
└── tests/                  # Tests unitaires et d'intégration

## Installation 

```bash
# Cloner le dépôt
git clone https://github.com/username/radar-fmcw.git
cd radar-fmcw

# Installer les dépendances
pip install -r requirements.txt

# Installation en mode développement
pip install -e .
```

