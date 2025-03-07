# Radar FMCW

Ce repository contient une implémentation complète d'un système radar FMCW (Frequency Modulated Continuous Wave), incluant le traitement du signal, l'analyse des données, et la visualisation des résultats. Le code fourni permet de générer des cartes range-Doppler, d'estimer la distance et la vitesse des cibles, et d'analyser les signaux radar dans diverses conditions.

## Qu'est ce qu'un radar FMCW 

Le *radar FMCW* (Frequency-modulated Continuous Wave) est un type de radar qui émet continuellement des ondes électromagnétiques dont la fréquence varie linéairement avec le temps, généralement en forme de rampe (chirp). Contrairement aux radars à impulsions traditionnels, le radar FMCW émet un signal continu, ce qui lui confère plusieurs avantages:

### 1. Faible puissance d'émission requise

En effet, un signal qui se propage dans l'air, diminue d'un facteur inversément proportionnel au carré de la distance parcourue ($Puissance \propto \frac{1}{d^2}$). De plus, avec un radar à impulsions, on ne transmet que pendant une courte fraction de période ($DC \approx 10^{-1}$), pendant que le reste du temps est utilisé pour récupérer le signal à son retour. Donc si on regarde à la puissance moyenne envoyée par ce système sur toute la période, on est bien plus bas que si on émettait sur toute la période. 

$$P_{avg} = P_{peak} \times \frac{t_{on}}{t_{on}+t_{off}} = P_{peak} \times DC$$

Ce qui nous force à augmenter bien plus fort notre **$P_{peak}$** pour obtenir la même puissance qu'un système qui émet continuellement qui a donc un *DC = 100%*. Ce qui demande généralement des instruments, plus spacieux et plus chers.

### 2. Mesure simultanée de la distance et de la vitesse

Mais le fait d'émettre un signal continu n'est pas suffisant pour faire du radar un bon radar. En fait, en émettant qu'à une seule et unique fréquence, notre radar ne serait même pas capable de détecter la distance ne sachant pas différencier quel écho correspond au premier ou à un des nombreux autres. C'est là qu'intervient la modulation en fréquence *(Frequency-modulated)*. Ce radar FMCW envoie un signal tel que la fréquence augmente en dents de scie. Alors naturellement on remarque une différence de fréquence entre le signal reçu et le signal envoyé qui nous donne des informations sur [la distance et la vitesse](#comment-mesurent-ils-la-distance-et-la-vitesse-)

### 3. Meilleure résolution à courte portée

Une autre raison qui fait du radar FMCW préférable aux radars à impulsions classiques. Là où les radars à impulsions ont une distance minimum de détection dûe au fait que le radar ne détecte rien lors de l'émission (sinon il détecterait l'émission directe), le minimum est donc lié à la durée de l'impulsion $\tau$:

$$R_{min} = \frac{c \times \tau}{2}$$

Et plus $\tau$ est grand plus la distance minimale détectable est élevée, avec une impulsion $\tau = 1 \text{\mathrm{\mu} s}$ on a:

$$R_{\text{min}} = \frac{3 \times 10^8 \times 10^{-6}}{2} = 150 \text{ m}$$

Les radars FMCW, eux, ne doivent pas attendre la fin d'une impulsion car ils se basent sur les fréquences battement $f_{beat}$ ce qui en fait un outil très pratique pour les radars embarqués en voiture.

### 4. Moins sensible aux interférences

Contrairement à un radar à impulsions qui ne fait qu'écouter un écho sans vérifier sa cohérence spectrale. Le principe de base repose sur le mélange du signal émis avec le signal reçu après réflexion sur une cible. Un signal parasite n'aura pas la même structure de fréquence et sera facilement filtré. 

## En quoi sont-ils plus utiles que les autres ?

Les radars FMCW sont extrêmement utiles dans de nombreuses applications pour plusieurs raisons:

1. **Précision à courte et moyenne portée** - Idéal pour les applications automobiles, la surveillance du trafic et les systèmes anticollision.
2. **Capacité à détecter des objets stationnaires** - Contrairement à certains radars Doppler qui ne détectent que les objets en mouvement.
3. **Résolution en distance et en vitesse** - Peut mesurer simultanément la distance et la vitesse radiale des cibles.
4. **Faible coût et taille réduite** - Les composants modernes permettent de construire des systèmes radar FMCW compacts et abordables.
5. **Robustesse environnementale** - Fonctionne dans des conditions de mauvaise visibilité (pluie, brouillard, neige) où les capteurs optiques échouent. 

## Comment mesurent-ils la distance et la vitesse ?

Les radars FMCW mesurent la vitesse des objets grâce à l'effet Doppler et à des techniques de traitement du signal avancées:

1. **Effet Doppler** - Le mouvement relatif entre le radar et la cible provoque un décalage de fréquence du signal réfléchi. Par exemple, si une cible se dirige vers le radar, il rencontrera le signal en 'avance' et renverra le signal mais avec une fréquence plus élevée. 

![Image](https://github.com/user-attachments/assets/d012fec1-43dd-4665-b695-8eff8082e6c7)

Maintenant il faudrait trouver un moyen de calculer cette nouvelle fréquence, et on pourrait simplement avec l'équation de Doppler retrouver la vitesse de l'objet. 

$$ f_D = \frac{2 v f_0}{c} \Leftrightarrow v = \frac{c f_D}{2 f_0} $$

Pour trouver cette nouvelle fréquence qui peut être très proche de la fréquence du signal émis on va jouer avec la fréquence de battement $f_{beat}$. Pour trouver cette fréquence, on peut **"mixer"** le signal recu avec le signal envoyé en les multipliant. Ce qui nous donne une onde modulée. 

$$ mix = sin(f_0t)\times sin(f_Dt) $$

C'est là qu'il faut se rappeler des formules de Simpson et on remarque qu'en multipliant nos 2 signaux on a en fait un nouveau signal composé de 2 nouveaux sinus

$$ mix = \frac{1}{2}\times[cos((f_0-f_D)t)-cos((f_0+f_D)t)] $$ 

Et on remarque que notre fréquence de battement $f_{beat}$ apparaît qui en pratique est d'ordre du kHz et on peut donc facilement récupérer cette fréquence de la fréquence de $f_0+f_D$ qui est de l'ordre du GHz avec un filtre passe-bas. Ainsi on peut enfin déterminer la vitesse radiale (pas angulaire!). Mais de là, on trouve un nouveau problème la fréquence haute une fois filtrée on se retrouve avec $cos(f_{beat}t)$ où l'on ne peut déterminer si la fréquence est positive ou négative. L'astuce est de mélanger **en quadrature** c'est à dire de multiplier notre signal reçu avec un signal décalé de 90°

2. **Traitement par chirps multiples** - En émettant une succession de rampes de fréquence (chirps) et en analysant les variations de phase entre ces chirps, le radar peut déterminer la vitesse des cibles.
3. **Carte Range-Doppler** - En appliquant une double transformation de Fourier (2D-FFT) sur les données recueillies à partir de plusieurs chirps consécutifs:
    * La première FFT (sur chaque chirp) donne l'information de distance
    * La seconde FFT (à travers les chirps) révèle l'information de vitesse

## Structure du dépôt

radar-fmcw/         
├── src/                    # Définitions des fct utiles        
│   ├── signal_processing/  # Algorithmes de traitement du signal       
│   ├── visualization/      # Outils de visualisation       
│   └── simulation/         # Simulateurs de signaux radar              
├── examples/               # Application sur des fichiers de données       
├── data/                   # Données d'exemple et fichiers de configuration        
├── docs/                   # Documentation détaillée           
└── tests/                  # Tests unitaires  

## Installation 

Pour installer et configurer ce projet, suivez les étapes ci-dessous :

### Prérequis

- Python 3.6 ou supérieur
- pip (gestionnaire de paquets Python)
- make (facultatif, mais recommandé pour faciliter l'installation)

### Installation manuelle

1. Clonez le dépôt :
```bash
# Cloner le dépôt
git clone https://github.com/mat-bau/projet-radar-fmcw.git
cd projet-radar-fmcw
```
2. Créez un environnement virtuel (recommandé)

3. Installez les dépendances :
```bash
pip install -r requirements.txt
```
4. Installez le package en mode développement :
```bash
pip install -e .
```
### Installation avec Make
Si vous préférez utiliser Make, une seule commande suffit :

make setup

Cette commande va :
1. Créer un environnement virtuel Python
2. Installer toutes les dépendances requises
3. Installer le package en mode développement

### Vérification de l'installation

Pour vérifier que tout fonctionne correctement, runnez un exemple :

```bash
make run-MS1-FMCW
```

## Utilisation
Pour la suite des commandes vous pouvez consulter la documentation d'aide du Makefile pour les différentes commandes disponibles :
```bash
make help
```
