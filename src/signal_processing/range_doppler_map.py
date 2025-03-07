import numpy as np
from scipy import signal
from scipy import ndimage

def calculate_range_axis(params, padded_samples=None):
    """
    Calcule l'axe de distance pour la carte Range-Doppler avec prise en compte du zero-padding
    
    Parameters:
    -----------
    params : dict
        Dictionnaire contenant les paramètres du radar
        
    padded_samples : int, optional
        Nombre d'échantillons après zero-padding
        
    Returns:
    --------
    range_axis : ndarray
        Axe de distance en mètres
    """
    # Vérifier que les paramètres nécessaires sont présents
    if 'samples_per_chirp' not in params or 'range_resolution' not in params:
        raise KeyError("Paramètres manquants: 'samples_per_chirp' ou 'range_resolution'")
    
    Ms = int(params['samples_per_chirp'])  # Convertir en entier pour éviter les problèmes de type
    range_resolution = params['range_resolution']
    
    # Si padded_samples n'est pas spécifié, utiliser Ms
    if padded_samples is None:
        padded_samples = Ms
    
    # Calcul de la nouvelle résolution avec zero-padding
    actual_resolution = range_resolution * (Ms / padded_samples)

    # Créer l'axe de distance
    range_axis = np.arange(padded_samples) * actual_resolution
    
    return range_axis

def calculate_velocity_axis(params, padded_chirps=None):
    """
    Calcule l'axe de vitesse pour la carte Range-Doppler avec prise en compte du zero-padding
    
    Parameters:
    -----------
    params : dict
        Dictionnaire contenant les paramètres du radar
        
    padded_chirps : int, optional
        Nombre de chirps après zero-padding
        
    Returns:
    --------
    velocity_axis : ndarray
        Axe de vitesse en m/s
    """
    # Vérifier que les paramètres nécessaires sont présents
    required_params = ['start_freq', 'num_chirps', 'chirp_time']
    for param in required_params:
        if param not in params:
            raise KeyError(f"Paramètre manquant: '{param}'")
    
    f0 = params['start_freq']
    Mc = int(params['num_chirps'])
    Tc = params['chirp_time']
    
    # Si padded_chirps n'est pas spécifié, utiliser Mc
    if padded_chirps is None:
        padded_chirps = Mc
    
    # Calcul de la vitesse maximale sans ambiguïté
    wavelength = 3e8 / f0  # Longueur d'onde (c/f)
    velocity_max = wavelength / (4 * Tc)  # Vitesse maximale sans ambiguïté
    
    # Calcul de la résolution de vitesse avec zero-padding
    velocity_resolution = (2 * velocity_max) / padded_chirps
    
    # Axe de vitesse centré sur zéro
    velocity_axis = np.arange(-padded_chirps//2, padded_chirps//2) * velocity_resolution
    
    return velocity_axis

def apply_window(data, window_type='hann'):
    """
    Applique une fenêtre aux données pour réduire les lobes secondaires
    
    Parameters:
    -----------
    data : ndarray
        Données de forme (num_shirps, samples_per_chirp)

    window_type : str
        Type de fenêtre à appliquer ('hann', 'hamming', 'blackman', etc.)
        
    Returns:
    --------
    windowed_data : ndarray
        Données avec fenêtre appliquée
    """
    Mc, Ms = data.shape
    
    try:
        # Créer les fenêtres
        range_window = getattr(signal.windows, window_type)(Ms)
        doppler_window = getattr(signal.windows, window_type)(Mc)
    except AttributeError:
        # Fallback si le type de fenêtre n'existe pas
        print(f"Type de fenêtre '{window_type}' non disponible, utilisation de 'hann' à la place.")
        range_window = signal.windows.hann(Ms)
        doppler_window = signal.windows.hann(Mc)
    
    # Application efficace des fenêtres (broadcasting NumPy)
    windowed_data = data * range_window[np.newaxis, :] * doppler_window[:, np.newaxis]
    
    return windowed_data

def remove_mean_per_row(data):
    """
    Supprime la moyenne temporelle de chaque ligne (chirp) dans les données radar
    
    Parameters:
    -----------
    data : ndarray
        Données de forme (num_chirps, samples_per_chirp)
        
    Returns:
    --------
    processed_data : ndarray
        Données avec la moyenne de chaque ligne soustraite
    """
    # la moyenne de chaque ligne
    row_means = np.mean(data, axis=1, keepdims=True)
    
    processed_data = data - row_means
    
    return processed_data

def generate_range_profile(data, window_type='hann'):
    """
    Génère un profil de distance à partir des données radar
    
    Parameters:
    -----------
    data : ndarray
        Données de forme (num_chirps, samples_per_chirp)

    window_type : str
        Type de fenêtre à appliquer
        
    Returns:
    --------
    range_profile : ndarray
        Profil de distance moyenné sur tous les chirps (en dB)
    """
    Mc, Ms = data.shape
    
    try:
        # Fenêtre en distance uniquement
        range_window = getattr(signal.windows, window_type)(Ms)
    except AttributeError:
        # Fallback si le type de fenêtre n'existe pas
        print(f"Type de fenêtre '{window_type}' non disponible, utilisation de 'hann' à la place.")
        range_window = signal.windows.hann(Ms)
    
    # Application de la fenêtre
    windowed_data = data * range_window[np.newaxis, :]
    
    # FFT sur chaque chirp (dimension distance)
    range_fft = np.fft.fft(windowed_data, axis=1)
    
    # Moyenner sur tous les chirps
    range_profile = np.mean(np.abs(range_fft), axis=0)
    
    # Conversion en dB avec protection contre les valeurs nulles
    range_profile_db = 20 * np.log10(np.maximum(range_profile, 1e-15))
    
    return range_profile_db

def generate_range_doppler_map_with_axes(data, params, window_type='hann', 
                                        range_padding_factor=2, doppler_padding_factor=4,
                                        apply_2d_filter=False, kernel_size=3):
    """
    Génère une Range-Doppler map à partir des données radar avec zero-padding
    et retourne également les axes physiques
    
    Parameters:
    -----------
    data : ndarray
        Données de forme (Mc, samples_per_chirp)
        
    params : dict
        Dictionnaire contenant les paramètres du radar

    window_type : str
        Type de fenêtre à appliquer ('hann', 'hamming', 'blackman', etc.)
        
    range_padding_factor : int
        Facteur de zero-padding pour l'axe distance (1 = pas de padding)
        
    doppler_padding_factor : int
        Facteur de zero-padding pour l'axe Doppler (1 = pas de padding)
        
    apply_2d_filter : bool
        Appliquer un filtre médian 2D pour réduire le bruit
        
    kernel_size : int
        Taille du noyau pour le filtre médian (si apply_2d_filter=True)
        
    Returns:
    --------
    range_doppler_map : ndarray
        Carte Range-Doppler en 2D (en dB)
        
    range_axis : ndarray
        Axe de distance en mètres
        
    velocity_axis : ndarray
        Axe de vitesse en m/s
    """
    Mc, Ms = data.shape
    
    # Nouvelles dimensions avec zero-padding
    padded_Ms = Ms * range_padding_factor
    padded_Mc = Mc * doppler_padding_factor
    
    # pour réduire les composantes DC
    data = data - np.mean(data, axis=0, keepdims=True)
    data = remove_mean_per_row(data) # on retire la moyenne de chaque ligne

    windowed_data = apply_window(data, window_type) # ici hann
    
    # 1. FFT sur l'axe de la distance (axe 1)
    range_fft = np.fft.fft(windowed_data, n=padded_Ms, axis=1)
    
    # 2. FFT sur l'axe Doppler (axe 0)
    range_doppler_map = np.fft.fft(range_fft, n=padded_Mc, axis=0)
    
    # 3. Shift pour centrer les vitesses autour de zéro
    range_doppler_map = np.fft.fftshift(range_doppler_map, axes=0)
    
    # calcul des axes physiques
    range_axis = calculate_range_axis(params, padded_Ms)
    velocity_axis = calculate_velocity_axis(params, padded_Mc)
    
    # Calcul de l'amplitude
    range_doppler_magnitude = np.abs(range_doppler_map)
    
    # Appliquer un filtre médian si demandé pour réduire le bruit
    if apply_2d_filter:
        range_doppler_magnitude = ndimage.median_filter(range_doppler_magnitude, size=kernel_size)
    
    # Conversion en dB en retirant les valeurs nulles
    range_doppler_db = 20 * np.log10(np.maximum(range_doppler_magnitude, 1e-15))
    
    return range_doppler_db, range_axis, velocity_axis

def generate_range_doppler_map(data, window_type='hann', range_padding_factor=2, doppler_padding_factor=2):
    """
    Génère une Range-Doppler map à partir des données radar avec zero-padding
    pour améliorer la résolution
    
    Parameters:
    -----------
    data : ndarray
        Données de forme (Mc, samples_per_chirp)

    window_type : str
        Type de fenêtre à appliquer ('hann', 'hamming', 'blackman', etc.)
        
    range_padding_factor : int
        Facteur de zero-padding pour l'axe distance (1 = pas de padding)
        
    doppler_padding_factor : int
        Facteur de zero-padding pour l'axe Doppler (1 = pas de padding)
        
    Returns:
    --------
    range_doppler_map : ndarray
        Carte Range-Doppler en 2D (en dB)
    """
    # Dimensions
    Mc, Ms = data.shape
    padded_Ms = Ms * range_padding_factor
    padded_Mc = Mc * doppler_padding_factor
    
    # Prétraitement: soustraire la moyenne pour réduire les composantes DC
    centered_data = data - np.mean(data, axis=0, keepdims=True)
    
    # Fenêtre pour réduire les lobes secondaires
    windowed_data = apply_window(centered_data, window_type)
    
    # FFT 2D avec padding
    range_fft = np.fft.fft(windowed_data, n=padded_Ms, axis=1)
    range_doppler_map = np.fft.fft(range_fft, n=padded_Mc, axis=0)

    # Shift pour centrer les vitesses autour de zéro
    range_doppler_map = np.fft.fftshift(range_doppler_map, axes=0)
    
    # Conversion en dB avec protection contre les valeurs nulles
    range_doppler_db = 20 * np.log10(np.maximum(np.abs(range_doppler_map), 1e-15))
    
    return range_doppler_db

def apply_cfar_detector(range_doppler_map, guard_cells=(2, 4), training_cells=(4, 8), 
                        threshold_factor=13.0, cfar_method='CA', percentile=75):
    """
    Applique un détecteur CFAR (Constant False Alarm Rate) sur la carte Range-Doppler
    
    Parameters:
    -----------
    range_doppler_map : ndarray
        Carte Range-Doppler en dB

    guard_cells : tuple
        Nombre de cellules de garde (doppler, range)

    training_cells : tuple
        Nombre de cellules d'entraînement (doppler, range)

    threshold_factor : float
        Facteur de seuil en dB
        
    cfar_method : str
        Méthode CFAR à utiliser ('CA' pour Cell Averaging, 'OS' pour Ordered Statistics)
    
    percentile : int
        Percentile à utiliser pour CFAR OS (entre 0 et 100)
        
    Returns:
    --------
    detections : ndarray
        Masque binaire indiquant les détections
    """
    doppler_size, range_size = range_doppler_map.shape
    detections = np.zeros_like(range_doppler_map, dtype=bool)
    
    # Extraire les paramètres
    guard_doppler, guard_range = guard_cells
    train_doppler, train_range = training_cells
    
    # Vérifier que la carte est assez grande
    min_doppler_size = 2*guard_doppler + 2*train_doppler + 1
    min_range_size = 2*guard_range + 2*train_range + 1
    
    if doppler_size < min_doppler_size or range_size < min_range_size:
        raise ValueError(f"La carte Range-Doppler est trop petite pour les paramètres CFAR spécifiés. "
                         f"Taille minimale requise: ({min_doppler_size}, {min_range_size})")
    
    # Limites pour le parcours
    d_start = train_doppler + guard_doppler
    d_end = doppler_size - (train_doppler + guard_doppler)
    r_start = train_range + guard_range
    r_end = range_size - (train_range + guard_range)
    
    # Appliquer le CFAR
    for i in range(d_start, d_end):
        for j in range(r_start, r_end):
            # Extraire la fenêtre
            window = range_doppler_map[
                i - (train_doppler + guard_doppler):i + (train_doppler + guard_doppler) + 1,
                j - (train_range + guard_range):j + (train_range + guard_range) + 1
            ]
            
            # Créer un masque pour isoler les cellules d'entraînement
            mask = np.ones_like(window, dtype=bool)
            mask[
                train_doppler:train_doppler + 2*guard_doppler + 1,
                train_range:train_range + 2*guard_range + 1
            ] = False
            
            # Calculer le niveau de bruit selon la méthode choisie
            if cfar_method.upper() == 'CA':
                # Cell Averaging CFAR
                noise_level = np.mean(window[mask])
            elif cfar_method.upper() == 'OS':
                # Ordered Statistics CFAR
                training_cells_values = window[mask].flatten()
                training_cells_values = np.sort(training_cells_values)
                index = int(len(training_cells_values) * percentile / 100)
                noise_level = training_cells_values[index]
            else:
                # Par défaut, utiliser Cell Averaging
                noise_level = np.mean(window[mask])
            
            # Appliquer le seuil
            if range_doppler_map[i, j] > noise_level + threshold_factor:
                detections[i, j] = True
    
    return detections

def estimate_target_parameters(range_doppler_map, detections, range_axis, velocity_axis):
    """
    Estime les paramètres précis des cibles détectées par interpolation parabolique
    
    Parameters:
    -----------
    range_doppler_map : ndarray
        Carte Range-Doppler en dB
    detections : ndarray
        Masque binaire indiquant les détections CFAR
    range_axis : ndarray
        Axe de distance en mètres
    velocity_axis : ndarray
        Axe de vitesse en m/s
        
    Returns:
    --------
    targets : list of dict
        Liste des cibles détectées avec leurs paramètres estimés
    """
    doppler_indices, range_indices = np.where(detections)
    
    targets = []
    
    for i in range(len(doppler_indices)):
        d_idx = doppler_indices[i]
        r_idx = range_indices[i]
        
        # S'assurer qu'on n'est pas au bord de la carte
        if (d_idx > 0 and d_idx < range_doppler_map.shape[0] - 1 and 
            r_idx > 0 and r_idx < range_doppler_map.shape[1] - 1):
            
            # Extraire les valeurs pour l'interpolation
            range_values = range_doppler_map[d_idx, r_idx-1:r_idx+2]
            doppler_values = range_doppler_map[d_idx-1:d_idx+2, r_idx]
            
            # Interpolation parabolique pour la distance
            if range_values[0] < range_values[1] > range_values[2]:  # Vérifier qu'on a bien un pic
                delta_r = 0.5 * (range_values[0] - range_values[2]) / (range_values[0] - 2*range_values[1] + range_values[2])
                range_refined = range_axis[r_idx] + delta_r * (range_axis[1] - range_axis[0])
            else:
                range_refined = range_axis[r_idx]
            
            # Interpolation parabolique pour la vitesse
            if doppler_values[0] < doppler_values[1] > doppler_values[2]:  # Vérifier qu'on a bien un pic
                delta_d = 0.5 * (doppler_values[0] - doppler_values[2]) / (doppler_values[0] - 2*doppler_values[1] + doppler_values[2])
                velocity_refined = velocity_axis[d_idx] + delta_d * (velocity_axis[1] - velocity_axis[0])
            else:
                velocity_refined = velocity_axis[d_idx]
            
            # Amplitude estimée
            amplitude = range_doppler_map[d_idx, r_idx]
            
            # Ajouter la cible à la liste
            target = {
                'range': range_refined,
                'velocity': velocity_refined,
                'amplitude': amplitude,
                'range_index': r_idx,
                'doppler_index': d_idx
            }
            
            targets.append(target)
    
    return targets

def remove_static_components(range_doppler_map, linear_scale=True):
    """
    Supprime les composantes statiques (vitesse proche de zéro) de la Range-Doppler map
    
    Parameters:
    -----------
    range_doppler_map : ndarray
        Carte Range-Doppler en dB ou en échelle linéaire
    linear_scale : bool
        Indique si les données sont déjà en échelle linéaire ou en dB
        
    Returns:
    --------
    processed_map : ndarray
        Carte Range-Doppler avec composantes statiques supprimées (même échelle que l'entrée)
    """
    # Conversion en linéaire si nécessaire
    if not linear_scale:
        linear_data = 10**(range_doppler_map/20)
    else:
        linear_data = range_doppler_map.copy()
    
    # Moyenne sur l'axe de distance pour chaque vitesse
    column_means = np.mean(linear_data, axis=1, keepdims=True)
    
    # Soustraire la moyenne
    linear_data_no_static = linear_data - column_means
    
    # Reconvertir en dB si nécessaire
    if not linear_scale:
        processed_map = 20 * np.log10(np.maximum(np.abs(linear_data_no_static), 1e-15))
    else:
        processed_map = linear_data_no_static
    
    return processed_map

def apply_clutter_threshold(range_doppler_map, threshold_db):
    """
    Applique un seuil de clutter à la Range-Doppler map
    
    Parameters:
    -----------
    range_doppler_map : ndarray
        Carte Range-Doppler en dB
    threshold_db : float
        Seuil en dB sous le maximum
        
    Returns:
    --------
    thresholded_map : ndarray
        Carte Range-Doppler avec seuil appliqué
    """
    if threshold_db <= 0:
        return range_doppler_map
    
    # Calculer le seuil
    max_val = np.max(range_doppler_map)
    threshold = max_val - threshold_db
    
    # Appliquer le seuil
    thresholded_map = range_doppler_map.copy()
    thresholded_map[thresholded_map < threshold] = threshold
    
    return thresholded_map