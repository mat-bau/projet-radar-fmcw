import numpy as np
from scipy import signal

def apply_window(data, window_type='hann'):
    """
    Applique une fenêtre aux données pour réduire les lobes secondaires
    
    Parameters:
    -----------
    data : ndarray
        Données de forme (num_chirps, samples_per_chirp)

    window_type : str
        Type de fenêtre à appliquer ('hann', 'hamming', 'blackman', etc.)
        
    Returns:
    --------
    windowed_data : ndarray
        Données avec fenêtre appliquée
    """
    num_chirps, num_samples = data.shape
    
    # crée les fenêtres
    range_window = getattr(signal.windows, window_type)(num_samples)
    doppler_window = getattr(signal.windows, window_type)(num_chirps)
    
    # Appliquer la fenêtre en distance ici colonnes
    windowed_data = data * range_window[np.newaxis, :]
    
    # Appliquer la fenêtre en vitesse ici lignes
    windowed_data = windowed_data * doppler_window[:, np.newaxis]
    
    return windowed_data

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
    num_chirps, num_samples = data.shape
    
    # fenêtre en distance uniquement
    range_window = getattr(signal.windows, window_type)(num_samples)
    windowed_data = data * range_window[np.newaxis, :]
    
    # FFT sur chaque chirp (dimension distance)
    range_fft = np.fft.fft(windowed_data, axis=1)
    
    # Moyenner sur tous les chirps
    range_profile = np.mean(np.abs(range_fft), axis=0)
    
    # convertion en dB
    range_profile_db = 20 * np.log10(range_profile + 1e-15)
    
    return range_profile_db

def generate_range_doppler_map_with_axes(data, params, window_type='hann', 
                                        range_padding_factor=2, doppler_padding_factor=4,
                                        apply_2d_filter=True, kernel_size=3):
    """
    Génère une carte Range-Doppler à partir des données radar avec zero-padding
    et retourne également les axes physiques
    
    Parameters:
    -----------
    data : ndarray
        Données de forme (num_chirps, samples_per_chirp)
        
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
        Taille du noyau du filtre 2D (si apply_2d_filter=True)
        
    Returns:
    --------
    range_doppler_map : ndarray
        Carte Range-Doppler en 2D (en dB)
        
    range_axis : ndarray
        Axe de distance en mètres
        
    velocity_axis : ndarray
        Axe de vitesse en m/s
    """
    # Extraire les dimensions
    num_chirps, num_samples = data.shape
    
    # Calculer les nouvelles dimensions avec zero-padding
    padded_num_samples = num_samples * range_padding_factor
    # print(f"padded_num_samples: {padded_num_samples}")
    padded_num_chirps = num_chirps * doppler_padding_factor
    
    # Appliquer une fenêtre aux données
    windowed_data = apply_window(data, window_type)
    
    # 1ère FFT pour la distance avec zero-padding
    range_fft = np.fft.fft(windowed_data, n=padded_num_samples, axis=1)
    # print(f"range_fft vite fait: {range_fft}") # encore du mal avec ce padding
    # 2e FFT pour la vitesse avec zero-padding
    range_doppler_map = np.fft.fft(range_fft, n=padded_num_chirps, axis=0)

    # et shift pour avoir 0 au centre
    range_doppler_map = np.fft.fftshift(range_doppler_map, axes=0)
    
    # Calculer les axes physiques
    range_axis = calculate_range_axis(params, padded_num_samples)
    velocity_axis = calculate_velocity_axis(params, padded_num_chirps)
    
    # Convertir en magnitude
    range_doppler_magnitude = np.abs(range_doppler_map)
    
    # Appliquer un filtre 2D pour réduire le bruit si demandé
    if apply_2d_filter:
        from scipy import ndimage
        range_doppler_magnitude = ndimage.median_filter(range_doppler_magnitude, size=kernel_size)
    
    # Convertir en dB
    range_doppler_db = 20 * np.log10(range_doppler_magnitude + 1e-15)
    
    return range_doppler_db, range_axis, velocity_axis


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
    # pour check que les paramètres nécessaires sont présents
    if 'samples_per_chirp' not in params or 'range_resolution' not in params:
        raise KeyError("Paramètres manquants: 'samples_per_chirp' ou 'range_resolution'")
    
    Ms = int(params['samples_per_chirp'])  # Convertir en entier pour éviter les problèmes de type
    range_resolution = params['range_resolution']
    
    # Si padded_samples n'est pas spécifié, utiliser Ms
    if padded_samples is None:
        padded_samples = Ms
    
    # Calcul de la nouvelle résolution avec zero-padding
    actual_resolution = range_resolution * (Ms / padded_samples)
    
    # Axe de distance
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
    Mc = int(params['num_chirps'])  # Convertir en entier pour éviter les problèmes de type
    Tc = params['chirp_time']
    
    # Si padded_chirps n'est pas spécifié, utiliser Mc
    if padded_chirps is None:
        padded_chirps = Mc
    
    # Calcul de la vitesse maximale sans ambiguïté
    wavelength = 3e8 / f0  # Longueur d'onde
    velocity_max = wavelength / (4 * Tc)  # Vitesse maximale sans ambiguïté
    
    # Calcul de la résolution de vitesse avec zero-padding
    velocity_resolution = (2 * velocity_max) / padded_chirps
    
    # Axe de vitesse centré sur zéro
    velocity_axis = np.arange(-padded_chirps//2, padded_chirps//2) * velocity_resolution
    
    return velocity_axis


def generate_range_doppler_map(data, window_type='hann', range_padding_factor=2, doppler_padding_factor=2):
    """
    Génère une carte Range-Doppler à partir des données radar avec zero-padding
    pour améliorer la résolution
    
    Parameters:
    -----------
    data : ndarray
        Données de forme (num_chirps, samples_per_chirp)

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
    # Extraire les dimensions
    num_chirps, num_samples = data.shape
    
    # Calculer les nouvelles dimensions avec zero-padding
    padded_num_samples = num_samples * range_padding_factor
    padded_num_chirps = num_chirps * doppler_padding_factor
    
    # Appliquer une fenêtre aux données
    windowed_data = apply_window(data, window_type)
    
    # 1ère FFT pour la distance avec zero-padding
    range_fft = np.fft.fft(windowed_data, n=padded_num_samples, axis=1)
    
    # 2e FFT pour la vitesse avec zero-padding
    range_doppler_map = np.fft.fft(range_fft, n=padded_num_chirps, axis=0)

    # et shift pour avoir 0 au centre
    range_doppler_map = np.fft.fftshift(range_doppler_map, axes=0)
    
    # Magnitude en dB!
    range_doppler_db = 20 * np.log10(np.abs(range_doppler_map) + 1e-15)
    
    return range_doppler_db

def apply_cfar_detector(range_doppler_map, guard_cells=(2, 4), training_cells=(4, 8), threshold_factor=13.0):
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
            
            # Calculer le niveau de bruit moyen
            noise_level = np.mean(window[mask])
            
            # Appliquer le seuil
            if range_doppler_map[i, j] > noise_level + threshold_factor:
                detections[i, j] = True
    
    return detections