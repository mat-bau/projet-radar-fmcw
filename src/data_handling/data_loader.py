import numpy as np
import os

def load_fmcw_data(filepath):
    """
    Charge les données FMCW depuis un fichier .npz
    
    Parameters:
    -----------
    filepath : str
        Chemin vers le fichier .npz contenant les données
        
    Returns:
    --------
    data_dict : dict
        Dictionnaire contenant les données et les paramètres
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Le fichier {filepath} n'existe pas.")
    
    try:
        data_dict = np.load(filepath, allow_pickle=True)
        required_keys = ['data', 'chirp']
        for key in required_keys:
            if key not in data_dict:
                raise KeyError(f"La clé '{key}' est manquante dans le fichier de données.")
        
        # paramètres du chirp
        f0, B, Ms, Mc, Ts, Tc = data_dict['chirp']
        
        # Créer un dictionnaire de paramètres plus explicite
        params = {
            'start_freq': f0,       # Fréquence porteuse (Hz)
            'bandwidth': B,         # Largeur de bande (Hz)
            'samples_per_chirp': Ms,  # Échantillons par rampe (sans les pauses)
            'num_chirps': Mc,       # Nombre de rampes
            'sample_time': Ts,      # Pas d'échantillonnage (s)
            'chirp_time': Tc,       # Temps entre chirps (s)
            'sample_rate': 1/Ts,    # Taux d'échantillonnage (Hz)
        }
        
        # autres paramètres calculés
        params['range_resolution'] = 3e8 / (2 * B)  # Résolution en distance (m)
        params['max_range'] = 3e8 * Ms * Ts / 2     # Portée maximale (m)
        params['velocity_resolution'] = 3e8 / (2 * f0 * Mc * Tc)  # Résolution en vitesse (m/s)
        params['max_velocity'] = 3e8 / (4 * f0 * Tc)  # Vitesse max (m/s)
        
        return data_dict['data'], params
        
    except Exception as e:
        raise Exception(f"Erreur lors du chargement du fichier: {str(e)}")

def extract_frame(data, frame_index=0, channel_indices=(0, 1)):
    """
    Extrait une frame spécifique des données et les canaux I/Q spécifiés
    
    Parameters:
    -----------
    data : ndarray
        Données brutes de forme (N_Frame, N_Chan, M)
        
    frame_index : int
        Indice de la frame à extraire

    channel_indices : tuple
        Indices des canaux à extraire (par défaut I1, Q1)
        
    Returns:
    --------
    complex_data : ndarray Données complexes (I + jQ) pour la frame spécifiée
    """
    if frame_index >= data.shape[0]:
        raise ValueError(f"L'indice de frame {frame_index} est hors limites. "
                        f"Le fichier contient {data.shape[0]} frames.")
    
    # Extraire I et Q
    I = data[frame_index, channel_indices[0], :]
    Q = data[frame_index, channel_indices[1], :]
    
    # e_z
    complex_data = I - 1j * Q
    
    return complex_data

def reshape_to_chirps(complex_data, params):
    """
    Reshape les données complexes en tableau 2D (chirps x échantillons)
    
    Parameters:
    -----------
    complex_data : ndarray
        Données complexes d'une frame
    params : dict
        Dictionnaire des paramètres
        
    Returns:
    --------
    reshaped_data : ndarray
        Données reshapées de forme (num_chirps, samples_per_chirp)
    """
    Ms = int(params['samples_per_chirp']) 
    Mc = int(params['num_chirps'])         
    
    expected_size = int(Mc * Ms)
    actual_size = len(complex_data)

    # on tronque ou on complète
    if actual_size >= expected_size:
        complex_data = complex_data[:expected_size]
    else:
        # si c'est plus court on complète avec des 0, ca n'arrive jamais en pratique
        complex_data = np.pad(complex_data, (0, expected_size - actual_size))
    
    # Reshape de data qui est en 1D [I1+jQ1...] en matrice 2D [Mc x Ms] = [[I1+jQ1, I2+jQ2, ..., I_Ms+jQ_Ms], ...] (chaque ligne est une chirp)
    reshaped_data = np.reshape(complex_data, (Mc, Ms))

    return reshaped_data

"""     
        # Reshape des données !! Attention c'est assez critique ici, je dois encore y regarder
        if len(complex_data) != expected_size:
            total_samples = len(complex_data)
            if total_samples % Mc == 0:
                Ms = total_samples // Mc
            else:
                Ms = total_samples // Mc
                complex_data = complex_data[:Ms * Mc]
            radar_data = np.reshape(complex_data, (Mc, Ms)) # ligne de taille Mc (info sur Doppler) et colonne de taille Ms (info sur distance)
"""