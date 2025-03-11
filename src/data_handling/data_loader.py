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
    parms : dict
        Dictionnaire contenant les paramètres du radar
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
        params['max_range'] = (3e8 * Ms) / (2 * B) # Distance max (m)
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
    complex_data = I + 1j * Q
    
    return complex_data

def reshape_to_chirps(complex_data, params, methode="without_pause"):
    """
    Reshape les données complexes en tableau 2D (chirps x échantillons)
    
    Parameters:
    -----------
    complex_data : ndarray
        Données complexes d'une frame
    params : dict
        Dictionnaire des paramètres
    methode : str
        Méthode de reshape des données (with_pause or without_pause)
    Returns:
    --------
    reshaped_data : ndarray
        Données reshapées de shape (num_chirps, samples_per_chirp)
    """
    Ms = int(params['samples_per_chirp']) 
    Mc = int(params['num_chirps'])     

    
    expected_size = int(Mc * Ms)
    actual_size = len(complex_data)
    Mpause = actual_size//Mc - Ms  # M = Mc*(Ms+Mpause)

    if actual_size != expected_size:
        radar_data = np.reshape(complex_data, (Mc, Ms+Mpause)) # ligne de taille Mc (chaque colonne est un chirp, un info sur Doppler) et colonne de taille Ms+Mpause (info sur distance)
        
        if methode == "with_pause": # on garde les échantillons de la pause
            return radar_data
        elif methode == "without_pause":
            radar_data = radar_data[:, :Ms]    # on enlève les échantillons de la pause à la fin de chaque colonne (chirp)
    
    return radar_data

def subtract_background(data, background_data):
    """
    Soustrait les données de fond des données radar
    
    Parameters:
    -----------
    data : ndarray
        Données radar originales
    background_data : ndarray
        Données de fond à soustraire
        
    Returns:
    --------
    subtracted_data : ndarray
        Données radar avec le fond soustrait
    """
    # Update du labo2 : il faut faire la moyenne des données de fond

    # S'assurer que les données ont la même forme
    if data.shape != background_data.shape:
        raise ValueError(f"Les dimensions des données ({data.shape}) et du fond ({background_data.shape}) ne correspondent pas")
    
    # Les données sont complexes, donc on fait une soustraction complexe
    return data - background_data