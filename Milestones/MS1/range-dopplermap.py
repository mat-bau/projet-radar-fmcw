import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows

def create_range_doppler_map(frame_data, chirp_params, range_padding=4, doppler_padding=4):
    """
    Crée une carte range-Doppler à partir des données d'une frame
    
    Args:
        frame_data: Données d'une frame de taille (N_Chan, M)
        chirp_params: Tuple (f0, B, Ms, Mc, Ts, Tc)
        range_padding: Facteur de zero-padding pour la FFT en distance
        doppler_padding: Facteur de zero-padding pour la FFT Doppler
    
    Returns:
        range_doppler_map: Carte range-Doppler en dB
        range_axis: Axe des distances en mètres
        velocity_axis: Axe des vitesses en m/s
    """
    # Extraction des paramètres et conversion en entiers si nécessaire
    f0, B, Ms, Mc, Ts, Tc = chirp_params
    Ms = int(Ms) 
    Mc = int(Mc) 
    
    print(f"Paramètres convertis: Ms={Ms}, Mc={Mc}")
    print(f"Forme des données de la frame: {frame_data.shape}")
    
    # Recalcul de la taille totale attendue pour vérification
    expected_size = Mc * Ms
    actual_size = len(frame_data[0])
    print(f"Taille attendue: {expected_size}, Taille réelle: {actual_size}")
    
    # Creation du signal complexe
    I1 = frame_data[0, :]
    Q1 = frame_data[1, :]
    complex_signal = I1 + 1j * Q1
    
    # Vérification de la taille des données
    if actual_size != expected_size:
        print("Attention: La taille des données ne correspond pas exactement au produit Ms*Mc")
        print("Ajustement de Ms et Mc pour correspondre aux données...")
        
        # Calcul des dimensions optimales
        total_samples = len(complex_signal)
        # Trouver un facteur Ms (échantillons par rampe) qui divise bien la taille totale
        if total_samples % Mc == 0:
            Ms = total_samples // Mc
            print(f"Ajustement: nouveau Ms={Ms}")
        else:
            # Si Mc ne divise pas bien, on peut tronquer les données
            Ms = total_samples // Mc
            complex_signal = complex_signal[:Ms*Mc]
            print(f"Troncature des données à {len(complex_signal)} échantillons, nouveau Ms={Ms}")
    
    # Reshape du signal pour obtenir une matrice de chirps
    # Chaque ligne correspond à un chirp, chaque colonne à un échantillon
    signal_matrix = np.reshape(complex_signal, (Mc, Ms))
    print(f"Forme de la matrice de signal après reshape: {signal_matrix.shape}")
    
    # Application d'une fenêtre de Hanning en 2D pour réduire les lobes secondaires
    range_window = windows.hann(Ms)
    doppler_window = windows.hann(Mc)
    
    windowed_signal = signal_matrix * doppler_window[:, np.newaxis]
    windowed_signal = windowed_signal * range_window[np.newaxis, :]
    
    # Tailles de FFT avec zero-padding
    n_range = Ms * range_padding
    n_doppler = Mc * doppler_padding
    
    print(f"Tailles de FFT avec zero-padding: range FFT={n_range}, doppler FFT={n_doppler}")
    
    # Calcul de la FFT 2D avec zero-padding
    # 1ère FFT sur chaque chirp (range) avec zero-padding
    range_fft = np.fft.fft(windowed_signal, n=n_range, axis=1)
    
    # 2ème FFT sur chaque bin de distance (Doppler) avec zero-padding
    range_doppler_map = np.fft.fftshift(np.fft.fft(range_fft, n=n_doppler, axis=0), axes=0)
    
    # Conversion en magnitude (dB)
    range_doppler_map_db = 20 * np.log10(np.abs(range_doppler_map) + 1e-10)
    
    # Calcul des axes
    c = 3e8  # Vitesse de la lumière en m/s
    lambda_wave = c / f0  # Longueur d'onde
    
    # Axe des distances (ajusté avec le zero-padding)
    range_res = c / (2 * B)  # Résolution en distance sans zero-padding
    max_range = range_res * n_range / 2
    range_axis = np.linspace(0, max_range, n_range)
    
    # Axe des vitesses (ajusté avec le zero-padding)
    doppler_res = lambda_wave / (2 * Tc * Mc)  # Résolution Doppler sans zero-padding
    max_velocity = doppler_res * n_doppler / 2
    velocity_axis = np.linspace(-max_velocity, max_velocity, n_doppler)
    
    return range_doppler_map_db, range_axis, velocity_axis

def plot_range_doppler_map(range_doppler_map, range_axis, velocity_axis, frame_idx):
    """
    Affiche la carte range-Doppler avec vitesse en abscisse et distance en ordonnée
    en utilisant plt.contourf pour une meilleure visualisation
    """
    plt.figure(figsize=(12, 8))
    
    # Transposer la carte pour inverser les axes
    transposed_map = range_doppler_map.T
    
    # Créer un maillage pour contourf
    V, R = np.meshgrid(velocity_axis, range_axis)
    
    # Déterminer les niveaux de contour
    # Trouver les valeurs min et max pour normaliser l'échelle de couleurs
    vmin = np.max(transposed_map) - 40  # Généralement -40dB sous le max est un bon seuil
    vmax = np.max(transposed_map)
    
    # Créer 20 niveaux entre vmin et vmax
    levels = np.linspace(vmin, vmax, 20)
    
    # Afficher avec contourf
    contour = plt.contourf(V, R, transposed_map, levels=levels, cmap='jet', extend='both')
    
    # Ajouter une colorbar
    cbar = plt.colorbar(contour)
    cbar.set_label('Magnitude (dB)')
    
    # Étiquettes et titre
    plt.xlabel('Vitesse (m/s)')
    plt.ylabel('Distance (m)')
    plt.title(f'Carte Range-Doppler (Contour) - Frame {frame_idx}')
    
    # Ajout de lignes de référence
    plt.axvline(x=0, color='w', linestyle='--', alpha=0.5)  # Ligne de vitesse zéro
    plt.grid(True, color='w', linestyle='-', alpha=0.2)
    
    plt.tight_layout()
    
    # Créer une deuxième figure pour comparer avec la méthode imshow
    plt.figure(figsize=(12, 8))
    plt.imshow(transposed_map, aspect='auto', origin='lower', 
               extent=[velocity_axis[0], velocity_axis[-1], range_axis[0], range_axis[-1]],
               cmap='jet', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Magnitude (dB)')
    plt.xlabel('Vitesse (m/s)')
    plt.ylabel('Distance (m)')
    plt.title(f'Carte Range-Doppler (Imshow) - Frame {frame_idx}')
    plt.axvline(x=0, color='w', linestyle='--', alpha=0.5)
    plt.grid(True, color='w', linestyle='-', alpha=0.2)
    plt.tight_layout()
    
    plt.show()

# Fonction principale qui charge le fichier et affiche directement le graphe
if __name__ == "__main__":
    # Nom du fichier à charger
    file_path = 'MS1-FMCW.npz'
    frame_to_process = 4  # Indice de la frame à traiter
    
    # Facteurs de zero-padding
    range_padding = 4  # Facteur de zero-padding pour la FFT en distance
    doppler_padding = 4  # Facteur de zero-padding pour la FFT Doppler
    
    # Chargement des données
    try:
        data_dict = np.load(file_path, allow_pickle=True)
        
        # Extrait les données
        radar_data = data_dict["data"]
        chirp_params = tuple(data_dict["chirp"])  # Conversion en tuple pour s'assurer de la compatibilité
        
        print(f"Fichier chargé avec succès! Dimensions des données: {radar_data.shape}")
        print(f"Paramètres du chirp: {chirp_params}")
        
        # Traitement d'une frame spécifique
        frame_data = radar_data[frame_to_process]
        
        # Création de la carte range-Doppler avec zero-padding
        range_doppler_map, range_axis, velocity_axis = create_range_doppler_map(
            frame_data, chirp_params, range_padding, doppler_padding
        )
        
        # Affichage de la carte
        plot_range_doppler_map(range_doppler_map, range_axis, velocity_axis, frame_to_process)
        
        print("Carte range-Doppler générée avec succès!")
        
    except Exception as e:
        import traceback
        print(f"Erreur lors du chargement ou du traitement du fichier: {e}")
        traceback.print_exc()