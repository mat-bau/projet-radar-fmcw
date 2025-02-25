import numpy as np
import matplotlib.pyplot as plt

# Fonction pour trouver la prochaine puissance de 2
def nextpow2(n):
    return int(2**np.ceil(np.log2(n)))

# Charger les données
data = np.load('MS1-FMCW.npz')
radar_data = data['data']
chirp_params = data['chirp']

# Extraire les paramètres de la modulation FMCW
f0, B, Ms, Mc, Ts, Tc = chirp_params

# Convertir Ms et Mc en entiers
Ms = int(Ms)
Mc = int(Mc)

# Extraire les données I1 et Q1
I1 = radar_data[:, 0, :]
Q1 = radar_data[:, 1, :]

# Déterminer la vraie valeur de Ms en fonction des données
M = radar_data.shape[2]  # Nombre total d’échantillons par frame
Ms_corrected = M // Mc   # Correction pour respecter la vraie taille

# --- Calcul de l'échelle de distance ---
c = 3e8  # Vitesse de la lumière (m/s)
range_resolution = c / (2 * B)  # Résolution en distance (m/bin)
max_range = Ms_corrected * range_resolution  # Distance max théorique
range_bins = np.linspace(0, max_range, Ms_corrected)  # Axe distance

# Filtrer pour garder seulement [0, 30] mètres
valid_range = range_bins <= 30
range_bins = range_bins[valid_range]
Ms_valid = len(range_bins)  # Nouvelle taille après filtrage

# --- Calcul de l'échelle de vélocité ---
lambda_radar = c / f0  # Longueur d’onde du radar
velocity_resolution = lambda_radar / (2 * Tc * Mc)  # Résolution en vitesse (m/s par bin)
max_velocity = velocity_resolution * (Mc // 2)  # Valeur max de la vitesse Doppler
doppler_bins = np.linspace(-max_velocity, max_velocity, Mc)  # Axe Doppler

# --- Définir les tailles de FFT avec zero-padding ---
N_range_fft = nextpow2(Ms_valid)  # Zero-padding en distance
N_doppler_fft = nextpow2(Mc)  # Zero-padding en Doppler

# Traiter une frame à la fois
N_Frame = radar_data.shape[0]
range_doppler_maps = []

for frame in range(N_Frame):
    # Extraire les échantillons de la frame courante
    I1_frame = I1[frame, :]
    Q1_frame = Q1[frame, :]
    
    # Combiner I1 et Q1 pour obtenir le signal complexe
    signal = I1_frame + 1j * Q1_frame
    
    # Reshape du signal en une matrice de taille (Mc, Ms_corrected)
    signal_matrix = signal.reshape(Mc, Ms_corrected)
    
    # Appliquer la FFT en range avec zero-padding
    range_fft = np.fft.fft(signal_matrix, n=N_range_fft, axis=1)
    
    # Appliquer la FFT en Doppler avec zero-padding
    doppler_fft = np.fft.fftshift(np.fft.fft(range_fft, n=N_doppler_fft, axis=0), axes=0)
    
    # Prendre le module et filtrer la distance
    range_doppler_map = np.abs(doppler_fft[:, :Ms_valid])
    range_doppler_maps.append(range_doppler_map)

# Mise à jour des axes pour correspondre aux nouvelles tailles de FFT
range_bins_interp = np.linspace(0, 30, N_range_fft)  # Interpolation pour la range FFT
doppler_bins_interp = np.linspace(-max_velocity, max_velocity, N_doppler_fft)  # Interpolation pour la Doppler FFT

# Afficher la carte range-Doppler pour la première frame
plt.figure(figsize=(8, 6))
plt.imshow(range_doppler_maps[0], aspect='auto', 
           extent=[0, 30, -max_velocity, max_velocity], origin='lower')
plt.title('Range-Velocity Map (Zero-Padded FFT)')
plt.xlabel('Distance (m)')
plt.ylabel('Velocity (m/s)')
plt.colorbar(label='Magnitude')
plt.show()
