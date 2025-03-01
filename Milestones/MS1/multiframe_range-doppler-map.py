import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows

def create_range_doppler_map(frame_data, chirp_params, range_padding=4, doppler_padding=4):
    f0, B, Ms, Mc, Ts, Tc = chirp_params
    Ms = int(Ms)
    Mc = int(Mc)

    I1 = frame_data[0, :]
    Q1 = frame_data[1, :]

    complex_signal = I1 - 1j * Q1  # Essayer avec + ?

    expected_size = Mc * Ms
    actual_size = len(complex_signal)

    if actual_size != expected_size:
        total_samples = len(complex_signal)
        if total_samples % Mc == 0:
            Ms = total_samples // Mc
        else:
            Ms = total_samples // Mc
            complex_signal = complex_signal[:Ms * Mc]

    signal_matrix = np.reshape(complex_signal, (Mc, Ms))

    # Suppression des moyennes des colonnes?
    signal_matrix -= np.mean(signal_matrix, axis=0, keepdims=True)

    range_window = windows.hann(Ms)
    doppler_window = windows.hann(Mc)

    windowed_signal = signal_matrix * doppler_window[:, np.newaxis]
    windowed_signal = windowed_signal * range_window[np.newaxis, :]

    n_range = Ms * range_padding
    n_doppler = Mc * doppler_padding

    range_fft = np.fft.fft(windowed_signal, n=n_range, axis=1)
    range_doppler_map = np.fft.fftshift(np.fft.fft(range_fft, n=n_doppler, axis=0), axes=0)

    range_doppler_map_db = 20 * np.log10(np.abs(range_doppler_map) + 1e-10)

    c = 3e8  
    lambda_wave = c / f0  

    range_res = c / (2 * B)  
    max_range = range_res * n_range / 2
    range_axis = np.linspace(0, max_range, n_range)

    doppler_res = lambda_wave / (2 * Tc * Mc)  
    max_velocity = doppler_res * n_doppler / 2
    velocity_axis = np.linspace(-max_velocity, max_velocity, n_doppler)

    # Supprimer les données liées aux vitesses nulles
    range_doppler_map_db[:, np.argmin(np.abs(velocity_axis))] = 0

    return range_doppler_map_db, range_axis, velocity_axis


def plot_multiple_range_doppler_maps(radar_data, chirp_params, num_frames=6, max_range=30, max_velocity=20):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))  
    axes = axes.flatten()  

    num_frames = min(num_frames, radar_data.shape[0])

    vmin_global = None
    vmax_global = None
    all_maps = []

    for i in range(num_frames):
        frame_data = radar_data[i]
        rd_map, range_axis, velocity_axis = create_range_doppler_map(
            frame_data, chirp_params, range_padding=4, doppler_padding=4
        )
        
        all_maps.append(rd_map)

        curr_max = np.max(rd_map)
        if vmax_global is None or curr_max > vmax_global:
            vmax_global = curr_max

    vmin_global = vmax_global - 40  

    for i in range(num_frames):
        ax = axes[i]
        rd_map = all_maps[i]

        transposed_map = rd_map.T

        im = ax.imshow(transposed_map, aspect='auto', origin='lower', 
                      extent=[-max_velocity, max_velocity, 0, max_range],
                      cmap='jet', vmin=vmin_global, vmax=vmax_global,
                      interpolation='bilinear')
        
        ax.set_xlabel('Vitesse (m/s)')
        if i % 3 == 0:  
            ax.set_ylabel('Distance (m)')
        ax.set_title(f'Frame {i}')
        
        ax.set_xlim([-max_velocity, max_velocity])
        ax.set_ylim([0, max_range])

        ax.axvline(x=0, color='w', linestyle='--', alpha=0.5)  
        ax.grid(True, color='w', linestyle='-', alpha=0.2)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Magnitude (dB)')

    plt.suptitle('Range-Doppler map', fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])  
    plt.savefig('multrange_doppler_maps.pdf')
    plt.show()

if __name__ == "__main__":
    file_path = 'MS1-FMCW.npz'
    
    try:
        data_dict = np.load(file_path, allow_pickle=True)
        
        radar_data = data_dict["data"]
        chirp_params = tuple(data_dict["chirp"])

        print(f"Fichier chargé avec succès! Dimensions des données: {radar_data.shape}")
        print(f"Paramètres du chirp: {chirp_params}")

        for i in range(6):
            print(f"Min/Max frame {i}: {np.min(radar_data[i])}/{np.max(radar_data[i])}")
        
        plot_multiple_range_doppler_maps(
            radar_data, 
            chirp_params, 
            num_frames=6, 
            max_range=30,  
            max_velocity=20  
        )
        
        print("Cartes range-Doppler générées avec succès!")
        
    except Exception as e:
        import traceback
        print(f"Erreur lors du chargement ou du traitement du fichier: {e}")
        traceback.print_exc()
