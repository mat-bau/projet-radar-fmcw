import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy import signal

# Ajouter le répertoire parent au chemin d'accès
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import des fonctions des packages dans src
from src.data_handling.data_loader import load_fmcw_data, extract_frame, reshape_to_chirps
from src.signal_processing.range_doppler_map import (
    generate_range_profile, 
    generate_range_doppler_map_with_axes,
    apply_cfar_detector
)
from src.visualization.plotting import (
    plot_range_profile, 
    plot_range_doppler,
    visualize_3d_range_doppler
)

def main():
    """Fonction principale pour l'analyse des données FMCW"""
    
    # Configuration des paramètres
    parser = argparse.ArgumentParser(description='Analyse des données FMCW')
    parser.add_argument('--data-file', type=str, required=False,
                       help='Chemin vers le fichier de données')
    parser.add_argument('--frame', type=int, default=0, # dans nos fichiers de labo, il y en a généralement 30 faut que je trouve un moyen de l'animer ou quoi
                       help='Indice de la frame à analyser')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Répertoire de sortie pour les visualisations')
    parser.add_argument('--detect-targets', action='store_true',
                       help='Activer la détection de cibles avec CFAR')
    parser.add_argument('--dynamic-range', type=int, default=40,
                       help='Plage dynamique en dB pour la visualisation')
    parser.add_argument('--range-padding', type=int, default=10,
                       help='Facteur de zero-padding pour l\'axe distance')
    parser.add_argument('--doppler-padding', type=int, default=10,
                       help='Facteur de zero-padding pour l\'axe Doppler')
    parser.add_argument('--window-type', type=str, default='hann',
                       help='Type de fenêtre à appliquer (hann, hamming, blackman, etc.)')
    parser.add_argument('--apply-2d-filter', action='store_true',
                       help='Appliquer un filtre 2D pour réduire le bruit')
    args = parser.parse_args()
    
    # # # # # # # # # # # # # #
    # Parse le nom du fichier #
    # # # # # # # # # # # # # #
    if args.data_file is None:
        if 'RADAR_DATA_FILE' in os.environ:
            args.data_file = os.environ['RADAR_DATA_FILE']
        else:
            args.data_file = 'data/MS1-FMCW.npz'  # Valeur par défaut

    filename_base = os.path.splitext(os.path.basename(args.data_file))[0]

    if args.output_dir is None:
        args.output_dir = os.path.join('output', filename_base)

    os.makedirs(args.output_dir, exist_ok=True)
    
    # check que le fichier existe
    if not os.path.exists(args.data_file):
        print(f"Erreur: Le fichier {args.data_file} n'existe pas.")
        print(f"Chemin absolu: {os.path.abspath(args.data_file)}")
        return
    
    print(f"Chargement des données depuis {args.data_file}...")
    
    try:
        data, params = load_fmcw_data(args.data_file)
        
        # Check que la frame demandée existe
        if args.frame >= data.shape[0]:
            print(f"Erreur: La frame {args.frame} n'existe pas. Le fichier contient {data.shape[0]} frames.")
            args.frame = 0
            print(f"Utilisation de la frame 0 à la place.")
        
        # Affichage des informations sur les données
        print("\nInformations sur les données:")
        print(f"Nombre de frames: {data.shape[0]}")
        print(f"Nombre de canaux: {data.shape[1]}")
        print(f"Nombre total d'échantillons par frame: {data.shape[2]}")
        
        print("\nParamètres du radar:")
        print(f"Fréquence porteuse: {params['start_freq']/1e9:.2f} GHz")
        print(f"Bande passante: {params['bandwidth']/1e6:.2f} MHz")
        print(f"Échantillons par chirp: {int(params['samples_per_chirp'])}")
        print(f"Nombre de chirps: {int(params['num_chirps'])}")
        print(f"Taux d'échantillonnage: {params['sample_rate']/1e6:.2f} MHz")
        
        print(f"\nPerformances calculées:")
        print(f"Résolution en distance: {params['range_resolution']:.2f} m")
        print(f"Portée maximale: {params['max_range']:.2f} m")
        print(f"Résolution en vitesse: {params['velocity_resolution']:.2f} m/s")
        print(f"Vitesse maximale détectable: ±{params['max_velocity']:.2f} m/s")
        
        # Paramètres de padding
        print(f"\nZero-padding:")
        print(f"Facteur de padding en distance: {args.range_padding}x")
        print(f"Facteur de padding en Doppler: {args.doppler_padding}x")
        
        print(f"\nAnalyse de la frame {args.frame}...")
        
        # Extraire une frame (I1 et Q1)
        complex_data = extract_frame(data, frame_index=args.frame, channel_indices=(0, 1))
        
        radar_data = reshape_to_chirps(complex_data, params, 2)
        
        print(f"Forme des données après reshape: {radar_data.shape}")

        print("\nGénération des visualisations...")
        
        # 1. Profil de distance
        print("Génération du profil de distance...")
        try:
            range_profile = generate_range_profile(radar_data, window_type=args.window_type)
        except AttributeError:
            print(f"Type de fenêtre '{args.window_type}' non disponible, utilisation de 'hamming' à la place.")
            range_profile = generate_range_profile(radar_data, window_type='hamming')
        
        # Calcul des axes sans zero-padding pour le profil de distance
        from src.signal_processing.range_doppler_map import calculate_range_axis
        range_axis_no_padding = calculate_range_axis(params)
        
        # Extraire le nom du fichier pour l'inclure dans les noms de fichiers de sortie
        filename_base = os.path.splitext(os.path.basename(args.data_file))[0]
        
        fig1 = plot_range_profile(
            range_profile, 
            range_axis_no_padding, 
            title=f"Profil de distance - {filename_base} - Frame {args.frame}",
            save_path=f"{args.output_dir}/range_profile_frame{args.frame}_{filename_base}.pdf"
        )
        
        # 2. Range-Doppler map avec la nouvelle fonction qui retourne aussi les axes
        print("Génération de la Range-Doppler Map avec zero-padding...")
        try:
            range_doppler_map, range_axis, velocity_axis = generate_range_doppler_map_with_axes(
                radar_data, 
                params,
                window_type=args.window_type,
                range_padding_factor=args.range_padding,
                doppler_padding_factor=args.doppler_padding,
                apply_2d_filter=args.apply_2d_filter
            )
            
            print(f"Shape de la Range-Doppler Map après zero-padding: {range_doppler_map.shape}")
            print(f"Range axis: de {range_axis[0]:.2f}m à {range_axis[-1]:.2f}m, {len(range_axis)} points")
            print(f"Velocity axis: de {velocity_axis[0]:.2f}m/s à {velocity_axis[-1]:.2f}m/s, {len(velocity_axis)} points")
            
        except Exception as e:
            print(f"Erreur lors de la génération de la Range-Doppler map: {str(e)}")
            print("Utilisation de la méthode standard sans zero-padding...")
            from src.signal_processing.range_doppler_map import generate_range_doppler_map, calculate_velocity_axis
            range_doppler_map = generate_range_doppler_map(radar_data, window_type=args.window_type)
            range_axis = range_axis_no_padding
            velocity_axis = calculate_velocity_axis(params)
        
        # Application de la détection CFAR si demandée
        detections = None
        if args.detect_targets:
            print("Application de la détection CFAR...")
            try:
                detections = apply_cfar_detector(
                    range_doppler_map, 
                    guard_cells=(2, 4), 
                    training_cells=(4, 8),
                    threshold_factor=15.0
                )
                num_targets = np.sum(detections)
                print(f"Nombre de cibles détectées: {num_targets}")
            except Exception as e:
                print(f"Erreur lors de la détection CFAR: {str(e)}")
        
        # Si des cibles sont détectées, les afficher sur la carte
        if detections is not None and np.sum(detections) > 0:
            plt.figure(figsize=(12, 9))
            
            # Limiter la plage dynamique
            vmax = np.max(range_doppler_map)
            vmin = vmax - args.dynamic_range
            
            plt.pcolormesh(range_axis, velocity_axis, range_doppler_map, 
                          cmap='jet', vmin=vmin, vmax=vmax, shading='auto')
            det_doppler_idx, det_range_idx = np.where(detections)
            
            # Indices vers valeurs physiques
            det_ranges = [range_axis[idx] if idx < len(range_axis) else 0 for idx in det_range_idx]
            det_velocities = [velocity_axis[idx] if idx < len(velocity_axis) else 0 for idx in det_doppler_idx]
            
            plt.scatter(det_ranges, det_velocities, c='r', marker='x', s=50, label='Détections')
            plt.legend()
            
            plt.colorbar(label='Magnitude (dB)')
            plt.xlabel('Distance (m)')
            plt.ylabel('Vitesse (m/s)')
            filename_base = os.path.splitext(os.path.basename(args.data_file))[0]
            plt.title(f'Carte Range-Doppler avec détections - {filename_base} - Frame {args.frame}')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.tight_layout()
            detect_path = f"{args.output_dir}/range_doppler_detections_frame{args.frame}_{filename_base}.pdf"
            plt.savefig(detect_path)
            print(f"Carte Range-Doppler avec détections sauvegardée dans {detect_path}")
        else:
            # Dans plotting.py
            print("Pas de détection CFAR, visualisation normale...")
            filename_base = os.path.splitext(os.path.basename(args.data_file))[0]
            fig2 = plot_range_doppler(
                range_doppler_map, 
                range_axis, 
                velocity_axis, 
                title=f"Range-Doppler map - {filename_base} (Padding {args.range_padding}x/{args.doppler_padding}x) - Frame {args.frame}",
                dynamic_range=args.dynamic_range,
                save_path=f"{args.output_dir}/range_doppler_map_frame{args.frame}_{filename_base}.pdf"
            )
        
        # 3. Visualisation 3D
        print("Génération de la visualisation 3D...")
        filename_base = os.path.splitext(os.path.basename(args.data_file))[0]
        fig3 = visualize_3d_range_doppler(
            range_doppler_map, 
            range_axis, 
            velocity_axis, 
            title=f"Visualisation 3D Range-Doppler - {filename_base} - Frame {args.frame} (Padding {args.range_padding}x/{args.doppler_padding}x)",
            save_path=f"{args.output_dir}/range_doppler_3d_frame{args.frame}_{filename_base}.pdf"
        )
        plt.show()
        print(f"\nToutes les visualisations ont été enregistrées dans le répertoire {args.output_dir}")
        
    except Exception as e:
        print(f"Erreur: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()