import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import matplotlib
matplotlib.use('Agg')  # Pour le rendu sans interface graphique

# Ajouter le répertoire parent au chemin d'accès
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import des fonctions des packages dans src
from src.data_handling.data_loader import load_fmcw_data, extract_frame, reshape_to_chirps
from src.signal_processing.range_doppler_map import (
    generate_range_doppler_map_with_axes,
    apply_cfar_detector
)

def main():
    """Fonction principale pour l'animation des données FMCW"""
    
    # Configuration des paramètres
    parser = argparse.ArgumentParser(description='Animation des données FMCW')
    parser.add_argument('--data-file', type=str, required=False,
                       help='Chemin vers le fichier de données')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Répertoire de sortie pour les visualisations')
    parser.add_argument('--detect-targets', action='store_true',
                       help='Activer la détection de cibles avec CFAR')
    parser.add_argument('--dynamic-range', type=int, default=40,
                       help='Plage dynamique en dB pour la visualisation')
    parser.add_argument('--range-padding', type=int, default=4,
                       help='Facteur de zero-padding pour l\'axe distance')
    parser.add_argument('--doppler-padding', type=int, default=4,
                       help='Facteur de zero-padding pour l\'axe Doppler')
    parser.add_argument('--window-type', type=str, default='hann',
                       help='Type de fenêtre à appliquer (hann, hamming, blackman, etc.)')
    parser.add_argument('--fps', type=int, default=10,
                       help='Images par seconde pour l\'animation')
    parser.add_argument('--start-frame', type=int, default=0,
                       help='Frame de départ')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Nombre maximal de frames à traiter')
    parser.add_argument('--view-type', type=str, default='2d',
                       choices=['2d', '3d'],
                       help='Type de visualisation (2d pour range-doppler, 3d pour vue 3D)')
    args = parser.parse_args()
    
    # Si aucun fichier n'est spécifié dans les arguments, utiliser la variable d'environnement
    if args.data_file is None:
        if 'RADAR_DATA_FILE' in os.environ:
            args.data_file = os.environ['RADAR_DATA_FILE']
        else:
            args.data_file = 'data/MS1-FMCW.npz'  # Valeur par défaut
    
    # Extraire le nom du fichier sans extension et sans chemin pour les sorties
    filename_base = os.path.splitext(os.path.basename(args.data_file))[0]
    
    # Si le répertoire de sortie n'est pas spécifié, créer un sous-répertoire basé sur le nom du fichier
    if args.output_dir is None:
        args.output_dir = os.path.join('output', filename_base)
    
    # Création du répertoire de sortie s'il n'existe pas
    os.makedirs(args.output_dir, exist_ok=True)
    
    # check que le fichier existe
    if not os.path.exists(args.data_file):
        print(f"Erreur: Le fichier {args.data_file} n'existe pas.")
        print(f"Chemin absolu: {os.path.abspath(args.data_file)}")
        return
    
    print(f"Chargement des données depuis {args.data_file}...")
    
    try:
        # Charger les données radar
        data, params = load_fmcw_data(args.data_file)
        
        # Affichage des informations sur les données
        print("\nInformations sur les données:")
        print(f"Nombre de frames: {data.shape[0]}")
        print(f"Nombre de canaux: {data.shape[1]}")
        print(f"Nombre total d'échantillons par frame: {data.shape[2]}")
        
        total_frames = data.shape[0]
        # Limiter le nombre de frames si spécifié
        if args.max_frames is not None:
            end_frame = min(args.start_frame + args.max_frames, total_frames)
        else:
            end_frame = total_frames
        
        print(f"Animation des frames {args.start_frame} à {end_frame-1} (total: {end_frame-args.start_frame})")
        
        # Préparer l'animation
        print("Préparation de l'animation...")
        
        # Variables pour stocker les éléments de visualisation
        mesh = None
        scatter = None
        surf = None
        title = None
        
        # Définir la figure et les axes en fonction du type de visualisation
        if args.view_type == '2d':
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.set_xlabel('Distance (m)')
            ax.set_ylabel('Vitesse (m/s)')
            ax.grid(True, linestyle='--', alpha=0.5)
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Pour la barre de couleur
            
            # Calculer la première frame pour initialiser les limites d'axes
            complex_data = extract_frame(data, frame_index=args.start_frame, channel_indices=(0, 1))
            radar_data = reshape_to_chirps(complex_data, params, 2)
            rdm, range_axis, velocity_axis = generate_range_doppler_map_with_axes(
                radar_data, 
                params,
                window_type=args.window_type,
                range_padding_factor=args.range_padding,
                doppler_padding_factor=args.doppler_padding
            )
            
            # Définir les limites d'axes basées sur les paramètres radar
            ax.set_xlim(range_axis[0], range_axis[-1])
            ax.set_ylim(velocity_axis[0], velocity_axis[-1])
            
            # Initialiser le titre
            title = ax.set_title(f"Range-Doppler Map - Frame {args.start_frame}/{end_frame-1}")
            
        elif args.view_type == '3d':
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('Distance (m)')
            ax.set_ylabel('Vitesse (m/s)')
            ax.set_zlabel('Magnitude (dB)')
            
            # Calculer la première frame pour initialiser les limites d'axes
            complex_data = extract_frame(data, frame_index=args.start_frame, channel_indices=(0, 1))
            radar_data = reshape_to_chirps(complex_data, params, 2)
            rdm, range_axis, velocity_axis = generate_range_doppler_map_with_axes(
                radar_data, 
                params,
                window_type=args.window_type,
                range_padding_factor=args.range_padding,
                doppler_padding_factor=args.doppler_padding
            )
            
            # Définir les limites d'axes basées sur les paramètres radar
            ax.set_xlim(range_axis[0], range_axis[-1])
            ax.set_ylim(velocity_axis[0], velocity_axis[-1])
            
            # Estimer la plage de valeurs Z
            vmax = np.max(rdm)
            vmin = vmax - args.dynamic_range
            ax.set_zlim(vmin, vmax + 5)  # Ajouter un peu de marge sur le dessus
            
            # Initialiser le titre
            title = ax.set_title(f"3D Range-Doppler - Frame {args.start_frame}/{end_frame-1}")
        
        # Fonction d'initialisation pour FuncAnimation
        def init():
            return title,
        
        # Fonction de mise à jour pour FuncAnimation
        def update(frame_idx):
            nonlocal mesh, scatter, surf, title
            
            frame_idx = frame_idx + args.start_frame
            if frame_idx >= end_frame:
                return title,
            
            # Mettre à jour le titre
            title.set_text(f"Range-Doppler - {filename_base} - Frame {frame_idx}/{end_frame-1}")
            
            # Extraire et traiter les données de la frame
            complex_data = extract_frame(data, frame_index=frame_idx, channel_indices=(0, 1))
            radar_data = reshape_to_chirps(complex_data, params, 2)
            
            # Générer la Range-Doppler map
            try:
                rdm, range_axis, velocity_axis = generate_range_doppler_map_with_axes(
                    radar_data, 
                    params,
                    window_type=args.window_type,
                    range_padding_factor=args.range_padding,
                    doppler_padding_factor=args.doppler_padding
                )
            except Exception as e:
                print(f"Erreur lors de la génération de la Range-Doppler map pour la frame {frame_idx}: {str(e)}")
                return title,
            
            # Limiter la plage dynamique
            vmax = np.max(rdm)
            vmin = vmax - args.dynamic_range
            
            if args.view_type == '2d':
                # Mettre à jour la visualisation 2D
                if mesh is not None:
                    mesh.remove()
                    
                # Créer un nouveau mesh avec les nouvelles données
                mesh = ax.pcolormesh(range_axis, velocity_axis, rdm, 
                                    cmap='jet', vmin=vmin, vmax=vmax, shading='auto')
                
                # Détection de cibles si activée
                if args.detect_targets:
                    if scatter is not None:
                        scatter.remove()
                        
                    try:
                        detections = apply_cfar_detector(
                            rdm, 
                            guard_cells=(2, 4), 
                            training_cells=(4, 8),
                            threshold_factor=15.0
                        )
                        
                        if np.sum(detections) > 0:
                            det_doppler_idx, det_range_idx = np.where(detections)
                            
                            # Indices vers valeurs physiques
                            det_ranges = [range_axis[idx] if idx < len(range_axis) else 0 for idx in det_range_idx]
                            det_velocities = [velocity_axis[idx] if idx < len(velocity_axis) else 0 for idx in det_doppler_idx]
                            
                            scatter = ax.scatter(det_ranges, det_velocities, c='r', marker='x', s=50)
                    except Exception as e:
                        print(f"Erreur lors de la détection CFAR pour la frame {frame_idx}: {str(e)}")
                
                # Mettre à jour la barre de couleur
                if frame_idx == args.start_frame:
                    plt.colorbar(mesh, cax=cbar_ax, label='Magnitude (dB)')
                
                return title, mesh
            
            elif args.view_type == '3d':
                # Mettre à jour la visualisation 3D
                ax.clear()
                
                # Recréer la grille pour le tracé 3D
                X, Y = np.meshgrid(range_axis, velocity_axis)
                
                # Créer la surface 3D
                surf = ax.plot_surface(X, Y, rdm, cmap='viridis', 
                                      vmin=vmin, vmax=vmax,
                                      linewidth=0, antialiased=True)
                
                # Réinitialiser les limites des axes et les étiquettes
                ax.set_xlim(range_axis[0], range_axis[-1])
                ax.set_ylim(velocity_axis[0], velocity_axis[-1])
                ax.set_zlim(vmin, vmax + 5)
                ax.set_xlabel('Distance (m)')
                ax.set_ylabel('Vitesse (m/s)')
                ax.set_zlabel('Magnitude (dB)')
                
                # Réinitialiser le titre
                title = ax.set_title(f"3D Range-Doppler - {filename_base} - Frame {frame_idx}/{end_frame-1}")
                
                # Ajouter une ligne à vitesse 0 pour référence
                zero_vel_idx = np.abs(velocity_axis).argmin()
                ax.plot(range_axis, [velocity_axis[zero_vel_idx]]*len(range_axis), 
                       [vmin]*len(range_axis), 'r-', lw=2)
                
                return title, surf
        
        # Créer l'animation
        num_frames = end_frame - args.start_frame
        ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, 
                           interval=1000/args.fps, blit=True)
        
        # Enregistrer l'animation
        output_file = os.path.join(args.output_dir, f"anim_{args.view_type}_{filename_base}.mp4")
        print(f"Enregistrement de l'animation dans {output_file}...")
        
        # Utiliser ffmpeg pour une meilleure qualité
        ani.save(output_file, writer='ffmpeg', fps=args.fps, dpi=200,
                extra_args=['-vcodec', 'h264', '-pix_fmt', 'yuv420p'])
        
        print(f"Animation terminée et enregistrée dans {output_file}")
        
    except Exception as e:
        print(f"Erreur: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()