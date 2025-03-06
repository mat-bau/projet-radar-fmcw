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
    generate_range_profile,
    generate_range_doppler_map_with_axes,
    apply_cfar_detector,
    remove_static_components,
    apply_clutter_threshold,
    estimate_target_parameters
)
from src.visualization.plotting import plot_range_doppler, visualize_3d_range_doppler, create_combined_visualization

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
    # S'assurer que les données ont la même forme
    if data.shape != background_data.shape:
        raise ValueError(f"Les dimensions des données ({data.shape}) et du fond ({background_data.shape}) ne correspondent pas")
    
    # Les données sont complexes, donc on fait une soustraction complexe
    return data - background_data

def main():
    """Fonction principale pour l'animation des données FMCW avec différents types de visualisation"""
    
    # Configuration des paramètres
    parser = argparse.ArgumentParser(description='Animation des données FMCW avec visualisation')
    parser.add_argument('--data-file', type=str, required=False,
                      help='Chemin vers le fichier de données')
    parser.add_argument('--background-file', type=str, required=False,
                      help='Chemin vers le fichier de données de fond pour la soustraction')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Répertoire de sortie pour les visualisations')
    parser.add_argument('--detect-targets', action='store_true',
                       help='Activer la détection de cibles avec CFAR')
    parser.add_argument('--dynamic-range', type=int, default=20,
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
    parser.add_argument('--remove-static', action='store_true',
                       help='Supprimer les composantes statiques de la Range-Doppler map')
    parser.add_argument('--clutter-threshold', type=float, default=0.0,
                       help='Seuil pour la suppression du clutter (0.0 = désactivé)')
    parser.add_argument('--view-type', type=str, default='combined',
                       choices=['combined', '2d', '3d'],
                       help='Type de visualisation (combined pour la vue combinée, 2d pour range-doppler uniquement, 3d pour vue 3D)')
    parser.add_argument('--cfar-method', type=str, default='CA', choices=['CA', 'OS'],
                       help='Méthode CFAR à utiliser (CA pour Cell Averaging, OS pour Ordered Statistics)')
    parser.add_argument('--cfar-threshold', type=float, default=15.0,
                       help='Facteur de seuil en dB pour la détection CFAR')
    parser.add_argument('--cfar-percentile', type=int, default=75,
                       help='Percentile à utiliser pour la méthode CFAR OS (0-100)')
    parser.add_argument('--apply-2d-filter', action='store_true',
                       help='Appliquer un filtre médian 2D pour réduire le bruit')
    parser.add_argument('--filter-size', type=int, default=3,
                       help='Taille du noyau pour le filtre médian 2D')
    parser.add_argument('--estimate-targets', action='store_true',
                       help='Estimer les paramètres précis des cibles par interpolation')
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
    
    # Check que le fichier existe
    if not os.path.exists(args.data_file):
        print(f"Erreur: Le fichier {args.data_file} n'existe pas.")
        print(f"Chemin absolu: {os.path.abspath(args.data_file)}")
        return
    
    print(f"Chargement des données depuis {args.data_file}...")
    
    # Charger les données radar
    data, params = load_fmcw_data(args.data_file)
    
    # Charger les données de fond si spécifiées
    background_data = None
    if args.background_file:
        if not os.path.exists(args.background_file):
            print(f"Attention: Le fichier de fond {args.background_file} n'existe pas. Aucune soustraction de fond ne sera effectuée.")
        else:
            print(f"Chargement des données de fond depuis {args.background_file}...")
            background_data_array, _ = load_fmcw_data(args.background_file)
            
            # Vérifier que les dimensions correspondent
            if background_data_array.shape != data.shape:
                print(f"Attention: Les dimensions des données de fond ({background_data_array.shape}) ne correspondent pas aux données radar ({data.shape}). Aucune soustraction de fond ne sera effectuée.")
                background_data = None
            else:
                background_data = background_data_array
    
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
    
    total_frames = data.shape[0]
    
    # Limiter le nombre de frames si spécifié
    if args.max_frames is not None:
        end_frame = min(args.start_frame + args.max_frames, total_frames)
    else:
        end_frame = total_frames
    
    print(f"Animation des frames {args.start_frame} à {end_frame-1} (total: {end_frame-args.start_frame})")
    
    # Préparer l'animation
    print(f"Préparation de l'animation ({args.view_type})...")
    # Initialiser la frame de départ pour l'initialisation
    frame_idx = args.start_frame
    
    # Extraire et traiter les données de la première frame
    complex_data = extract_frame(data, frame_index=frame_idx, channel_indices=(0, 1))
    
    # Soustraire le fond si spécifié
    if background_data is not None:
        bg_complex_data = extract_frame(background_data, frame_index=frame_idx, channel_indices=(0, 1))
        complex_data = subtract_background(complex_data, bg_complex_data)
    
    # Transformer les données en 2D (chirps x échantillons)
    radar_data = reshape_to_chirps(complex_data, params, "without_pause")
    
    # Générer le profil de distance
    range_profile = generate_range_profile(radar_data, window_type=args.window_type)
    
    # Générer la Range-Doppler map avec les axes
    range_doppler_map, range_axis, velocity_axis = generate_range_doppler_map_with_axes(
        radar_data, 
        params,
        window_type=args.window_type,
        range_padding_factor=args.range_padding,
        doppler_padding_factor=args.doppler_padding,
        apply_2d_filter=args.apply_2d_filter,
        kernel_size=args.filter_size
    )
    
    # Supprimer les composantes statiques si demandé
    if args.remove_static:
        range_doppler_map = remove_static_components(range_doppler_map, linear_scale=False)
    
    # Appliquer un seuil de clutter si spécifié
    if args.clutter_threshold > 0:
        range_doppler_map = apply_clutter_threshold(range_doppler_map, args.clutter_threshold)
    
    # Variables pour stocker les éléments de visualisation
    mesh = None
    scatter = None
    surf = None
    title = None
    
    # Initialiser la figure et les axes en fonction du type de visualisation
    if args.view_type == 'combined':
        # Créer la figure et les axes avec la fonction de visualisation combinée
        fig, (ax_range, ax_velocity, ax_rdm) = create_combined_visualization(
            range_doppler_map, 
            range_profile, 
            range_axis, 
            velocity_axis, 
            title=f"Analyse FMCW - {filename_base} - Frame {frame_idx}/{end_frame-1}",
            dynamic_range=args.dynamic_range
        )
        
        # Variables pour stocker les éléments de visualisation à mettre à jour
        range_line = ax_range.get_lines()[0]
        velocity_line = ax_velocity.get_lines()[0]
        rdm_mesh = ax_rdm.collections[0]
        title = fig.texts[0]
        scatter = None
        
        # Fonction d'initialisation pour FuncAnimation
        def init():
            return range_line, velocity_line, rdm_mesh, title
        
        # Fonction de mise à jour pour FuncAnimation
        def update(frame_offset):
            nonlocal scatter, range_line, velocity_line, rdm_mesh, title
            
            frame_idx = args.start_frame + frame_offset
            if frame_idx >= end_frame:
                return range_line, velocity_line, rdm_mesh, title
            
            # Mettre à jour le titre
            title.set_text(f"Analyse FMCW - {filename_base} - Frame {frame_idx}/{end_frame-1}")
            
            # Extraire et traiter les données de la frame actuelle
            complex_data = extract_frame(data, frame_index=frame_idx, channel_indices=(0, 1))
            
            # Soustraire le fond si spécifié
            if background_data is not None:
                bg_complex_data = extract_frame(background_data, frame_index=frame_idx, channel_indices=(0, 1))
                complex_data = subtract_background(complex_data, bg_complex_data)
            
            radar_data = reshape_to_chirps(complex_data, params, "without_pause")
            
            # Mettre à jour le profil de distance
            range_profile = generate_range_profile(radar_data, window_type=args.window_type)
            range_line.set_ydata(range_profile[:len(range_axis)])
            
            # Générer la Range-Doppler map
            try:
                range_doppler_map, _, _ = generate_range_doppler_map_with_axes(
                    radar_data, 
                    params,
                    window_type=args.window_type,
                    range_padding_factor=args.range_padding,
                    doppler_padding_factor=args.doppler_padding,
                    apply_2d_filter=args.apply_2d_filter,
                    kernel_size=args.filter_size
                )
                
                # Supprimer les composantes statiques si demandé
                if args.remove_static:
                    range_doppler_map = remove_static_components(range_doppler_map, linear_scale=False)
                
                # Appliquer un seuil de clutter si spécifié
                if args.clutter_threshold > 0:
                    range_doppler_map = apply_clutter_threshold(range_doppler_map, args.clutter_threshold)
                
                # Mettre à jour le profil de vitesse
                velocity_profile = np.mean(range_doppler_map, axis=1)
                velocity_line.set_ydata(velocity_profile[:len(velocity_axis)])
                
                # Mettre à jour la Range-Doppler map
                vmax = np.max(range_doppler_map)
                vmin = vmax - args.dynamic_range
                
                rdm_mesh.set_array(range_doppler_map[:len(velocity_axis), :len(range_axis)].ravel())
                rdm_mesh.set_clim(vmin=vmin, vmax=vmax)
                
                # Détection de cibles si activée
                if args.detect_targets:
                    if scatter is not None:
                        scatter.remove()
                    
                    try:
                        detections = apply_cfar_detector(
                            range_doppler_map, 
                            guard_cells=(2, 4), 
                            training_cells=(4, 8),
                            threshold_factor=args.cfar_threshold,
                            cfar_method=args.cfar_method,
                            percentile=args.cfar_percentile
                        )
                        
                        if np.sum(detections) > 0:
                            if args.estimate_targets:
                                # Utiliser les positions estimées des cibles
                                targets = estimate_target_parameters(range_doppler_map, detections, range_axis, velocity_axis)
                                target_ranges = [target['range'] for target in targets]
                                target_velocities = [target['velocity'] for target in targets]
                                scatter = ax_rdm.scatter(target_ranges, target_velocities, c='r', marker='x', s=50)
                            else:
                                # Utiliser les positions brutes des détections CFAR
                                det_doppler_idx, det_range_idx = np.where(detections)
                                
                                # Indices vers valeurs physiques
                                det_ranges = [range_axis[idx] if idx < len(range_axis) else 0 for idx in det_range_idx]
                                det_velocities = [velocity_axis[idx] if idx < len(velocity_axis) else 0 for idx in det_doppler_idx]
                                
                                scatter = ax_rdm.scatter(det_ranges, det_velocities, c='r', marker='x', s=50)
                            return range_line, velocity_line, rdm_mesh, title, scatter
                    
                    except Exception as e:
                        print(f"Erreur lors de la détection CFAR pour la frame {frame_idx}: {str(e)}")
                
            except Exception as e:
                print(f"Erreur lors de la génération de la Range-Doppler map pour la frame {frame_idx}: {str(e)}")
            
            return range_line, velocity_line, rdm_mesh, title
        
    elif args.view_type == '2d':
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Vitesse (m/s)')
        ax.grid(True, linestyle='--', alpha=0.5)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Pour la barre de couleur
        
        # Définir les limites d'axes basées sur les paramètres radar
        ax.set_xlim(range_axis[0], range_axis[-1])
        ax.set_ylim(velocity_axis[0], velocity_axis[-1])
        
        # Initialiser le titre
        title = ax.set_title(f"Range-Doppler Map - {filename_base} - Frame {frame_idx}/{end_frame-1}")
        
        # Limiter la plage dynamique
        vmax = np.max(range_doppler_map)
        vmin = vmax - args.dynamic_range
        
        # Créer le mesh initial
        mesh = ax.pcolormesh(range_axis, velocity_axis, range_doppler_map[:len(velocity_axis), :len(range_axis)], 
                            cmap='jet', vmin=vmin, vmax=vmax, shading='auto')
        plt.colorbar(mesh, cax=cbar_ax, label='Magnitude (dB)')
        
        # Fonction d'initialisation pour FuncAnimation
        def init():
            return title, mesh
        
        # Fonction de mise à jour pour FuncAnimation
        def update(frame_offset):
            nonlocal mesh, scatter, title
            
            frame_idx = args.start_frame + frame_offset
            if frame_idx >= end_frame:
                return title, mesh
            
            # Mettre à jour le titre
            title.set_text(f"Range-Doppler Map - {filename_base} - Frame {frame_idx}/{end_frame-1}")
            
            # Extraire et traiter les données de la frame
            complex_data = extract_frame(data, frame_index=frame_idx, channel_indices=(0, 1))
            
            # Soustraire le fond si spécifié
            if background_data is not None:
                bg_complex_data = extract_frame(background_data, frame_index=frame_idx, channel_indices=(0, 1))
                complex_data = subtract_background(complex_data, bg_complex_data)
            
            radar_data = reshape_to_chirps(complex_data, params, "without_pause")
            
            # Générer la Range-Doppler map
            try:
                range_doppler_map, _, _ = generate_range_doppler_map_with_axes(
                    radar_data, 
                    params,
                    window_type=args.window_type,
                    range_padding_factor=args.range_padding,
                    doppler_padding_factor=args.doppler_padding,
                    apply_2d_filter=args.apply_2d_filter,
                    kernel_size=args.filter_size
                )
                
                # Supprimer les composantes statiques si demandé
                if args.remove_static:
                    range_doppler_map = remove_static_components(range_doppler_map, linear_scale=False)
                
                # Appliquer un seuil de clutter si spécifié
                if args.clutter_threshold > 0:
                    range_doppler_map = apply_clutter_threshold(range_doppler_map, args.clutter_threshold)
                
                # Limiter la plage dynamique
                vmax = np.max(range_doppler_map)
                vmin = vmax - args.dynamic_range
                
                # Mettre à jour la visualisation
                mesh.set_array(range_doppler_map[:len(velocity_axis), :len(range_axis)].ravel())
                mesh.set_clim(vmin=vmin, vmax=vmax)
                
                # Détection de cibles si activée
                if args.detect_targets:
                    if scatter is not None:
                        scatter.remove()
                        
                    try:
                        detections = apply_cfar_detector(
                            range_doppler_map, 
                            guard_cells=(2, 4), 
                            training_cells=(4, 8),
                            threshold_factor=args.cfar_threshold,
                            cfar_method=args.cfar_method,
                            percentile=args.cfar_percentile
                        )
                        
                        if np.sum(detections) > 0:
                            if args.estimate_targets:
                                # Utiliser les positions estimées des cibles
                                targets = estimate_target_parameters(range_doppler_map, detections, range_axis, velocity_axis)
                                target_ranges = [target['range'] for target in targets]
                                target_velocities = [target['velocity'] for target in targets]
                                scatter = ax.scatter(target_ranges, target_velocities, c='r', marker='x', s=50)
                            else:
                                # Utiliser les positions brutes des détections CFAR
                                det_doppler_idx, det_range_idx = np.where(detections)
                                
                                # Indices vers valeurs physiques
                                det_ranges = [range_axis[idx] if idx < len(range_axis) else 0 for idx in det_range_idx]
                                det_velocities = [velocity_axis[idx] if idx < len(velocity_axis) else 0 for idx in det_doppler_idx]
                                
                                scatter = ax.scatter(det_ranges, det_velocities, c='r', marker='x', s=50)
                            return title, mesh, scatter
                    except Exception as e:
                        print(f"Erreur lors de la détection CFAR pour la frame {frame_idx}: {str(e)}")
                
            except Exception as e:
                print(f"Erreur lors de la génération de la Range-Doppler map pour la frame {frame_idx}: {str(e)}")
            
            return title, mesh
        
    elif args.view_type == '3d':
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Vitesse (m/s)')
        ax.set_zlabel('Magnitude (dB)')
        
        # Définir les limites d'axes basées sur les paramètres radar
        ax.set_xlim(range_axis[0], range_axis[-1])
        ax.set_ylim(velocity_axis[0], velocity_axis[-1])
        
        # Estimer la plage de valeurs Z
        vmax = np.max(range_doppler_map)
        vmin = vmax - args.dynamic_range
        ax.set_zlim(vmin, vmax + 5)  # Ajouter un peu de marge sur le dessus
        
        # Initialiser le titre
        title = ax.set_title(f"3D Range-Doppler - {filename_base} - Frame {frame_idx}/{end_frame-1}")
        
        # Créer la grille pour le tracé 3D
        X, Y = np.meshgrid(range_axis, velocity_axis)
        Z = range_doppler_map[:len(velocity_axis), :len(range_axis)]
        
        # Créer la surface 3D initiale
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', 
                               vmin=vmin, vmax=vmax,
                               linewidth=0, antialiased=True)
        
        # Fonction d'initialisation pour FuncAnimation
        def init():
            return title,
        
        # Fonction de mise à jour pour FuncAnimation
        def update(frame_offset):
            nonlocal surf, title
            
            frame_idx = args.start_frame + frame_offset
            if frame_idx >= end_frame:
                return title,
            
            # Mettre à jour le titre
            title.set_text(f"3D Range-Doppler - {filename_base} - Frame {frame_idx}/{end_frame-1}")
            
            # Extraire et traiter les données de la frame
            complex_data = extract_frame(data, frame_index=frame_idx, channel_indices=(0, 1))
            
            # Soustraire le fond si spécifié
            if background_data is not None:
                bg_complex_data = extract_frame(background_data, frame_index=frame_idx, channel_indices=(0, 1))
                complex_data = subtract_background(complex_data, bg_complex_data)
            
            radar_data = reshape_to_chirps(complex_data, params, "without_pause")
            
            # Générer la Range-Doppler map
            try:
                range_doppler_map, _, _ = generate_range_doppler_map_with_axes(
                    radar_data, 
                    params,
                    window_type=args.window_type,
                    range_padding_factor=args.range_padding,
                    doppler_padding_factor=args.doppler_padding,
                    apply_2d_filter=args.apply_2d_filter,
                    kernel_size=args.filter_size
                )
                
                # Supprimer les composantes statiques si demandé
                if args.remove_static:
                    range_doppler_map = remove_static_components(range_doppler_map, linear_scale=False)
                
                # Appliquer un seuil de clutter si spécifié
                if args.clutter_threshold > 0:
                    range_doppler_map = apply_clutter_threshold(range_doppler_map, args.clutter_threshold)
                
                # Mettre à jour la visualisation 3D
                ax.clear()
                
                # Recréer la grille pour le tracé 3D
                X, Y = np.meshgrid(range_axis, velocity_axis)
                Z = range_doppler_map[:len(velocity_axis), :len(range_axis)]
                
                # Limiter la plage dynamique
                vmax = np.max(Z)
                vmin = vmax - args.dynamic_range
                
                # Créer la surface 3D
                surf = ax.plot_surface(X, Y, Z, cmap='viridis', 
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
                
            except Exception as e:
                print(f"Erreur lors de la génération de la Range-Doppler map pour la frame {frame_idx}: {str(e)}")
            
            return title, surf
    
    # Créer l'animation
    num_frames = end_frame - args.start_frame
    ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, 
                        interval=1000/args.fps, blit=True)
    
    # Enregistrer l'animation
    view_type_str = args.view_type
    output_file = os.path.join(args.output_dir, f"anim_{view_type_str}_{filename_base}.mp4")
    print(f"Enregistrement de l'animation dans {output_file}...")
    
    # Utiliser ffmpeg pour une meilleure qualité
    ani.save(output_file, writer='ffmpeg', fps=args.fps, dpi=200,
             extra_args=['-vcodec', 'h264', '-pix_fmt', 'yuv420p'])
    
    print(f"Animation terminée et enregistrée dans {output_file}")

if __name__ == "__main__":
    main()