import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy import signal

# Ajouter le répertoire parent au chemin d'accès
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import des fonctions des packages dans src
from src.data_handling.data_loader import load_fmcw_data, extract_frame, reshape_to_chirps, subtract_background
from src.signal_processing.range_doppler_map import (
    generate_range_profile, 
    generate_range_doppler_map_with_axes,
    apply_cfar_detector,
    remove_static_components,
    apply_clutter_threshold,
    estimate_target_parameters,
    estimate_imbalance_parameters,
    correct_imbalance,
    calculate_imbalance_metrics
)
from src.visualization.plotting import (
    plot_range_profile, 
    plot_range_doppler,
    visualize_3d_range_doppler,
    plot_spectrogram
)


def main():
    """Fonction principale pour l'analyse des données FMCW"""
    
    # DEBUT DU PARSE DES ARGUMENTS
    parser = argparse.ArgumentParser(description='Analyse des données FMCW')
    parser.add_argument('--data-file', type=str, required=False,
                       help='Chemin vers le fichier de données')
    parser.add_argument('--background-file', type=str, required=False,
                       help='Chemin vers le fichier de données de fond pour la soustraction') # a changer dans le makefile
    parser.add_argument('--frame', type=int, default=0,
                       help='Indice de la frame à analyser')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Répertoire de sortie pour les visualisations')
    parser.add_argument('--generate-spectrogram', action='store_true',
                    help='Générer un spectrogramme des données radar')
    parser.add_argument('--detect-targets', action='store_true',
                       help='Activer la détection de cibles avec CFAR')
    parser.add_argument('--dynamic-range', type=int, default=20, # a changer pour voir plus en profondeur!!!
                       help='Plage dynamique en dB pour la visualisation') 
    parser.add_argument('--range-padding', type=int, default=10, # change pour meilleure qualité
                       help='Facteur de zero-padding pour l\'axe distance')
    parser.add_argument('--doppler-padding', type=int, default=10, # change pour meilleure qualité
                       help='Facteur de zero-padding pour l\'axe Doppler')
    parser.add_argument('--window-type', type=str, default='hann',
                       help='Type de fenêtre à appliquer (hann, hamming, blackman, etc.)')
    parser.add_argument('--remove-static', action='store_true',
                       help='Supprimer les composantes statiques de la Range-Doppler map')
    parser.add_argument('--clutter-threshold', type=float, default=0.0,
                       help='Seuil pour la suppression du clutter (0.0 = désactivé)')
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
    parser.add_argument('--estimate-iq-imbalance', action='store_true',
                   help='Estimer les coefficients de déséquilibre IQ')
    parser.add_argument('--correct-iq', action='store_true',
                    help='Appliquer la correction du déséquilibre IQ')
    parser.add_argument('--analyze-iq-imbalance', action='store_true',
                    help='Analyser et corriger le déséquilibre IQ')
    parser.add_argument('--compare-channels', action='store_true',
                   help='Comparer les cartes Range-Doppler des canaux déséquilibrés (0,1) et équilibrés (2,3)')
    # FIN DU PARSE DES ARGUMENTS
    args = parser.parse_args()
    
    # Si aucun fichier n'est spécifié dans les arguments, utiliser la variable d'environnement
    if args.data_file is None:
        if 'RADAR_DATA_FILE' in os.environ:
            args.data_file = os.environ['RADAR_DATA_FILE']
        else:
            args.data_file = 'data/MS1-FMCW.npz'  # Par défaut

    filename_base = os.path.splitext(os.path.basename(args.data_file))[0] # nom du fichier sans les extensions

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
        
        # Charger les données de fond si spécifiées
        background_data = None
        if args.background_file:
            if not os.path.exists(args.background_file):
                print(f"Attention: Le fichier de fond {args.background_file} n'existe pas. Aucune soustraction de fond ne sera effectuée.")
            else:
                print(f"Chargement des données de fond depuis {args.background_file}...")
                background_data_array, _ = load_fmcw_data(args.background_file)
                
                # Vérifier que les dimensions correspondent (nb de frames)
                if background_data_array.shape != data.shape:
                    print(f"Attention: Les dimensions des données de fond ({background_data_array.shape}) ne correspondent pas aux données radar ({data.shape}). Aucune soustraction de fond ne sera effectuée.")
                    background_data = None
                else:
                    background_data = background_data_array
        
        # Check que la frame demandée existe
        if args.frame >= data.shape[0]:
            print(f"Erreur: La frame {args.frame} n'existe pas. Le fichier contient {data.shape[0]} frames.")
            args.frame = 0
            print(f"Utilisation de la frame 0 à la place.")
        
        # Qlq infos sur les données
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
        
        ###########################
        # Démarage de l'analyse #
        ###########################
        
        # Extraire les données d'une frame (I1 et Q1)
        complex_data = extract_frame(data, frame_index=args.frame, channel_indices=(0, 1))
        ref_data = extract_frame(data, frame_index=args.frame, channel_indices=(2, 3))
        
        # Soustraire le fond si spécifié
        if background_data is not None:
            bg_complex_data = extract_frame(background_data, frame_index=args.frame, channel_indices=(0, 1))
            complex_data = subtract_background(complex_data, bg_complex_data)


        """""
        # pour estimer les coefficients de déséquilibre IQ si demandée
        if args.estimate_iq_imbalance:
            print("Estimation des coefficients de déséquilibre IQ...")
            alpha, beta = estimate_imbalance_parameters(complex_data, ref_data)
            
            # Appliquer la correction si demandée
            if args.correct_iq:
                print("Application de la correction du déséquilibre IQ...")
                complex_data = correct_imbalance(complex_data, alpha=alpha, beta=beta)
        """""
        radar_data = reshape_to_chirps(complex_data, params, "without_pause")

        
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
        
        fig1 = plot_range_profile(
            range_profile, 
            range_axis_no_padding, 
            title=f"Profil de distance - Frame {args.frame}",
            save_path=f"{args.output_dir}/range_profile_frame{args.frame}.pdf"
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
                apply_2d_filter=args.apply_2d_filter,
                kernel_size=args.filter_size
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
        
        # Appliquer les traitements supplémentaires si demandés
        if args.remove_static:
            print("Suppression des composantes statiques...")
            range_doppler_map = remove_static_components(range_doppler_map, linear_scale=False)
        
        if args.clutter_threshold > 0:
            print(f"Application d'un seuil de clutter de {args.clutter_threshold} dB...")
            range_doppler_map = apply_clutter_threshold(range_doppler_map, args.clutter_threshold)
        
        # Application de la détection CFAR si demandée
        detections = None
        targets = None
        
        if args.detect_targets:
            print(f"Application de la détection CFAR ({args.cfar_method})...")
            try:
                detections = apply_cfar_detector(
                    range_doppler_map, 
                    guard_cells=(2, 4), 
                    training_cells=(4, 8),
                    threshold_factor=args.cfar_threshold,
                    cfar_method=args.cfar_method,
                    percentile=args.cfar_percentile
                )
                num_targets = np.sum(detections)
                print(f"Nombre de cibles détectées: {num_targets}")
                
                # Estimation des paramètres des cibles si demandée
                if args.estimate_targets and num_targets > 0:
                    print("Estimation des paramètres précis des cibles...")
                    targets = estimate_target_parameters(range_doppler_map, detections, range_axis, velocity_axis)
                    
                    # Affichage des cibles estimées
                    print("\nCibles détectées:")
                    for i, target in enumerate(targets):
                        print(f"Cible {i+1}: distance = {target['range']:.2f} m, vitesse = {target['velocity']:.2f} m/s, amplitude = {target['amplitude']:.2f} dB")
                
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
            
            if targets is not None and args.estimate_targets:
                # Utiliser les positions estimées des cibles
                target_ranges = [target['range'] for target in targets]
                target_velocities = [target['velocity'] for target in targets]
                plt.scatter(target_ranges, target_velocities, c='r', marker='x', s=50, label='Détections (estimées)')
            else:
                # Utiliser les positions brutes des détections CFAR
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
            print("Pas de détection CFAR ou détections demandées, visualisation normale...")
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

        # 4. Générer le spectrogramme si demandé
        if args.generate_spectrogram:
            print("Génération du spectrogramme...")
            filename_base = os.path.splitext(os.path.basename(args.data_file))[0]
            fig4 = plot_spectrogram(
                complex_data,
                params,
                title=f"Spectrogramme - {filename_base} - Frame {args.frame}",
                save_path=f"{args.output_dir}/spectrogram_frame{args.frame}_{filename_base}.pdf"
                )
        
        # 5. Comparaison des RDM déséquilibrée et corrigée
        if args.analyze_iq_imbalance or args.correct_iq:
            print("\nGénération de la comparaison entre RDM déséquilibrée, corrigée et référence...")
            
            # Extraire les données déséquilibrées
            unbalanced_data = extract_frame(data, frame_index=args.frame, channel_indices=(0, 1))
            
            # Récupérer les données de référence si disponibles
            reference_data = None
            if data.shape[1] >= 4:
                reference_data = extract_frame(data, frame_index=args.frame, channel_indices=(2, 3))
                alpha, beta = estimate_imbalance_parameters(unbalanced_data, reference_data)
            else:
                # Sinon, utiliser la méthode statistique
                alpha, beta = estimate_imbalance_parameters(unbalanced_data)
            
            # Corriger le déséquilibre
            corrected_data = correct_imbalance(unbalanced_data, alpha, beta)
            
            # Transformer les données en format chirps x échantillons
            unbalanced_radar_data = reshape_to_chirps(unbalanced_data, params, "without_pause")
            corrected_radar_data = reshape_to_chirps(corrected_data, params, "without_pause")
            
            # Préparer les données de référence si disponibles
            reference_radar_data = None
            if reference_data is not None:
                reference_radar_data = reshape_to_chirps(reference_data, params, "without_pause")
            
            # Générer les cartes Range-Doppler
            unbalanced_rdm, range_axis, velocity_axis = generate_range_doppler_map_with_axes(
                unbalanced_radar_data, 
                params,
                window_type=args.window_type,
                range_padding_factor=args.range_padding,
                doppler_padding_factor=args.doppler_padding,
                apply_2d_filter=args.apply_2d_filter,
                kernel_size=args.filter_size
            )
            
            corrected_rdm, _, _ = generate_range_doppler_map_with_axes(
                corrected_radar_data,
                params,
                window_type=args.window_type,
                range_padding_factor=args.range_padding,
                doppler_padding_factor=args.doppler_padding,
                apply_2d_filter=args.apply_2d_filter,
                kernel_size=args.filter_size
            )
            
            # Générer la RDM de référence si disponible
            reference_rdm = None
            if reference_radar_data is not None:
                reference_rdm, _, _ = generate_range_doppler_map_with_axes(
                    reference_radar_data,
                    params,
                    window_type=args.window_type,
                    range_padding_factor=args.range_padding,
                    doppler_padding_factor=args.doppler_padding,
                    apply_2d_filter=args.apply_2d_filter,
                    kernel_size=args.filter_size
                )
            
            # Appliquer les traitements supplémentaires si demandés
            if args.remove_static:
                unbalanced_rdm = remove_static_components(unbalanced_rdm, linear_scale=False)
                corrected_rdm = remove_static_components(corrected_rdm, linear_scale=False)
                if reference_rdm is not None:
                    reference_rdm = remove_static_components(reference_rdm, linear_scale=False)
            
            if args.clutter_threshold > 0:
                unbalanced_rdm = apply_clutter_threshold(unbalanced_rdm, args.clutter_threshold)
                corrected_rdm = apply_clutter_threshold(corrected_rdm, args.clutter_threshold)
                if reference_rdm is not None:
                    reference_rdm = apply_clutter_threshold(reference_rdm, args.clutter_threshold)
            
            # Créer la figure avec trois sous-plots (ou deux si pas de référence)
            num_plots = 3 if reference_rdm is not None else 2
            fig, axes = plt.subplots(1, num_plots, figsize=(6*num_plots, 8))
            
            # Limiter la plage dynamique
            dynamic_range = args.dynamic_range
            vmax_unbalanced = np.max(unbalanced_rdm)
            vmin_unbalanced = vmax_unbalanced - dynamic_range
            
            vmax_corrected = np.max(corrected_rdm)
            vmin_corrected = vmax_corrected - dynamic_range
            
            # Plot de la RDM déséquilibrée
            im1 = axes[0].pcolormesh(range_axis, velocity_axis, unbalanced_rdm[:len(velocity_axis), :len(range_axis)],
                                cmap='jet', vmin=vmin_unbalanced, vmax=vmax_unbalanced, shading='auto')
            axes[0].set_title('Range-Doppler Déséquilibré (I1,Q1)')
            axes[0].set_xlabel('Distance (m)')
            axes[0].set_ylabel('Vitesse (m/s)')
            plt.colorbar(im1, ax=axes[0], label='Magnitude (dB)')
            axes[0].grid(True, linestyle='--', alpha=0.5)
            axes[0].axhline(y=0, color='r', linestyle='-', alpha=0.3)
            
            # Plot de la RDM corrigée
            im2 = axes[1].pcolormesh(range_axis, velocity_axis, corrected_rdm[:len(velocity_axis), :len(range_axis)],
                                cmap='jet', vmin=vmin_corrected, vmax=vmax_corrected, shading='auto')
            axes[1].set_title('Range-Doppler Corrigé')
            axes[1].set_xlabel('Distance (m)')
            if num_plots == 2:  # Si pas de référence, afficher aussi l'axe Y
                axes[1].set_ylabel('Vitesse (m/s)')
            plt.colorbar(im2, ax=axes[1], label='Magnitude (dB)')
            axes[1].grid(True, linestyle='--', alpha=0.5)
            axes[1].axhline(y=0, color='r', linestyle='-', alpha=0.3)
            
            # Plot de la RDM de référence si disponible
            if reference_rdm is not None:
                vmax_reference = np.max(reference_rdm)
                vmin_reference = vmax_reference - dynamic_range
                
                im3 = axes[2].pcolormesh(range_axis, velocity_axis, reference_rdm[:len(velocity_axis), :len(range_axis)],
                                    cmap='jet', vmin=vmin_reference, vmax=vmax_reference, shading='auto')
                axes[2].set_title('Range-Doppler Référence (I2,Q2)')
                axes[2].set_xlabel('Distance (m)')
                plt.colorbar(im3, ax=axes[2], label='Magnitude (dB)')
                axes[2].grid(True, linestyle='--', alpha=0.5)
                axes[2].axhline(y=0, color='r', linestyle='-', alpha=0.3)
            
            # Caractéristiques du déséquilibre
            imbalance_metrics = calculate_imbalance_metrics(unbalanced_data, alpha, beta)
            irr_db = imbalance_metrics.get('irr_dB', 'N/A')
            amp_imbalance = imbalance_metrics.get('amplitude_imbalance_dB', 'N/A')
            phase_error = imbalance_metrics.get('phase_error_deg', 'N/A')
            
            # Ajouter un titre global avec les infos de déséquilibre
            plt.suptitle(f"Comparaison RDM - {filename_base} - Frame {args.frame}\n"
                        f"IRR: {irr_db} dB | Amp. Imbalance: {amp_imbalance} dB | Phase Error: {phase_error}°", 
                        fontsize=14)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ajuster pour le titre global
            
            # Sauvegarder la figure
            iq_compare_path = f"{args.output_dir}/iq_correction_rdm_frame{args.frame}_{filename_base}.pdf"
            plt.savefig(iq_compare_path, dpi=300)
            print(f"Comparaison des RDMs avant/après correction sauvegardée dans {iq_compare_path}")
        # génère tous les plots
        plt.show()
        print(f"\nToutes les visualisations ont été enregistrées dans le répertoire {args.output_dir}")
        
    except Exception as e:
        print(f"Erreur: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()