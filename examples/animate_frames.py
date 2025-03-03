import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse

# Ajouter le répertoire parent au chemin d'accès
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_handling.data_loader import load_fmcw_data, extract_frame, reshape_to_chirps
from src.signal_processing.range_doppler import (generate_range_doppler_map,
                                               calculate_range_axis,
                                               calculate_velocity_axis)

def main():
    """Fonction principale pour créer une animation des frames radar"""
    
    # Parser les arguments de ligne de commande
    parser = argparse.ArgumentParser(description='Animation des données FMCW')
    parser.add_argument('--data-file', type=str, default='data/MS1-FMCW.npz',
                       help='Chemin vers le fichier de données')
    parser.add_argument('--start-frame', type=int, default=0,
                       help='Frame de départ')
    parser.add_argument('--num-frames', type=int, default=10,
                       help='Nombre de frames à animer')
    parser.add_argument('--output-file', type=str, default='output/radar_animation.mp4',
                       help='Fichier de sortie pour l\'animation')
    args = parser.parse_args()
    
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    print(f"Chargement des données depuis {args.data_file}...")
    
    # Charger les données
    data, params = load_fmcw_data(args.data_file)
    
    # Vérifier que le nombre de frames demandé est valide
    max_frames = data.shape[0]
    if args.start_frame + args.num_frames > max_frames:
        args.num_frames = max_frames - args.start_frame
        print(f"Attention: Ajustement du nombre de frames à {args.num_frames}")
    
    # Calculer les axes
    range_axis = calculate_range_axis(params)
    velocity_axis = calculate_velocity_axis(params)
    
    # Créer la figure et l'axe
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Fonction pour initialiser l'animation
    def init():
        ax.clear()
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Vitesse (m/s)')
        ax.set_title('Animation Carte Range-Doppler')
        return []
    
    # Fonction pour mettre à jour l'animation à chaque frame
    def update(frame_idx):
        ax.clear()
        
        # Indice réel de la frame
        real_idx = args.start_frame + frame_idx
        
        # Extraire la frame
        complex_data = extract_frame(data, frame_index=real_idx, channel_indices=(0, 1))
        radar_data = reshape_to_chirps(complex_data, params)
        
        # Générer la carte Range-Doppler
        range_doppler_map = generate_range_doppler_map(radar_data)
        
        # Calculer les limites pour la colormap
        vmax = np.max(range_doppler_map)
        vmin = vmax - 60
        
        # Adapter les dimensions aux axes
        map_to_plot = range_doppler_map[:len(velocity_axis), :len(range_axis)]
        
        # Afficher la carte Range-Doppler
        im = ax.pcolormesh(range_axis, velocity_axis, map_to_plot, 
                         cmap='viridis', vmin=vmin, vmax=vmax, shading='auto')
        
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Vitesse (m/s)')
        ax.set_title(f'Carte Range-Doppler - Frame {real_idx}')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Ajouter une ligne horizontale à v=0
        ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        return [im]
    
    print(f"Création de l'animation avec {args.num_frames} frames...")
    
    # Créer l'animation
    ani = animation.FuncAnimation(fig, update, frames=args.num_frames,
                                init_func=init, blit=True, interval=200)
    
    # Enregistrer l'animation
    print(f"Enregistrement de l'animation dans {args.output_file}...")