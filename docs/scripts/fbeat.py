import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

f0 = 5.0 
fd = 5.5  

t_max = 5.0 
sampling_rate = 1000
t = np.linspace(0, t_max, int(t_max * sampling_rate))
A = 1.0

signal_transmitted = A * np.sin(2 * np.pi * f0 * t)
signal_received = A * np.sin(2 * np.pi * fd * t)
signal_mixed = signal_transmitted * signal_received

beat_freq = abs(fd - f0)
beat_component = 0.5 * A * np.cos(2 * np.pi * beat_freq * t) 
fig = plt.figure(figsize=(12, 10))
gs = GridSpec(3, 1, figure=fig, height_ratios=[1, 1, 2], hspace=0.4)

# Signal émis
ax1 = fig.add_subplot(gs[0])
ax1.plot(t, signal_transmitted, 'b-', label=f'Signal envoyé (f_0 = {f0} Hz)')
ax1.set_xlim(0, t_max)
ax1.set_ylim(-1.1*A, 1.1*A)
ax1.set_ylabel('Amplitude')
ax1.set_title('Signal émis')
ax1.grid(True)
ax1.legend(loc='upper right')

# Signal reçu
ax2 = fig.add_subplot(gs[1])
ax2.plot(t, signal_received, 'g-', label=f'Signal reçu (f_D = {fd} Hz)')
ax2.set_xlim(0, t_max)
ax2.set_ylim(-1.1*A, 1.1*A)
ax2.set_ylabel('Amplitude')
ax2.set_title('Signal reçu (décalé par effet Doppler)')
ax2.grid(True)
ax2.legend(loc='upper right')

# Signal mixé et fbeat
ax3 = fig.add_subplot(gs[2])
ax3.plot(t, signal_mixed, 'b-', alpha=0.8, label='Mix')
ax3.plot(t, beat_component, 'r-', linewidth=2, label=f'Beat Frequency')
ax3.set_xlim(0, t_max)
ax3.set_ylim(-1.1*A, 1.1*A)
ax3.set_xlabel('Temps (s)')
ax3.set_ylabel('Amplitude')
ax3.set_title('Signal mixé et composante basse fréquence')
ax3.grid(True)
ax3.legend(loc='upper right')


fig.suptitle('Visualisation du mélangeur réel', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
output_dir = 'docs/images'
import os
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f'{output_dir}/doppler_signal_mixing_with_beat_component.png', dpi=300)
print(f"Graphique enregistré dans {output_dir}/doppler_signal_mixing_with_beat_component.png")

plt.show()