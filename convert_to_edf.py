import numpy as np
import pyedflib
from script import *

# Lire les données depuis le fichier
sig, srate, unit, time = read_one_mouse_from_nc('jon thermiser test 250724 ms2')

# Assurez-vous que sig est bien de type float64
sig = sig.astype(np.float64)

# Nom du fichier EDF
edf_file = 'jon_thermiser_test_250724_ms2.edf'

# Créez et ouvrez le fichier EDF pour l'écriture
with pyedflib.EdfWriter(edf_file, 1, file_type=pyedflib.FILETYPE_EDFPLUS) as f:
    # Définir les métadonnées du signal avec précision réduite
    channel_info = {
        'label': 'sig',
        'dimension': unit,
        'sample_rate': int(round(srate)),  # Arrondir la fréquence d'échantillonnage
        'physical_max': round(float(np.max(sig)), 6),
        'physical_min': round(float(np.min(sig)), 6),
        'digital_max': 32767,
        'digital_min': -32768
    }
    
    # Définir les en-têtes du signal
    f.setSignalHeaders([channel_info])
    
    # Écrire les échantillons physiques directement
    try:
        # Passer directement le tableau 1D sig
        f.writePhysicalSamples(sig)
    except AttributeError:
        print("La méthode 'writePhysicalSamples' n'est pas disponible dans cette version de pyedflib.")
        print("Veuillez vérifier votre installation de pyedflib.")

print(f'Fichier EDF créé : {edf_file}')
