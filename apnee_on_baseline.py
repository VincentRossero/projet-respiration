from script import *
import physio 
import sonpy
import neo
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pprint import pprint
import pandas as pd
from pandasgui import show
from scipy import signal

duree_baseline =300
date_injection =852

sig, srate, unit, time = read_one_mouse_from_nc('147 J1')
resp_bis = iirfilt(sig, srate, lowcut = 0.5, highcut = 30)
resp_bis =signal.detrend(resp_bis)

resp, resp_cycles = physio.compute_respiration(resp_bis, srate, parameter_preset='rat_plethysmo') 
resp_cycles['frequence_respi']=60/resp_cycles['cycle_duration']

inspi_index = resp_cycles['inspi_index'].values
expi_index = resp_cycles['expi_index'].values

mask_baseline = (resp_cycles['inspi_time'] < date_injection) & (resp_cycles['inspi_time'] > (date_injection-duree_baseline))

# mask_baseline = ((resp_cycles['inspi_time'] < date_injection) & 
#                  (resp_cycles['inspi_time'] > (date_injection - duree_baseline)) &
#                  ((resp_cycles['inspi_time'] < 805) | (resp_cycles['inspi_time'] > 821)))



frequence_moyenne= resp_cycles[mask_baseline]['frequence_respi'].mean()
amplitude_moyenne=resp_cycles[mask_baseline]['total_amplitude'].mean()
volume_moyen = resp_cycles[mask_baseline]['total_volume'].mean()
duree_moyenne= resp_cycles[mask_baseline]['cycle_duration'].mean()

print("Paramètres de la période baseline :")
print(f"Fréquence moyenne : {frequence_moyenne} cycles/min")
print(f"Amplitude moyenne : {amplitude_moyenne}")
print(f"Volume moyen : {volume_moyen} L")

threshold = duree_moyenne * 3
seuil_amplitude = amplitude_moyenne * 0.2
print("Threshold:", threshold)
print("Seuil Amplitude:", seuil_amplitude)

apnea_cycles = resp_cycles[mask_baseline & (resp_cycles['cycle_duration'] > threshold)]
apnea_cycles = apnea_cycles.reset_index(drop=True)
# indices_a_supprimer = [11]
# apnea_cycles = apnea_cycles.drop(indices_a_supprimer)


apnea_times_sec = apnea_cycles['inspi_time'].values
apnea_cycles['total_duration']=apnea_cycles['cycle_duration']
apnea_cycles['apnee']=True

# show (apnea_cycles)
mask_amplitude = resp_cycles[mask_baseline & (resp_cycles['total_amplitude'] < seuil_amplitude)]
index_amplitude = mask_amplitude.index
index_df = pd.DataFrame(index_amplitude, columns=['index'])
index_df['group'] = (index_df['index'] != index_df['index'].shift() + 1).cumsum()
consecutive_groups = index_df.groupby('group')['index'].apply(list)

premiers_cycles = pd.DataFrame(columns=resp_cycles.columns)
for group in consecutive_groups:
    indices = group
    total_duration = resp_cycles.loc[indices, 'cycle_duration'].sum()
    if total_duration > threshold :
        first_index = indices[0]
        premier_cycle = resp_cycles.loc[first_index].copy()
        premier_cycle['total_duration'] = total_duration
        premiers_cycles = pd.concat([premiers_cycles, premier_cycle.to_frame().T], ignore_index=True)
        premiers_cycles['apnee']=False

# show (premiers_cycles)


inspi_index_to_drop = premiers_cycles['inspi_index'].tolist()
apnea_cycles = apnea_cycles[~apnea_cycles['inspi_index'].isin(inspi_index_to_drop)]



total_apnea=pd.concat([apnea_cycles,premiers_cycles], axis = 0,ignore_index=True)

# show (total_apnea)

total_apnea = total_apnea.sort_values(by='inspi_index')
mask_baseline_apnea = (total_apnea['inspi_time'] < date_injection) & (total_apnea['inspi_time'] > (date_injection-duree_baseline))


fig,ax= plt.subplots(nrows=1,sharex=True)


ax.plot(time,resp_bis)
ax.scatter(time[inspi_index], resp[inspi_index], marker='o', color='green')
ax.scatter(time[expi_index], resp[expi_index], marker='o', color='red')
ax.axvline(x=date_injection, color='r', linestyle='--')  
ax.axvline(x=date_injection-duree_baseline, color='r', linestyle='--') 


for index, row in total_apnea[mask_baseline_apnea].iterrows():
    inspi_time = row['inspi_time']
    if row['apnee']:
        ax.axvline(inspi_time, color='blue', lw=2, linestyle='--')
    else:
        ax.axvline(inspi_time, color='green', lw=2, linestyle='--')


plt.show()


nb_baseline = total_apnea[mask_baseline_apnea].shape[0]
temps_total_baseline= total_apnea[mask_baseline_apnea]['total_duration'].sum()
average_baseline = total_apnea[mask_baseline_apnea]['total_duration'].mean()
print ("nombre d'apnee baseline",nb_baseline)
print ("temps passé en apnée baseline",temps_total_baseline)
print ("temps moyen d'une apnee baseline",average_baseline)