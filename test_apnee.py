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
date_injection =811
crise =1512
post_ictal =1572
end_time =post_ictal +2400

sig, srate, unit, time = read_one_mouse_from_nc('89 J1')
resp_bis = iirfilt(sig, srate, lowcut = 0.5, highcut = 30)
resp_bis =signal.detrend(resp_bis)

resp, resp_cycles = physio.compute_respiration(resp_bis, srate, parameter_preset='rat_plethysmo') 
resp_cycles['frequence_respi']=60/resp_cycles['cycle_duration']


#ici on vire les cycles qui corresponde a du bruit de fond et on remplace les valeurs pour une seule detection 

 
# condition1 = (resp_cycles['inspi_time'] >= 1443.5) & (resp_cycles['inspi_time'] <= 1449)
# condition2 = (resp_cycles['inspi_time'] >= 1450.2) & (resp_cycles['inspi_time'] <= 1454.5)

# total_duration = resp_cycles.loc[condition1, 'cycle_duration'].sum()
# resp_cycles = resp_cycles.drop(resp_cycles[condition1].index)
# total_duration2 = resp_cycles.loc[condition2, 'cycle_duration'].sum()
# resp_cycles = resp_cycles.drop(resp_cycles[condition2].index)

# resp_cycles.loc[9990, 'total_amplitude'] = 0
# resp_cycles.loc[9990, 'cycle_duration'] = total_duration
# resp_cycles.loc[9990, 'frequence_respi'] = 60/total_duration

# resp_cycles.loc[10015, 'total_amplitude'] = 0
# resp_cycles.loc[10015, 'cycle_duration'] = total_duration2
# resp_cycles.loc[10015, 'frequence_respi'] = 60/total_duration2


inspi_index = resp_cycles['inspi_index'].values
expi_index = resp_cycles['expi_index'].values


#un mask pour la baseline et un pour de la crise à la fin de l'enregistrement 
mask_baseline = (resp_cycles['inspi_time'] < date_injection) & (resp_cycles['inspi_time'] > (date_injection-duree_baseline))
mask_crise = (resp_cycles['inspi_time']< end_time) & (resp_cycles['inspi_time'] > crise)
mask_crise_s5 = (resp_cycles['inspi_time']< post_ictal+240) & (resp_cycles['inspi_time'] > post_ictal)
mask_crise_s10 = (resp_cycles['inspi_time']< post_ictal+480) & (resp_cycles['inspi_time'] > post_ictal+240)

#etude des parametres 

#BASELINE
frequence_moyenne= resp_cycles[mask_baseline]['frequence_respi'].mean()
amplitude_moyenne=resp_cycles[mask_baseline]['total_amplitude'].mean()
volume_moyen = resp_cycles[mask_baseline]['total_volume'].mean()
duree_moyenne= resp_cycles[mask_baseline]['cycle_duration'].mean()

print("Paramètres de la période baseline :")
print(f"Fréquence moyenne : {frequence_moyenne} cycles/min")
print(f"Amplitude moyenne : {amplitude_moyenne}")
print(f"Volume moyen : {volume_moyen} L")



#SEIZURE 5MIN 
duree_moyenne_s5 = resp_cycles[mask_crise_s5]['cycle_duration'].mean()
amplitude_moyenne_s5 = resp_cycles[mask_crise_s5]['total_amplitude'].mean()

#SEIZURE 10 MIN 
duree_moyenne_s10 =resp_cycles[mask_crise_s10]['cycle_duration'].mean()
amplitude_moyenne_s10 = resp_cycles[mask_crise_s10]['total_amplitude'].mean()

#les seuils

#BASELINE
threshold = duree_moyenne * 3
seuil_amplitude = amplitude_moyenne * 0.2
print("Threshold:", threshold)
print("Seuil Amplitude:", seuil_amplitude)

#SEIZURE 5 MIN 
threshold_s5 = duree_moyenne_s5 * 3
seuil_amplitude_s5 = amplitude_moyenne_s5 * 0.2
print("Threshold s5:", threshold_s5)
print("Seuil Amplitude s5:", seuil_amplitude_s5)

#SEIZURE 10 MIN 
threshold_s10 = duree_moyenne_s10 * 3
seuil_amplitude_s10 = amplitude_moyenne_s10 *0.2
print("Threshold s10:", threshold_s10)
print("Seuil Amplitude s10:", seuil_amplitude_s10)


#dataframe des apnées de duree ACTUALISATION
apnea_baseline = resp_cycles[mask_baseline & (resp_cycles['cycle_duration'] > threshold)]
apnea_s5 = resp_cycles[mask_crise_s5 & (resp_cycles['cycle_duration'] > threshold_s5)]
apnea_s10 = resp_cycles[mask_crise_s10 & (resp_cycles['cycle_duration'] > threshold_s10)]


apnea_cycles = pd.concat([apnea_baseline, apnea_s5, apnea_s10])


apnea_cycles = apnea_cycles.reset_index(drop=True)
indices_a_supprimer = []
apnea_cycles = apnea_cycles.drop(indices_a_supprimer)


apnea_times_sec = apnea_cycles['inspi_time'].values
apnea_cycles['total_duration']=apnea_cycles['cycle_duration']
apnea_cycles['apnee']=True


# indices_a_supprimer = [0,1,2,3,4,5,8,11,12,18]
# apnea_cycles = apnea_cycles.drop(indices_a_supprimer)
# apnea_cycles = apnea_cycles.reset_index(drop=True)


show (apnea_cycles)

#dataframe des apnées d'amplitude 
apnea_amplitude_baseline = resp_cycles[mask_baseline & (resp_cycles['total_amplitude'] < seuil_amplitude)]
apnea_amplitude_s5 = resp_cycles[mask_crise_s5 & (resp_cycles['total_amplitude'] < seuil_amplitude_s5)]
apnea_amplitude_s10 = resp_cycles[mask_crise_s10 & (resp_cycles['total_amplitude'] < seuil_amplitude_s10)]
mask_amplitude = pd.concat([apnea_amplitude_baseline, apnea_amplitude_s5, apnea_amplitude_s10])


#show (mask_amplitude)

# on classe dans consecutive groups les indexs du dataframe d'amplitude les groupes d'indices consecutifs 
index_amplitude = mask_amplitude.index
index_df = pd.DataFrame(index_amplitude, columns=['index'])
index_df['group'] = (index_df['index'] != index_df['index'].shift() + 1).cumsum()
consecutive_groups = index_df.groupby('group')['index'].apply(list)

#uniiquement les groupes de au moins 3 cycles d'affilés 


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

show (premiers_cycles)

total_apnea=pd.concat([apnea_cycles,premiers_cycles], axis = 0,ignore_index=True)

total_apnea =total_apnea.drop_duplicates(subset='inspi_index')

total_apnea = total_apnea.sort_values(by='inspi_index')
#show(total_apnea)


#affichage du signal respi + apnée normal en bleu , apnee amplitude en vert 

mask_baseline_apnea = (total_apnea['inspi_time'] < date_injection) & (total_apnea['inspi_time'] > (date_injection-duree_baseline))
mask_crise_apnea = (total_apnea['inspi_time']< end_time) & (total_apnea['inspi_time'] > crise)
mask_crise_apnea_s5 = (total_apnea['inspi_time']< post_ictal+240) & (total_apnea['inspi_time'] > post_ictal)
mask_crise_apnea_s10 = (total_apnea['inspi_time']< post_ictal+480) & (total_apnea['inspi_time'] > post_ictal+240)

fig,ax= plt.subplots(nrows=1,sharex=True)

ax.plot(time,resp_bis)
ax.scatter(time[inspi_index], resp[inspi_index], marker='o', color='green')
ax.scatter(time[expi_index], resp[expi_index], marker='o', color='red')
ax.axvline(x=date_injection, color='r', linestyle='--')  
ax.axvline(x=date_injection-duree_baseline, color='r', linestyle='--') 
ax.axvline(x=crise,color='k',linestyle='--')
ax.axvline(x=post_ictal,color='k',linestyle='--')
ax.axvline(x=crise+300,color='k')

for index, row in total_apnea[mask_baseline_apnea].iterrows():
    inspi_time = row['inspi_time']
    if row['apnee']:
        ax.axvline(inspi_time, color='blue', lw=2, linestyle='--')
    else:
        ax.axvline(inspi_time, color='green', lw=2, linestyle='--')

for index, row in total_apnea[mask_crise_apnea].iterrows():
    inspi_time = row['inspi_time']
    if row['apnee']:
        ax.axvline(inspi_time, color='blue', lw=2, linestyle='--')
        
    else:
        ax.axvline(inspi_time, color='green', lw=2, linestyle='--')

plt.show()


#recuperation des INFOS d'interet 

#Mask baseline 

nb_baseline = total_apnea[mask_baseline_apnea].shape[0]
temps_total_baseline= total_apnea[mask_baseline_apnea]['total_duration'].sum()
average_baseline = total_apnea[mask_baseline_apnea]['total_duration'].mean()
print ("nombre d'apnee baseline",nb_baseline)
print ("temps passé en apnée baseline",temps_total_baseline)
print ("temps moyen d'une apnee baseline",average_baseline)

#Mask seizure 5 minutes

nb_s5 = total_apnea[mask_crise_apnea_s5].shape[0]
temps_total_s5= total_apnea[mask_crise_apnea_s5]['total_duration'].sum()
average_s5 = total_apnea[mask_crise_apnea_s5]['total_duration'].mean()
print ("nombre d'apnee s5",nb_s5)
print ("temps passé en apnée s5",temps_total_s5)
print ("temps moyen d'une apnee s5",average_s5)

#Mask seizure 10 minutes 

nb_s10 = total_apnea[mask_crise_apnea_s10].shape[0]
temps_total_s10 = total_apnea[mask_crise_apnea_s10]['total_duration'].sum()
average_s10 = total_apnea[mask_crise_apnea_s10]['total_duration'].mean()
print("nombre d'apnee s5", nb_s10)
print("temps passé en apnée s5", temps_total_s10)
print("temps moyen d'une apnee s5", average_s10)

