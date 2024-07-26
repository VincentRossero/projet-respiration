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


duree_baseline =300
date_injection =1034
date_crise = 2317
date_fin_respi = 0   #0 si pas de fin respi 
colonnes_a_garder = ['inspi_time', 'expi_time', 'cycle_duration', 'inspi_duration', 'expi_duration','frequence_respi', 'inspi_volume', 'expi_volume', 'inspi_amplitude', 'expi_amplitude','relative_time','relative_time_minutes']


sig, srate, unit, timevector = read_one_mouse_from_nc('139 J1')
nom_excel = 'mouse2 J1.xlsx'
#times = np.arange(sig.size) / srate

resp_bis = iirfilt(sig, srate, lowcut = 0.5, highcut = 30)
resp, resp_cycles = physio.compute_respiration(resp_bis, srate, parameter_preset='rat_plethysmo') 


mask_baseline = (resp_cycles['inspi_time'] < date_injection) & (resp_cycles['inspi_time'] > (date_injection-duree_baseline))

'''
mask_baseline = ((resp_cycles['inspi_time'] < date_injection) & 
                 (resp_cycles['inspi_time'] > (date_injection - duree_baseline)) &
                 ((resp_cycles['inspi_time'] < 568) | (resp_cycles['inspi_time'] > 675)))
'''

automatic_treshold = resp_cycles[mask_baseline]['expi_duration'].mean() * 2
print(automatic_treshold)
print(srate)

duree_moyenne= resp_cycles[mask_baseline]['cycle_duration'].mean()


frequence_moyenne_1 = 60 / duree_moyenne

print("Fréquence moyenne : ", frequence_moyenne_1)


# DataFrame pour les cycles de la période de baseline sur 5 minutes avant crises 

baseline_cycles = resp_cycles[mask_baseline].copy()

baseline_cycles['relative_time'] = baseline_cycles['inspi_time'] - (date_injection-duree_baseline)
#baseline_cycles['relative_time'] = baseline_cycles['inspi_time'] - baseline_cycles['inspi_time'].min()
baseline_cycles['relative_time_minutes'] = baseline_cycles ['relative_time']/60
baseline_cycles['frequence_respi']=60/baseline_cycles['cycle_duration']
baseline_cycles = baseline_cycles.loc[:, colonnes_a_garder]



# DataFrame pour les cycles de la période de la date de crise à la fin de l'enregistrement

if (date_fin_respi != 0) :
     crisis_to_end_cycles = resp_cycles[(resp_cycles['inspi_time'] >= date_crise)&(resp_cycles['inspi_time'] < date_fin_respi)].copy()
else :     
    crisis_to_end_cycles = resp_cycles[(resp_cycles['inspi_time'] >= date_crise)].copy()

crisis_to_end_cycles['relative_time'] = crisis_to_end_cycles['inspi_time'] - date_crise
crisis_to_end_cycles['relative_time_minutes'] = crisis_to_end_cycles ['relative_time']/60
crisis_to_end_cycles['frequence_respi']=60/crisis_to_end_cycles['cycle_duration']
crisis_to_end_cycles = crisis_to_end_cycles.loc[:, colonnes_a_garder]
#Dataframe de calcul des apnées 
threshold_apnea = automatic_treshold

apnea_cycles_baseline = resp_cycles[(resp_cycles['expi_duration'] > threshold_apnea) & (resp_cycles['inspi_time'] > (date_injection-duree_baseline))&(resp_cycles['inspi_time'] < date_injection)]
if (date_fin_respi != 0) :
    apnea_cycles_crise = resp_cycles[(resp_cycles['expi_duration'] > threshold_apnea) &(resp_cycles['inspi_time']>date_crise)&(resp_cycles['inspi_time']<date_fin_respi)]
else : 
    apnea_cycles_crise = resp_cycles[(resp_cycles['expi_duration'] > threshold_apnea) &(resp_cycles['inspi_time']>date_crise)]

apnea_cycles_baseline = apnea_cycles_baseline[apnea_cycles_baseline.index.to_series().diff() != 1]
apnea_cycles_crise = apnea_cycles_crise[apnea_cycles_crise.index.to_series().diff() != 1]

apnea_cycles_baseline['relative_time'] = apnea_cycles_baseline['inspi_time'] - (date_injection-duree_baseline)
apnea_cycles_crise['relative_time'] = apnea_cycles_crise['inspi_time'] - date_crise

apnea_cycles_baseline['relative_time_minutes']=apnea_cycles_baseline['relative_time']/60
apnea_cycles_crise['relative_time_minutes']=apnea_cycles_crise['relative_time']/60

apnea_cycles_baseline['frequence_respi']= 60/apnea_cycles_baseline['cycle_duration']
apnea_cycles_crise['frequence_respi']=60/apnea_cycles_crise['cycle_duration']

apnea_cycles_baseline = apnea_cycles_baseline.loc[:, colonnes_a_garder]
apnea_cycles_crise = apnea_cycles_crise.loc[:, colonnes_a_garder]


apnea_times_sec_baseline = apnea_cycles_baseline['expi_time'].values
apnea_times_sec_crise = apnea_cycles_crise['expi_time'].values
inspi_index = resp_cycles['inspi_index'].values
expi_index = resp_cycles['expi_index'].values



fig, ax = plt.subplots()
ax.plot(timevector, sig)
#ax.plot(times, resp)
ax.plot(timevector,resp_bis,color = 'orange')
ax.scatter(timevector[inspi_index], resp[inspi_index], marker='o', color='green')
ax.scatter(timevector[expi_index], resp[expi_index], marker='o', color='red')

plt.axvline(x=date_injection, color='r', linestyle='--')  
plt.axvline(x=date_injection-duree_baseline, color='r', linestyle='--')    
plt.axvline(x=date_crise, color='b', linestyle='--') 
'''
for t in apnea_times_sec_crise:
    ax.axvline(t, color = 'k', lw = 2)
'''
ax.set_ylabel('resp')
plt.show()

#une troncature dans l'enregistrement 

'''
crisis_to_end_cycles = crisis_to_end_cycles[
    (crisis_to_end_cycles['inspi_time'] < 1395) | 
    (crisis_to_end_cycles['inspi_time'] > 1413)
]

apnea_cycles_crise = apnea_cycles_crise[
    (apnea_cycles_crise['inspi_time'] < 1395) | 
    (apnea_cycles_crise['inspi_time'] > 1413)
]
'''

dataframes = {
    'respiration_baseline': baseline_cycles,
    'respiration_peri_ictal': crisis_to_end_cycles,
    'apnea_baseline': apnea_cycles_baseline,
    'apnea_peri_ictal': apnea_cycles_crise
}


writer = pd.ExcelWriter(nom_excel, engine='xlsxwriter')

for onglet, df in dataframes.items():
    df.to_excel(writer, sheet_name=onglet, index=False)

writer.close() 

