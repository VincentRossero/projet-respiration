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
date_injection =2113
crise =4011
post_ictal =4011
end_time =post_ictal +2400

sig, srate, unit, timevector = read_one_mouse_from_nc('281 J1 ICF1 220724')
resp_bis = iirfilt(sig, srate, lowcut = 0.5, highcut = 30)
resp_bis =signal.detrend(resp_bis)

resp, resp_cycles = physio.compute_respiration(resp_bis, srate, parameter_preset='rat_plethysmo') 
resp_cycles['frequence_respi']=60/resp_cycles['cycle_duration']

#suppression de cycle artefactuels 

# condition1 = (resp_cycles['inspi_time'] >= 1443.5) & (resp_cycles['inspi_time'] <= 1449)
# condition2 = (resp_cycles['inspi_time'] >= 1450) & (resp_cycles['inspi_time'] <= 1454.5)
# condition = condition1 | condition2
# resp_cycles.loc[condition, ['frequence_respi', 'total_amplitude']] = 0




inspi_index = resp_cycles['inspi_index'].values
expi_index = resp_cycles['expi_index'].values

mask_baseline = (resp_cycles['inspi_time'] < date_injection) & (resp_cycles['inspi_time'] > (date_injection-duree_baseline))


frequence_moyenne= resp_cycles[mask_baseline]['frequence_respi'].mean()
amplitude_moyenne=resp_cycles[mask_baseline]['total_amplitude'].mean()
volume_moyen = resp_cycles[mask_baseline]['total_volume'].mean()
duree_moyenne= resp_cycles[mask_baseline]['cycle_duration'].mean()


fig, ax = plt.subplots(nrows=1)


ax.plot(timevector, sig)
ax.plot(timevector,resp_bis,color = 'orange')
ax.scatter(timevector[inspi_index], resp[inspi_index], marker='o', color='green')
ax.scatter(timevector[expi_index], resp[expi_index], marker='o', color='red')
ax.axvline(x=date_injection, color='r', linestyle='--')  
ax.axvline(x=date_injection-duree_baseline, color='r', linestyle='--') 
ax.axvline(x=crise,color='k',linestyle='--')
ax.axvline(x=post_ictal,color='k',linestyle='--')

plt.show()






# mask_baseline = ((resp_cycles['inspi_time'] < date_injection) & 
#                  (resp_cycles['inspi_time'] > (date_injection - duree_baseline)) &
#                  ((resp_cycles['inspi_time'] < 670) | (resp_cycles['inspi_time'] > 675)))



frequence_moyenne_1 = 60 / duree_moyenne

df = pd.DataFrame(resp_cycles[mask_baseline])
show(df)


fig,axs = plt.subplots(nrows= 2,sharex= True)

ax=axs[0]
ax.plot(resp_cycles['inspi_time'],resp_cycles['frequence_respi'])
ax.set_title('Frequence respiratoire ')
ax.axhline(y=frequence_moyenne,color='k',linestyle='--')
ax.axvline(x=date_injection-duree_baseline, color='r', linestyle='--')
ax.axvline(x=date_injection, color='r', linestyle='--') 
ax.axvline(x=crise, color='k', linestyle='--') 
ax.axvline(x=post_ictal, color='k', linestyle='--') 
ax.set_ylabel('frequence en nombre de cycle par minute')
ax.plot()

ax=axs[1]
ax.plot(resp_cycles['inspi_time'],resp_cycles['total_amplitude'])
ax.set_title('Amplitude respiratoire')
ax.axhline(y=amplitude_moyenne,color='k',linestyle='--')
ax.axvline(x=date_injection-duree_baseline, color='r', linestyle='--')
ax.axvline(x=date_injection, color='r', linestyle='--') 
ax.axvline(x=crise, color='k', linestyle='--') 
ax.axvline(x=post_ictal, color='k', linestyle='--') 
ax.set_ylabel('Amplitude totale par cycle ')
ax.plot()


plt.show()

#etude du retour à la baseline en frequence et en amplitude

mask_post_ictal = resp_cycles[(resp_cycles['inspi_time'] >= post_ictal) & (resp_cycles['inspi_time'] <= end_time)]
mask_post_ictal['minute'] = ((mask_post_ictal['inspi_time'] - post_ictal) // 60).astype(int)
grouped_data = mask_post_ictal.groupby('minute').mean().reset_index()

#on essaye de voir comment la baseline évolue sur 5 minutes 
mask_etude_baseline = resp_cycles[(resp_cycles['inspi_time'] >= date_injection-duree_baseline) & (resp_cycles['inspi_time'] <= date_injection)]
mask_etude_baseline['minute'] = ((mask_etude_baseline['inspi_time'] - (date_injection-duree_baseline)) // 60).astype(int)
grouped_data_baseline = mask_etude_baseline.groupby('minute').mean().reset_index()


df = pd.DataFrame(grouped_data)
show(df)



plt.figure(figsize=(10, 6))
plt.bar(grouped_data['minute'], grouped_data['frequence_respi'], width=0.8, align='center', color='skyblue', edgecolor='black')
plt.axhline(y=frequence_moyenne,color='r',linestyle='--',label=f'valeur moyenne baseline')
plt.xlabel('Minutes post-ictale')
plt.ylabel('Fréquence respiratoire (cycles par minute)')
plt.title('Évolution de la fréquence respiratoire minute par minute ')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(grouped_data['minute'])
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(grouped_data['minute'], grouped_data['total_amplitude'], width=0.8, align='center', color='skyblue', edgecolor='black')
plt.axhline(y=amplitude_moyenne,color='r',linestyle='--',label=f'valeur moyenne baseline')
plt.xlabel('Minutes post-ictale')
plt.ylabel('Amplitude respiratoire (cycles par minute)')
plt.title("Évolution de l'amplitude respiratoire minute par minute ")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(grouped_data['minute'])
plt.legend()
plt.show()


plt.figure(figsize=(10, 6))
plt.bar(grouped_data_baseline['minute'], grouped_data_baseline['frequence_respi'], width=0.8, align='center', color='skyblue', edgecolor='black')
plt.axhline(y=frequence_moyenne,color='r',linestyle='--',label=f'valeur moyenne baseline')
plt.xlabel('Minutes baseline')
plt.ylabel('Fréquence respiratoire (cycles par minute)')
plt.title('Évolution de la fréquence respiratoire minute par minute ')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(grouped_data_baseline['minute'])
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(grouped_data_baseline['minute'], grouped_data_baseline['total_amplitude'], width=0.8, align='center', color='skyblue', edgecolor='black')
plt.axhline(y=amplitude_moyenne,color='r',linestyle='--',label=f'valeur moyenne baseline')
plt.xlabel('Minutes baseline')
plt.ylabel('Amplitude respiratoire (cycles par minute)')
plt.title("Évolution de l'amplitude respiratoire minute par minute ")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(grouped_data_baseline['minute'])
plt.legend()
plt.show()


print("Fréquence moyenne approximée : ", frequence_moyenne_1)
print(f"Fréquence respiratoire moyenne: {frequence_moyenne} respirations par minute")
print(f"Amplitude respiratoire moyenne: {amplitude_moyenne} unités d'amplitude")
print(f"Volume respiratoire moyen: {volume_moyen} unités de volume")


# threshold_double= duree_moyenne*2
# threshold_triple= duree_moyenne*3

# mask_apnea_double = resp_cycles[mask_baseline]['cycle_duration']>threshold_double
# # les apnées detectés
# df_view= resp_cycles[mask_baseline][mask_apnea_double]
# df = pd.DataFrame (df_view)
# show (df)
# nb_apneas_double = np.sum(mask_apnea_double) 
# times_apneas = resp_cycles[mask_baseline][mask_apnea_double]['cycle_duration'].sum()
# print ("nombre apnee avec seuil de 2* duree moyenne: ",nb_apneas_double)
# print (times_apneas)


# mask_apnea_triple = resp_cycles[mask_baseline]['cycle_duration'] > threshold_triple
# # les apnées detectés
# df_view = resp_cycles[mask_baseline][mask_apnea_triple]
# df = pd.DataFrame(df_view)
# show(df)
# nb_apneas_triple = np.sum(mask_apnea_triple)
# times_apneas = resp_cycles[mask_baseline][mask_apnea_triple]['cycle_duration'].sum()
# print("nombre apnee avec seuil de 3* duree moyenne: ", nb_apneas_triple)
# print(times_apneas)






# fig, axs = plt.subplots(nrows=2,sharex=True,sharey=True)

# ax = axs[0]

# #ax.plot(timevector, sig)
# ax.plot(timevector,resp_bis,color = 'orange')
# ax.scatter(timevector[inspi_index], resp[inspi_index], marker='o', color='green')
# ax.scatter(timevector[expi_index], resp[expi_index], marker='o', color='red')
# ax.axvline(x=date_injection, color='r', linestyle='--')  
# ax.axvline(x=date_injection-duree_baseline, color='r', linestyle='--') 
# ax.axvline(x=crise,color='k',linestyle='--')
# ax.axvline(x=post_ictal,color='k',linestyle='--')
# apnea_times_d = resp_cycles[mask_baseline][mask_apnea_double]['inspi_time']
# for apnea_time in apnea_times_d:
#     ax.axvline(x=apnea_time, color='b', linestyle='--')
# ax =axs[1]

# #ax.plot(timevector, sig)
# ax.plot(timevector,resp_bis,color = 'orange')
# ax.scatter(timevector[inspi_index], resp[inspi_index], marker='o', color='green')
# ax.scatter(timevector[expi_index], resp[expi_index], marker='o', color='red')
# ax.axvline(x=date_injection, color='r', linestyle='--')  
# ax.axvline(x=date_injection-duree_baseline, color='r', linestyle='--') 
# ax.axvline(x=crise,color='k',linestyle='--')
# ax.axvline(x=post_ictal,color='k',linestyle='--')
# apnea_times_t = resp_cycles[mask_baseline][mask_apnea_triple]['inspi_time']
# for apnea_time in apnea_times_t:
#     ax.axvline(x=apnea_time, color='b', linestyle='--')

# plt.show()

































