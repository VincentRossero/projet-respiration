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

sig, srate, unit, time = read_one_mouse_from_nc('jon thermiser test 250724 ms2')



debut_respi= 0
fin_baseline= 30000

sig =sig [int(debut_respi*srate):int(fin_baseline*srate)]
time =time[int(debut_respi*srate):int(fin_baseline*srate)]

resp_bis = iirfilt(sig, srate, lowcut = 0.5, highcut = 30)
resp_bis =signal.detrend(resp_bis)


resp, resp_cycles = physio.compute_respiration(resp_bis, srate, parameter_preset='rat_plethysmo') 
resp_cycles['frequence_respi']=60/resp_cycles['cycle_duration']

inspi_index = resp_cycles['inspi_index'].values
expi_index = resp_cycles['expi_index'].values


fig,ax= plt.subplots(nrows=1)
ax.plot(time,resp_bis)
ax.scatter(time[inspi_index], resp[inspi_index], marker='o', color='green')
ax.scatter(time[expi_index], resp[expi_index], marker='o', color='red')
plt.show()

fig,ax=plt.subplots(nrows=1)
ax.plot(resp_cycles['inspi_time'],resp_cycles['frequence_respi'])
ax.set_title('Frequence respiratoire ')
ax.set_ylabel('frequence en nombre de cycle par minute')
plt.show()

resp_cycles['minute'] = (resp_cycles['inspi_time']  // 60).astype(int)
show (resp_cycles)

grouped_data= resp_cycles.groupby('minute').mean().reset_index()
show(grouped_data)

plt.figure(figsize=(10, 6))
plt.bar(grouped_data['minute'], grouped_data['frequence_respi'], width=0.8, align='center', color='skyblue', edgecolor='black')
plt.xlabel('Minutes ')
plt.ylabel('Fréquence respiratoire (cycles par minute)')
plt.title('Évolution de la fréquence respiratoire minute par minute ')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(grouped_data['minute'])
plt.legend()
plt.show()