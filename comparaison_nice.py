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

debut_baseline =573
date_injection =861
date_crise = 1459
sig, srate, unit, timevector = read_one_mouse_from_nc('258 j10 220624')

resp_bis = iirfilt(sig, srate, lowcut = 0.5, highcut = 30)
resp, resp_cycles = physio.compute_respiration(resp_bis, srate, parameter_preset='rat_plethysmo') 
inspi_index = resp_cycles['inspi_index'].values
expi_index = resp_cycles['expi_index'].values

fig,axs = plt.subplots(nrows=2,sharex=True)

'''
ax=axs[0]
ax.plot(timevector,resp)
ax.scatter(timevector[inspi_index], resp[inspi_index], marker='o', color='green')
ax.scatter(timevector[expi_index], resp[expi_index], marker='o', color='red')
'''

ax=axs[0]
ax.plot(resp_cycles['inspi_time'],60/resp_cycles['cycle_duration'])
ax.set_title('Frequence respiratoire ')
ax.axhline(y=491,color='k',linestyle='--')
ax.axvline(x=debut_baseline, color='g', linestyle='--')
ax.axvline(x=date_injection, color='g', linestyle='--') 
ax.axvline(x=date_crise, color='r', linestyle='--') 
ax.set_ylabel('frequence en nombre de cycle par minute')
ax.plot()

ax=axs[1]
ax.plot(resp_cycles['inspi_time'],resp_cycles['total_amplitude'])
ax.set_title('Amplitude respiratoire')
ax.axvline(x=debut_baseline, color='g', linestyle='--')
ax.axvline(x=date_injection, color='g', linestyle='--') 
ax.axvline(x=date_crise, color='r', linestyle='--') 
ax.set_ylabel('Amplitude totale par cycle ')
ax.plot()


plt.show()



def calculate_means(resp_cycles, post_ictal, end_time):
    resp_post_ictal = resp_cycles[(resp_cycles['inspi_time'] >= post_ictal) & (resp_cycles['inspi_time'] <= end_time)]
    resp_post_ictal['minute'] = ((resp_post_ictal['inspi_time'] - post_ictal) // 60).astype(int)
    grouped_data = resp_post_ictal.groupby('minute').mean().reset_index()
    grouped_data['respiratory_rate'] = 60 / grouped_data['cycle_duration']
    return grouped_data

def plot_graphs(grouped_data, metric, ylabel, title):
    plt.figure(figsize=(10, 6))
    plt.bar(grouped_data['minute'], grouped_data[metric], width=0.8, align='center', color='skyblue', edgecolor='black')
    plt.xlabel('Minutes post-ictale')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(grouped_data['minute'])
    plt.show()

grouped_data = calculate_means(resp_cycles, date_crise, 2170)
plot_graphs(grouped_data, 'respiratory_rate', 'Fréquence respiratoire (cycles par minute)', 'Évolution de la fréquence respiratoire minute par minute ')
plot_graphs(grouped_data, 'total_amplitude', 'Amplitude moyenne (unités)', 'Évolution de l\'amplitude moyenne minute par minute (20 minutes post-ictale)')