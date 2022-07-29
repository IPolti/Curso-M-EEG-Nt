# -*- coding: utf-8 -*-
"""
Introducción a MNE-Python

Autor original: Alexandre Gramfort & Denis A. Engemann
Modificado por: Ignacio Polti
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

import mne  # Carga la librería MNE-Python
mne.set_log_level('warning')  # Definir el nivel de umbral de mensajes

# RECORDATORIO: Si necesitás ayuda... preguntale a la computadora!
mne.pick_types?

# =============================================================================
# === M/EEG TP: Del dato crudo a los potenciales evocados =====================
# =============================================================================

# %% Acceder al dato crudo
# Para empezar el TP, la carpeta ds000117 tiene que estar instalada en la PC
"""
 Los datos con los que vamos a trabajar pertenecen al dataset publicado aquí:
https://www.nature.com/articles/sdata20151
"""

# Cambia el acceso a dónde instalaste la carpeta ds000117
data_path = os.path.join("D:/CursoNt/Data/")
subj_path = dict({'sub-01': 'sub-01/ses-meg/meg/'
                  'sub-01_ses-meg_task-facerecognition_run-01_meg.fif',
                  'sub-02': 'sub-02/ses-meg/meg/'
                  'sub-01_ses-meg_task-facerecognition_run-01_meg.fif'})

# Seleccionar que participante queremos usar para analizar sus datos (01 o 02)
raw_fname = os.path.join(data_path, subj_path['sub-01'])
print(raw_fname)



# %%  === Leer los datos del archivo ==========================================
# =============================================================================
mne.io.read_raw_fif?

raw = mne.io.read_raw_fif(raw_fname, preload=False)
# [!!] 'preload=False' especifica que los datos no están en la memoria
print(raw)

"""
Más información sobre como importar datos de EEG:
    https://mne.tools/stable/auto_tutorials/io/plot_20_reading_eeg_data.html
"""

"""
Ahora vamos a explorar la estructura 'info'. Nos va a dar detalles de:
    - frecuencia de sampleo
    - parámetros de filtrado
    - tipos de canales disponibles
    - canales malos
    - etc.
"""
print(raw.info)

# %%% Ejercicio
"""
1) Cuántos canales existen para cada tipo de sensor?
2) Cuál es la frecuencia de sampleo?
3) Están filtrados los datos?
4) Cuál es la frecuencia del ruido de la linea eléctrica?
5) Hay algún canal malo?

Tip: raw.info? es tu amigo ;)

INSERTAR RESPUESTAS DEBAJO
"""



# %% === Definir tipos de canales =============================================
# =============================================================================
"""
Los investigadores que proveen el dataset nos avisan que debido a un problema
de software algunos canales están mal definidos como canales de EEG (sección
"M/EEG acquisition": https://www.nature.com/articles/sdata20151):
- 2 canales (EEG061 y EEG062) son electrooculogramas (HEOG y VEOG)
- el canal EEG063 es un electrocardiograma (ECG)
- el canal EEG064 nunca se conectó a la cabeza del participante (ups!)

Definir adecuadamente estos canales nos va a simplificar la tarea de eliminar
artefactos en la señal EEG
"""
raw.set_channel_types?

# Ver los sensores en espacio 3D
raw.plot_sensors(kind='3d',
                 ch_type='eeg',
                 ch_groups='position')
# Ver los sensores en espacio 2D
raw.plot_sensors(kind='topomap',
                 ch_type='eeg',
                 ch_groups='position')  # sphere=(0.001, 0.01, 0, 0.085)

# Redefinir los canales por su clasificación correcta
raw.set_channel_types({'EEG061': 'eog',
                       'EEG062': 'eog',
                       'EEG063': 'ecg',
                       'EEG064': 'misc'})  # electrodo flotando en el aire (!)

# Renombrar los canales por su nombre correcto
raw.rename_channels({'EEG061': 'EOG061',
                     'EEG062': 'EOG062',
                     'EEG063': 'ECG063'})

# Identificar el canal de EEG 064 como malo
raw.info['bads'] = ['EEG064']
raw.info
raw.plot_sensors(kind='topomap', ch_type='eeg')



# %% === Acceder a los datos ==================================================
# =============================================================================
"""
Para acceder a los datos vamos a usar [] para explorar los elementos de una
lista, diccionario, etc.
"""
# Definir el valor inicial y final del registro de EEG
start, stop = 0, 10
# Recuperar todos los canales y los primeros 10 puntos de registro
data, times = raw[:, start:stop]
print(data.shape)
print(times.shape)
times  # En Python, por convención el tiempo SIEMPRE empieza en 0

# raw[] devuelve los valores del registro EEG Y el tiempo
plt.plot(raw[0][1][:10], raw[0][0][0, :10])



# %% === Cambiar la frecuencia de sampleo =====================================
# =============================================================================

# Por qué? Para acelerar el tiempo de procesamiento
raw.load_data()  # Carga los datos en la memoria
raw.resample(300)  # Cambia la frecuencia de sampleo a 300Hz



# %% === Remover canales innecesarios =========================================
# =============================================================================
raw.drop_channels?

to_drop = ['STI201', 'STI301', 'MISC201', 'MISC202', 'MISC203',
           'MISC204', 'MISC205', 'MISC206', 'MISC301', 'MISC302',
           'MISC303', 'MISC304', 'MISC305', 'MISC306']

raw.drop_channels(to_drop)
raw.info
# Seleccionar los canales de EEG (ignorar los canales de MEG)
raw.pick(['eeg', 'eog', 'ecg', 'stim', 'misc'])



# %% === Visualizar los datos crudos ==========================================
# =============================================================================
"""
Con la función de visualización podemos:
    - navegar los datos
    - activar/desactivar las proyecciones del PCA/SSP
    - marcar segmentos del trazado EEG malos en anotaciones
    - agrupar canales por tipos
    - agrupar canales por localización
"""
raw.plot?

raw.plot()

# %%% Ejercicio
"""
1) Ves algún canal malo?
2) Ves algún segmento malo en el registro EEG ?
3) Ves algo más que artefactos generados por parpadeos?

INSERTAR RESPUESTAS DEBAJO
"""



# %% === Filtrado =============================================================
# =============================================================================
raw.filter?

# %%% Ejercicio
"""
Filtrar los datos crudos entre 0Hz y 40Hz

INSERTAR RESPUESTAS DEBAJO
"""



# %% === Canal de presentación de estímulos ===================================
# =============================================================================
"""
Graficar los primeros 50 segundos del canal de estimulación
Tips:
    - el nombre del canal de estimulación lo encuentran en raw.info.ch_names
    - extraer el registro del canal de estimulación usando raw.get_data()
    - graficar usando plt.plot()

INSERTAR RESPUESTAS DEBAJO
"""



# %% === Definir y leer épocas de registro ====================================
# =============================================================================

# Primero tenemos que extraer los eventos (estímulos presentados)
events = mne.find_events(raw, stim_channel='STI101', verbose=True)

# %%% Ejercicio
"""
1) Qué tipo de variable es "events"?
2) Cuál es el significado de cada una de las columnas de "events"?
  Tip: https://mne.tools/stable/auto_tutorials/intro/10_overview.html
3) Cuántos eventos de tipo "5" hay?

INSERTAR RESPUESTAS DEBAJO
"""



# %% === Ajustar tiempo de presentación de los eventos ========================
# =============================================================================
"""
En la sección "M/EEG acqusition" (https://www.nature.com/articles/sdata20151)
se especifica que hubo un delay fijo de 34ms entre el registro de la señal de
presentación de estímulo en el archivo MEG (canal STI101) y la aparición del
estímulo en la pantalla.
"""
delay = int(round(0.0345 * raw.info['sfreq']))
events[:, 0] = events[:, 0] + delay



# %% === Visualización del paradigma experimental =============================
# =============================================================================

events = events[events[:, 2] < 20]  # tomar sólo los eventos con código < 20
fig = mne.viz.plot_events(events, raw.info['sfreq'], first_samp=raw.first_samp)

"""
Para eventos de presentación de estímulos y condiciones usamos un diccionario
de Python con elementos que contiene "/" para agrupar sub-condiciones
"""
event_id = {
    'face/famous/first': 5,
    'face/famous/immediate': 6,
    'face/famous/long': 7,
    'face/unfamiliar/first': 13,
    'face/unfamiliar/immediate': 14,
    'face/unfamiliar/long': 15,
    'scrambled/first': 17,
    'scrambled/immediate': 18,
    'scrambled/long': 19,
}

# Visualizar sólo los eventos
fig = mne.viz.plot_events(events,
                          sfreq=raw.info['sfreq'],
                          event_id=event_id,
                          first_samp=raw.first_samp)

# Visualizar todos los canales + eventos
raw.plot(event_id=event_id,
         events=events,
         event_color={5: 'blue', 6: 'orange', 7: 'green',
                      13: 'red', 14: 'purple', 15: 'brown',
                      17: 'pink', 18: 'gray', 19: 'olive'})

# Visualizar sólo el canal de estimulación + eventos
raw_tmp = raw.copy().pick_types(stim=True)
raw_tmp.plot(event_id=event_id,
             events=events,
             event_color={5: 'blue', 6: 'orange', 7: 'green',
                          13: 'red', 14: 'purple', 15: 'brown',
                          17: 'pink', 18: 'gray', 19: 'olive'})



# %% === Definir los parámetros de las épocas =================================
# =============================================================================

tmin = -0.5  # inicio de cada época (500ms ANTES de presentación de estímulo)
tmax = 2.0  # fin de cada época (2000ms POST presentación de estímulo)



# %% === Definir el período de línea de base ==================================
# =============================================================================
baseline = (-0.2, 0)  # 200ms antes de la presentación del estímulo (t = 0)



# %% === Eliminación de artefactos en la señal M/EEG ==========================
# =============================================================================
"""
Definir la amplitud del umbral de rechazo de artefactos en los
canales del EEG y del EOG (ElectroOculoGrama). Éstos parámetros pueden variar
mucho dependiendo el dataset que tengamos. El proyecto "autoreject" tiene como
objectivo resolver el problema de la selección de parámetros de rechazo usando
algoritmos de machine-learning (https://autoreject.github.io).
"""
reject = dict(eeg=200e-6, eog=150e-6)

picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=True,
                       stim=False, exclude='bads')


# %%% === Extraer las épocas ==================================================
epochs = mne.Epochs(raw, events,
                    event_id=event_id,
                    tmin=tmin,
                    tmax=tmax,
                    proj=True,
                    picks=picks,
                    baseline=baseline,
                    reject=reject)
print(epochs)

epochs.drop_bad()  # remover épocas malas según el umbral de rechazo definido

epochs.load_data()  # cargar los datos en la memoria
"""
Explorar el objeto "epochs"

Tip: epochs.<TAB> 
"""


# %%% === Explorar cuantas épocas fueron eliminadas ===========================
epochs.plot_drop_log()

for drop_log in epochs.drop_log[:20]:
    print(drop_log)

# Tambien podemos eliminar épocas arbitrariamente
epochs.copy().drop(10, reason="Ésta no me gusta :(").plot_drop_log()
epochs.copy().drop(10, reason="Ésta no me gusta :(").drop_log[:20]

"""
PERDIMOS LA MITAD DE LAS ÉPOCAS USANDO VALORES UMBRALES... NO BUENO :(

Cómo lo mejoramos? Usando un algoritmo de Análisis de Componentes Principales
(PCA, también conocido como SSP por "Signal-Space Projection") que nos va a
permitir eliminar de la señal de EEG los componentes relacionados a artefactos
oculares, cardíacos, etc.

1) Detectamos los artefactos oculares (identificados con el EOG)
2) Visualizamos su impacto en la señal EEG
3) Estimamos que patrón característico posee en la señal EEG para removerlo
"""


# %%% === Artefactos oculares =================================================

# Explorar los artefactos oculares en los datos crudos
raw.plot()

# Crear épocas alrededor de los artefactos oculares
eog_epochs = mne.preprocessing.create_eog_epochs(
    raw.copy().filter(l_freq=1, h_freq=None))
eog_epochs.average().plot_joint()
eog_epochs.plot_image(combine='mean')

raw.plot(events=eog_epochs.events)

projs_eog, _ = mne.preprocessing.compute_proj_eog(raw, n_eeg=3, average=True)
projs_eog

mne.viz.plot_projs_topomap(projs_eog, info=raw.info)

# Comparar registro EEG con/sin remoción de artefactos oculares de la señal
for title in ('Sin', 'Con'):
    if title == 'Con':
        raw.add_proj(projs_eog)

    with mne.viz.use_browser_backend('matplotlib'):
        fig = raw.plot()
    fig.subplots_adjust(top=0.9)  # make room for title
    fig.suptitle('{} proyecciones EOG'.format(title), size='xx-large',
                 weight='bold')

"""
Cuántos componentes uno debería mantener? Alguno de ellos no parecen ser parte
de artefactos de ruido en la señal. La buena noticia es que no tenemos que
decidirlo AHORA.
"""


# %%% === Artefactos cardíacos ================================================

# Explorar los artefactos cardíacos en los datos crudos
raw.plot()

# Crear épocas alrededor de los artefactos cardíacos
ecg_epochs = mne.preprocessing.create_ecg_epochs(
    raw.copy().filter(l_freq=1, h_freq=None))
ecg_epochs.average().plot_joint()
ecg_epochs.plot_image(combine='mean')

projs_ecg, _ = mne.preprocessing.compute_proj_ecg(raw, n_eeg=3, average=True)
projs_ecg

mne.viz.plot_projs_topomap(projs_ecg, info=raw.info)

# Comparar registro EEG con/sin remoción de artefactos cardíacos de la señal
for title in ('Sin', 'Con'):
    if title == 'Con':
        raw.add_proj(projs_ecg)

    with mne.viz.use_browser_backend('matplotlib'):
        fig = raw.plot()
    fig.subplots_adjust(top=0.9)  # make room for title
    fig.suptitle('{} proyecciones ECG'.format(title), size='xx-large',
                 weight='bold')


# %%% === Remoción de artefactos oculares y cardíacos =========================

# Ahora veamos como podría mejorar "en teoría" la preservación de los datos
reject2 = dict(eeg=reject['eeg'])

epochs_clean = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False,
                          picks=picks, baseline=baseline,
                          preload=False,
                          reject=reject2)

# Agregar las proyecciones al objeto "epochs"
epochs_clean.add_proj(projs_eog + projs_ecg)

# Comparar las épocas con/sin remoción de artefactos EOG y ECG
for title in ('Sin', 'Con'):
    with mne.viz.use_browser_backend('matplotlib'):
        if title == 'Con':
            epochs_clean.copy().add_proj(projs_eog + projs_ecg)
            fig = epochs_clean.copy().apply_proj().average().plot(spatial_colors=True)
        else:
            fig = epochs_clean.average().plot(spatial_colors=True)
    fig.subplots_adjust(top=0.9)  # make room for title
    fig.suptitle('{} proyecciones EOG + ECG'.format(title), size='large',
                 weight='bold')

# Sobre-escribimos las épocas
epochs = epochs_clean

# %%% Ejercicio
"""
- Usar ICA en vez de PCA/SSP para remover artefactos
- Cuáles son los potenciales beneficios o desventajas?
Tips: https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html

INSERTAR RESPUESTAS DEBAJO
"""



# %% === Visualizar las épocas ================================================
# =============================================================================
epochs.plot_psd(picks='eeg', fmax = 40)
epochs.plot_psd_topomap(ch_type='eeg')
# Ejemplo de como hacer una imágen de Potenciales Evocados (ERP)
epochs.plot_image(picks='EEG065', sigma=1.)

# Graficar las épocas crudas con los eventos
epochs.plot(events=events)

"""
El objeto 'epochs' es la navaja suiza de MNE-Python para procesar datos
segmentados
    - Métodos especializados de visualización diagnóstica de los datos
    - Estimación de promedios
    - Grabar datos en el disco
    - Manipular datos, p.ej. reordenar o eliminar trials individuales,
      resampleo, etc.
"""

# %%% Ejercicio
"""
Cómo podrías obtener las épocas que corresponden a la condición:
    1) "face"?
    2) "famous face"?
    3) "scrambled"?

Tip: https://mne.tools/stable/auto_tutorials/epochs/10_epochs_overview.html
    
INSERTAR RESPUESTAS DEBAJO
"""



# %% === IO (Input/Output) básico =============================================
# =============================================================================
"""
El escenario estándar es grabar las épocas en un archivo .fif junto con la
información de cabecera
"""
epochs_fname = raw_fname.replace('_meg.fif', '-epo.fif')
epochs_fname

# notar que las épocas se graban en archivos que terminan con -epo.fif
epochs.save(epochs_fname, overwrite=True)

data = epochs.get_data()
data.shape



# %% === Promedio de las épocas para obtener un ERP y visualizarlo ============
# =============================================================================
evoked = epochs.average()
evoked.del_proj()  # borramos las proyecciones anteriores

# tomamos las primeras 3 proyecciones para cada tipo de sensor
evoked.add_proj(projs_eog[::3] + projs_ecg[::3])
evoked.apply_proj()  # aplicamos las proyecciones

"""
GFP (Global Field Power) es una medida de correlación de las señales
obtenidas por todos los sensores del EEG: si todos los sensores tienen el mismo
valor en un determinado momento, el GFP=0; si las señales difieren, el GFP será
distinto de 0. Picos en la medida GFP pueden reflejar actividad cerebral
"interesante" que conviene explorar. GFP = sd(sensores) * tiempo
"""
evoked.plot(spatial_colors=True, proj=True, gfp=True)
times = [0.0, 0.1, 0.18]
evoked.plot_topomap(ch_type='eeg', times=times, proj=True)
evoked.plot_topomap(ch_type='eeg', times=np.linspace(0.05, 0.45, 8), proj=True)

# %%% Ejercicio
"""
Cómo la proyección SSP impacta en los ERPs? Usa proj='interactive' para
explorar

INSERTAR RESPUESTAS DEBAJO
"""

# También podemos visualizar en una misma figura la topografía y el trazado
evoked.plot_joint(times=[0.17])



# %% === Acceder e indexar épocas por condición ===============================
# =============================================================================
"""
Las épocas pueden ser indexadas con números enteros o rodajas para seleccionar
un subgrupo de épocas, pero también se puede indexar con texto si queremos
seleccionar por condiciones  experimentales: epochs['condition']
"""
epochs[0]  # primera época
epochs[:10]  # primeras 10 épocas
epochs['face']  # épocas de la condición 'face'

"""
En event_id, '/' selecciona condiciones de manera jerárquica, p. ej.
'face' vs 'scrambled', 'famous' vs 'unfamiliar', y MNE-python las puede
seleccionar individualmente.
"""
epochs['face'].average().\
    pick_types(eeg=True).crop(-0.1, 0.25).plot(spatial_colors=True)

# Aplicamos ésto para visualizar todas las condiciones en event_id
for condition in ['face', 'scrambled']:
    epochs[condition].average().plot_topomap(times=[0.1, 0.15],
                                             title=condition)



# %% === Escribir los ERPs en el disco ========================================
# =============================================================================
evoked_fname = raw_fname.replace('_meg.fif', '-ave.fif')
evoked_fname

# notar que los ERPs se graban en archivos que terminan con -ave.fif
evoked.save(evoked_fname)

# También podemos grabar muchas condiciones en un archivo
evokeds_list = [epochs[k].average() for k in event_id]  # obtener los ERPs
mne.write_evokeds(evoked_fname, evokeds_list)



# %% === Leer los ERPs del disco ==============================================
# =============================================================================
# Es posible descargar los ERPs guardados en un archivo .fif
evokeds_list = mne.read_evokeds(evoked_fname, baseline=(None, 0), proj=True)

# O dar el nombre explícito de la condición promediada
evoked1 = mne.read_evokeds(evoked_fname, condition="face/famous/first",
                           baseline=(None, 0), proj=True)



# %% === Estimar un contraste de ERPs =========================================
# =============================================================================
evoked_face = epochs['face'].average()
evoked_scrambled = epochs['scrambled'].average()

contrast = mne.combine_evoked([evoked_face, evoked_scrambled], [0.5, -0.5])
"""
Notar que esto combina los ERPs teniendo en consideración el número de épocas
promediadas (para escalar la varianza del ruido)
"""
print(evoked.nave)  # average of 12 epochs
print(contrast.nave)  # average of 116 epochs

print(contrast)

fig = contrast.copy().pick('eeg').crop(-0.1, 0.3).plot_joint()
evoked_face
evoked_scrambled
contrast.plot_topomap(times=np.linspace(0.05, 0.15, 5), ch_type='eeg')


# %%% === Grabar tus visualizaciones como pdf =================================
contrast.plot_topomap(times=np.linspace(0.05, 0.15, 5), ch_type='eeg')
plt.savefig('toto.pdf')

# %%% Ejercicio
"""
1) Estimar los ERPs para las caras "famous", "unfamiliar" y "scrambled"
2) Cortar los datos entre -0.1s y 0.4s usando la función .crop()
3) Visualizar el canal EEG065 en las tres condiciones usando la función
   mne.viz.plot_compare_evokeds

Tip: https://mne.tools/stable/auto_tutorials/evoked/30_eeg_erp.html#sphx-glr-auto-tutorials-evoked-30-eeg-erp-py

INSERTAR RESPUESTAS DEBAJO
"""



# %% === Análisis del espectro de frecuencia ==================================
# =============================================================================
freqs = np.logspace(*np.log10([6, 35]), num=8)
n_cycles = freqs / 2.  # different number of cycle per frequency
power, itc = mne.time_frequency.tfr_morlet(epochs, freqs=freqs,
                                           n_cycles=n_cycles, use_fft=True,
                                           return_itc=True, decim=3, n_jobs=1)

"""
Power: Medida de actividad en un determinado rango de frecuencia
"""
power.plot_topo(baseline=(-0.5, 0), mode='logratio', title='Average power')

fig, axis = plt.subplots(1, 2, figsize=(7, 4))
power.plot_topomap(ch_type='eeg', tmin=0.5, tmax=1.5, fmin=8, fmax=12,
                   baseline=(-0.5, 0), mode='logratio', axes=axis[0],
                   title='Alpha', show=False)
power.plot_topomap(ch_type='eeg', tmin=0.5, tmax=1.5, fmin=13, fmax=25,
                   baseline=(-0.5, 0), mode='logratio', axes=axis[1],
                   title='Beta', show=False)
mne.viz.tight_layout()
plt.show()

"""
Inter-Trial-Coherence: Medida de coherencia de fase de ritmos cerebrales
entre trials. 1=alineación de fase perfecta | 0=ditribución de fase uniforme
"""
itc.plot_topo(title='Coherencia entre trials', vmin=0., vmax=1., cmap='Reds')
itc.plot_joint(baseline=(-0.5, 0), mode='mean', tmin=-.5, tmax=2,
               timefreqs=[(.15, 10), (1., 8)], title='Coherencia entre trials')



# %% == Crear un reporte para cada participante ===============================
# =============================================================================
"""
Con MNE-Python podemos crear reportes en formato html para poder visualizar
toda la información relevante de cada participante analizado.
"""
report = mne.Report(title='Reporte sub-01',
                    raw_psd=True,  # estimar el espectro de frequencia
                    projs=True,  # incluir las proyecciones
                    verbose=True)
path = os.path.join(data_path, 'sub-01/ses-meg/meg/')

report.parse_folder(path, pattern='*01_meg.fif', render_bem=False)
report.save('reporte_basico.html', overwrite=True)

# %%% Ejercicio
"""
Hacer un reporte del análisis de potenciales evocados
tip: https://mne.tools/stable/auto_tutorials/intro/70_report.html

INSERTAR RESPUESTAS DEBAJO
"""



# %% === BONUS TRACK: Cómo hacer un gif de una topografía =====================
# =============================================================================
times = np.arange(-0.1, 1, 0.01)
fig, anim = evoked.animate_topomap(times=times, ch_type='eeg', frame_rate=30,
                                   time_unit='s', butterfly=True,
                                   blit=False, show=True)

anim.save('topo_gif2.gif', dpi=300, fps=15)
