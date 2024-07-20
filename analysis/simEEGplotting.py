from netpyne import sim
from netpyne.plotting import plotCSDPSD
from netpyne.support.morlet import MorletSpec, index2ms
from netpyne.analysis import spikes_legacy
from simTools import simTools
import os
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use("MacOSX")
from matplotlib import pyplot as plt
from lfpykit.eegmegcalc import NYHeadModel

batch = 'CandTest0708'  # Name of batch for fig saving

stim_on = 3000
# calcEEG = {'start': 3624, 'stop': 4124}
# filter = {'lowCut':2, 'hiCut': 12}
# plotERP = {'useFilter': True}
# plotSpectrogram = {'useFilter': False}
# plotPSD = {'useFilter': True}
# plotRaster = {'timeRange': [0, 6000]}
# PSDSpect = {'timeRange': [3000, 4000], 'useLFP': False, 'useCSD': True}
# plotMUA = {'populations': ['TC', 'IRE', 'ITP4', 'ITS4'], 'stimDur': 200}

calcEEG = False
filter = False
plotERP = False
plotSpectrogram = False
plotPSD = False
plotRaster = False
PSDSpect = False
plotMUA = False

# Load sim EEG data
base_dir = '/Users/scottmcelroy/A1_scz/A1_sim_data/' + batch + '/'  # Define dir from saved data dir
for file in os.listdir(base_dir): # loop through that folder
    if file.endswith('_data.pkl') or file.endswith('_data.json'): # make sure you only download output data
        # Valery did this and fixed some problems, not sure why necessary
        sim.initialize()
        all = sim.loadAll(os.path.join(base_dir, file))
        fname = file[0:-9] #+ '_11_' # Create filename (can change to whatever)
        if not os.path.exists('/Users/scottmcelroy/A1_scz/A1_figs/SIMfigs/' + batch):
            os.mkdir( '/Users/scottmcelroy/A1_scz/A1_figs/SIMfigs/' + batch)  # Create Figure directory if one doesn't already exist

        sim.analysis.plotSpikeStats(stat = ['rate', 'isicv'], saveFig=('/Users/scottmcelroy/A1_scz/A1_figs/SIMfigs/' + fname))
        # Calculate EEG signal at one electode (currently set to 'Cz'
        if calcEEG:
            stim_data, stim_window = simTools.calculateEEG(
                sim,
                start=calcEEG['start'],
                end=calcEEG['stop'])  # Generate EEG data from dipole sums
            # Time vector starting at t=0 instead of timeRange[0]
            offsetStart = calcEEG['start'] - stim_on
            endWindow = calcEEG['stop'] - stim_on
            t = np.arange(offsetStart, endWindow, 0.05)

        # Filter EEG data
        if filter:
            filtered_data = simTools.filterEEG(
                stim_data,
                lowcut=filter['lowCut'],
                highcut=filter['hiCut'],
                fs=20000,
                order=2)  # Filter (if needed)

        # Plot ERP '
        if plotERP:
            if plotERP['useFilter'] == True:
                simTools.plotERP(
                    data = filtered_data,
                    time = stim_window,
                    fname = fname,
                    batch = batch)  # Create ERP plot of time window specified

            else:
                simTools.plotERP(
                    data = stim_data,
                    time = t,
                    fname = fname,
                    batch = batch)

        # Plot EEG Spectrogram
        if plotSpectrogram:
            if plotSpectrogram['useFilter'] == True:
                simTools.plot_spectrogram(
                    data=filtered_data,
                    time=t,
                    fname=fname,
                    batch=batch)

            else:
                simTools.plot_spectrogram(
                    data=stim_data,
                    time=t,
                    fname=fname,
                    batch=batch)

        # Plot EEG PSD
        if plotPSD:
            if plotPSD['useFilter'] == True:
                simTools.plot_PSD(
                    data=filtered_data,
                    fname=fname,
                    batch=batch)

            else:
                simTools.plot_PSD(
                    data=stim_data,
                    fname=fname,
                    batch=batch)

        # Plot Raster
        if plotRaster:
                sim.analysis.plotRaster(
                    include=[sim.cfg.allThalPops + sim.cfg.allCorticalPops],
                    orderInverse=True,
                    timeRange=plotRaster['timeRange'],
                    markerSize=50,
                    figSize=(25, 25),
                    saveFig='/Users/scottmcelroy/A1_scz/A1_figs/SIMfigs/'
                            + batch + '/' + fname + '_Raster.png')


        if PSDSpect:
            simTools.plotPSDSpectrogram(
                    sim = sim,
                    batch = batch,
                    fname = fname,
                    timeRange = PSDSpect['timeRange'],
                    useLFP = PSDSpect['useLFP'],
                    useCSD = PSDSpect['useCSD'])


        if plotMUA:
            simTools.plotMUApops(
                sim = sim,
                populations = plotMUA['populations'],
                bin_start_times = [3000, 3724, 4448, 5172],
                bin_duration = plotMUA['stimDur'],
                batch = batch,
                fname = fname)



            # sim.plotting.plotCSD(overlay = 'LFP', timeRange =[3000, 3600], saveFig='/Users/scottmcelroy/A1_scz/A1_figs/SIMfigs/'
            #                 + batch + '/' + fname + '_CSD.png')
# # Plot LFP PSD
#         sim.analysis.plotLFP(plots = 'PSD', saveFig= '/Users/scottmcelroy/A1_scz/A1_figs/SIMfigs/'+batch+ '/'+fname+ 'LFP.png')
# # Plot CSD
#         sim.plotting.plotCSD(overlay= 'CSD', timeRange=[2500, 5000],saveFig='/Users/scottmcelroy/A1_scz/A1_figs/SIMfigs/' + batch+'/'+fname+'CSDpad.jpeg')
#         spikes_legacy.plotRatePSD(include = ['IT2', 'IT3', 'ITP4', 'ITS4', 'IT5A', 'CT5A', 'IT5B', 'CT5B' , 'PT5B', 'IT6', 'CT6'],timeRange=[3000, 5000],  saveFig='/Users/scottmcelroy/Desktop/ASSRratePSDAll.png')
#         spikes_legacy.plotSpikeHist(include = ['cochlea', 'TC'], timeRange=[3000, 6000], saveFig='/Users/scottmcelroy/A1_scz/A1_figs/SIMfigs/' + batch + '/' + fname + 'RateHisto')
        # spikes_legacy.plotSpikeStats(saveFig= '/Users/scottmcelroy/A1_scz/A1_figs/SIMfigs/' + batch + '/' + fname + 'SpikeStats')
# spikes_legacy.plotRatePSD(include = ['IT5A', 'CT5A'], timeRange=[3000, 5000], saveFig='/Users/scottmcelroy/Desktop/ASSRratePSDL5A.png')
# spikes_legacy.plotRatePSD(include = ['IT5B', 'CT5B' , 'PT5B'], timeRange=[3000, 5000], saveFig='/Users/scottmcelroy/Desktop/ASSRratePSDL5B.png')
# spikes_legacy.plotRatePSD(include = ['IT6', 'CT6'], timeRange=[3000, 5000], saveFig='/Users/scottmcelroy/Desktop/ASSRratePSDL6.png')


################## Scrap for resampling if needed later ##################################################
# num_samples = len(stim_data)
# new_num_samples = int(num_samples * 1000/ 20000)
# resampled_data = signal.resample(stim_data, new_num_samples)