import matplotlib.pyplot as plt
import numpy as np
import re
import os
from datetime import datetime, timedelta
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from netCDF4 import MFDataset
from netCDF4 import Dataset


def walktree(top):
	values = top.groups.values()
	yield values
	for value in top.groups.values():
		for children in walktree(value):
			yield children

def clusterData():
	plt.rcParams['figure.figsize'] = (30, 30)
	X, y = make_blobs(n_samples=len(f.variables["time"][:]), n_features=3, centers=4)
	X[:, 1] = athPressure
	X[:, 2] = rh
	X[:, 0] = meanTemp
	#X[:, 3] = f.variables["time"][:]

	y = f.variables["time"][:]

	# Initializing KMeans
	kmeans = KMeans(n_clusters=4)
	# Fitting with inputs
	kmeans = kmeans.fit(X)
	# Predicting the clusters
	labels = kmeans.predict(X)
	# Getting the cluster centers
	C = kmeans.cluster_centers_

	fig = plt.figure()
	ax = Axes3D(fig)
	ax.scatter(f.variables["atmos_pressure"][:], f.variables["rh_mean"][:], f.variables["temp_mean"][:], c=y)
	ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c='#050505', s=1000)

	plt.show()

            
def addSeconds(inputTime, secs):
	date = inputTime.date()
	time = inputTime.time()
	currentTime = inputTime + timedelta(seconds=secs)
	
	return currentTime
	
	
def importDataFromFile(fileName):
	f = MFDataset(fileName)
	
	#find the regex of 2018-01-01 00:21:01 in f.history
	historyString = f.history[:]
	
	capturedDateString = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', historyString)
	exactTime = datetime.strptime(capturedDateString.group(), '%Y-%m-%d %H:%M:%S')

	athPressure = f.variables["atmos_pressure"][:]
	rh = f.variables["rh_mean"][:]
	meanTemp = f.variables["temp_mean"][:]
	everySecondTime = f.variables["time"][:]
	fiveSecondTime = everySecondTime[0::5]
	
	for i in range(len(everySecondTime)):
		addSeconds(exactTime, everySecondTime[i])

	updatedAthPressure = [np.average(athPressure[i:i+5]) for i in range(0, len(athPressure), 5)]
	updatedRH = [np.average(rh[i:i+5]) for i in range(0, len(rh), 5)]
	updatedMeanTemp = [np.average(meanTemp[i:i+5]) for i in range(0, len(meanTemp), 5)]
	updatedTime = [] #TODO every five second
	
	return updatedAthPressure, updatedRH, updatedMeanTemp, updatedTime
	
def createCDFFile(finalIntervalAthPressure, finalIntervalRH, finalIntervalMeanTemp, finalIntervalTime):
	
	f = MFDataset("sgpmetE13.b1.20180101.000000.cdf")
	
	command_line = f.command_line[:]
	process_version = f.process_version[:]
	dod_version = f.dod_version[:]
	input_source = f.input_source[:]
	site_id = f.site_id[:]
	platform_id = f.platform_id[:]
	facility_id = f.facility_id[:]
	data_level = f.data_level[:]
	location_description = f.location_description[:]
	datastream = f.datastream[:]
	serial_number = f.serial_number[:]
	sampling_interval = f.sampling_interval[:]
	averaging_interval = "300 seconds"
	averaging_interval_comment = f.averaging_interval_comment[:]
	tbrg = f.tbrg[:]
	pwd = f.pwd[:]
	wind_speed_offset = f.wind_speed_offset[:]
	wind_speed_slope = f.wind_speed_slope[:]
	tbrg_precip_corr_info = f.tbrg_precip_corr_info[:]
	qc_bit_comment = f.qc_bit_comment[:]
	qc_bit_1_description = f.qc_bit_1_description[:]
	qc_bit_1_assessment = f.qc_bit_1_assessment[:]
	qc_bit_2_description = f.qc_bit_2_description[:]
	qc_bit_2_assessment = f.qc_bit_2_assessment[:]
	qc_bit_3_description = f.qc_bit_3_description[:]
	qc_bit_3_assessment = f.qc_bit_3_assessment[:]
	qc_bit_4_description = f.qc_bit_4_description[:]
	qc_bit_4_assessment = f.qc_bit_4_assessment[:]
	history = "created by Gokhan Kul on machine gkul at 2018-01-01 00:21:01, using python"
    
	#TODO write to file
	##GROUPS
	##dimensions:
    ##variables:
	
	
	

def main():
	
	rootgrp = Dataset("asd.cdf", "r", format="NETCDF4")
	
	#print rootgrp
	#for children in walktree(rootgrp):
	#	for child in children:
	#		print child
	
	i = 1
	
	finalIntervalAthPressure = []
	finalIntervalRH = []
	finalIntervalMeanTemp = []
	finalIntervalTime = []
	
	while os.path.exists(os.getcwd() + "/sgpmetE13.b1.2018010%s.000000.cdf" % i):
	
		print "we are in the while loop"
		
		updatedAthPressure, updatedRH, updatedMeanTemp, updatedTime = importDataFromFile("sgpmetE13.b1.2018010%s.000000.cdf" % i)
		
		print len(updatedAthPressure)
		print len(updatedRH)
		print len(updatedMeanTemp)
		print len(updatedTime)
		
		finalIntervalAthPressure.extend(updatedAthPressure)
		finalIntervalRH.extend(updatedRH)
		finalIntervalMeanTemp.extend(updatedMeanTemp)
		finalIntervalTime.extend(updatedTime)
		
		print len(finalIntervalAthPressure)
		print len(finalIntervalRH)
		print len(finalIntervalMeanTemp)
		print len(finalIntervalTime)
		
		i += 1

	createCDFFile(finalIntervalAthPressure, finalIntervalRH, finalIntervalMeanTemp, finalIntervalTime)
	
	
main()

## THERE ARE 7200 RECORDS IN ALL, 1440 RECORDS IN FIRST FILE




#sum the history with the time variable.


	


#plt.plot(f.variables["time"][:], f.variables["atmos_pressure"][:])
#plt.plot(f.variables["time"][:], f.variables["rh_mean"][:])
#plt.plot(f.variables["time"][:], f.variables["temp_mean"][:])
#plt.legend(['atmos_pressure', 'rh_mean', 'temp_mean'], loc='upper left')
#plt.xlabel('time')
#plt.ylabel('variables')
#plt.show()


#COMPUTE DERIVATIVE OF TEMP WITH TIME ARRAY
#function = (y[i+1]-y[i])/(x[i+1]-x[i]),
#d = N.diff(y)
#d = N.diff(y)/N.diff(x)
#xd = (x[1:]+x[:-1])/2.


