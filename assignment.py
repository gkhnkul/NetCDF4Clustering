import matplotlib.pyplot as plt
import matplotlib.colors as clt
import numpy as np
import pandas as pd
import re
import os
from datetime import datetime, timedelta
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn import preprocessing
from netCDF4 import MFDataset
from netCDF4 import Dataset



def walktree(top):
	values = top.groups.values()
	yield values
	for value in top.groups.values():
		for children in walktree(value):
			yield children

	
def KMeanClusteringNormalized(finalIntervalAthPressure, finalIntervalRH, finalIntervalMeanTemp, finalIntervalTime, dayTime):
	
	print len(finalIntervalAthPressure)
	print len(finalIntervalRH)
	print len(finalIntervalMeanTemp)
	print len(finalIntervalTime)
	print len(dayTime)
	
	df = pd.DataFrame({
    	'athPressure': finalIntervalAthPressure,
    	'rh': finalIntervalRH,
    	'temp' : finalIntervalMeanTemp,
    	'cumilative_time' : finalIntervalTime,
    	'day_time' : dayTime
	})
	
	df['athPressure'] = (df['athPressure'] - df['athPressure'].mean()) / (df['athPressure'].max() - df['athPressure'].min())
	df['rh'] = (df['rh'] - df['rh'].mean()) / (df['rh'].max() - df['rh'].min())
	df['temp'] = (df['temp'] - df['temp'].mean()) / (df['temp'].max() - df['temp'].min())
	
	#df = preprocessData(df)
	
	X = np.matrix(zip(df['athPressure'].values, df['rh'].values, df['temp'].values))
	
	k = getK(X)
	
	print "Using " + str(k) + " clusters"
	
	kmeans = KMeans(n_clusters=k).fit(X)
	
	labels = kmeans.predict(X)
	centroids = kmeans.cluster_centers_
	
	colmap = []
	for name, color in clt.cnames.iteritems():
		colmap.append(color)
		#print color
	
	colors = map(lambda x: colmap[x], labels)
	#alphas = map(lambda x: values[x], df['day_time'])
	#f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
	#ax1.scatter(df['athPressure'], df['rh'], color=colors, alpha=0.5, edgecolor='k')
	#for idx, centroid in enumerate(centroids):
	#	ax1.scatter(*centroid, color=colmap[idx+1])
	#ax2.scatter(df['rh'], df['temp'], color=colors, alpha=0.5, edgecolor='k')
	#for idx, centroid in enumerate(centroids):
	#	ax2.scatter(*centroid, color=colmap[idx+1])
	#ax3.scatter(df['athPressure'], df['temp'], color=colors, alpha=0.5, edgecolor='k')
	#for idx, centroid in enumerate(centroids):
	#	ax3.scatter(*centroid, color=colmap[idx+1])
	#ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c='#050505', s=1000)
	
	fig = plt.figure()
	ax = Axes3D(fig)
	ax.scatter(df['athPressure'], df['rh'], df["temp"], c=colors)
	for idx, centroid in enumerate(centroids):
		#ax.scatter(*centroid, marker='*', color=colmap[idx+1])
		ax.scatter(*centroid, marker='*', c='#050505')

	plt.show()
	
def KMeanClustering(finalIntervalAthPressure, finalIntervalRH, finalIntervalMeanTemp, finalIntervalTime, dayTime):
	
	print len(finalIntervalAthPressure)
	print len(finalIntervalRH)
	print len(finalIntervalMeanTemp)
	print len(finalIntervalTime)
	print len(dayTime)
	
	df = pd.DataFrame({
    	'athPressure': finalIntervalAthPressure,
    	'rh': finalIntervalRH,
    	'temp' : finalIntervalMeanTemp,
    	'cumilative_time' : finalIntervalTime,
    	'day_time' : dayTime
	})
	
	X = np.matrix(zip(df['athPressure'].values, df['rh'].values, df['temp'].values))
	
	k = getK(X)
	
	print "Using " + str(k) + " clusters"
	
	kmeans = KMeans(n_clusters=k).fit(X)
	
	labels = kmeans.predict(X)
	centroids = kmeans.cluster_centers_
	
	colmap = []
	for name, color in clt.cnames.iteritems():
		colmap.append(color)
		#print color
	
	colors = map(lambda x: colmap[x], labels)
	#alphas = map(lambda x: values[x], df['day_time'])
	#f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
	#ax1.scatter(df['athPressure'], df['rh'], color=colors, alpha=0.5, edgecolor='k')
	#for idx, centroid in enumerate(centroids):
	#	ax1.scatter(*centroid, color=colmap[idx+1])
	#ax2.scatter(df['rh'], df['temp'], color=colors, alpha=0.5, edgecolor='k')
	#for idx, centroid in enumerate(centroids):
	#	ax2.scatter(*centroid, color=colmap[idx+1])
	#ax3.scatter(df['athPressure'], df['temp'], color=colors, alpha=0.5, edgecolor='k')
	#for idx, centroid in enumerate(centroids):
	#	ax3.scatter(*centroid, color=colmap[idx+1])
	#ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c='#050505', s=1000)
	
	fig = plt.figure()
	ax = Axes3D(fig)
	ax.scatter(df['athPressure'], df['rh'], df["temp"], c=colors)
	for idx, centroid in enumerate(centroids):
		#ax.scatter(*centroid, marker='*', color=colmap[idx+1])
		ax.scatter(*centroid, marker='*', c='#050505')

	plt.show()
	
def getK(data):
	retVal = 0
	highest = 0

	for k in [2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,30]:
		kmeans = KMeans(n_clusters=k).fit(data)
	
		labels = kmeans.predict(data)
		centroids = kmeans.cluster_centers_
	
		sil = silhouette_score(data, labels, metric='cosine')
		print "For number of clusters: " + str(k) + " average silhouette score:" + str(sil)
		if highest < sil:
			highest = sil
			retVal = k
	return retVal
    	
            
def addSeconds(inputTime, secs):
	date = inputTime.date()
	time = inputTime.time()
	currentTime = inputTime + timedelta(seconds=secs)
	
	return currentTime
	
	
def importDataFromFile(fileName, timeIndex):
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
	updatedTime = []
	dayTime = []
	
	
	#for i in range(len(everySecondTime)):
	#	addSeconds(exactTime, everySecondTime[i])

	for p in fiveSecondTime:
		if len(timeIndex) > 0:
			updatedTime.append(timeIndex[-1] + 300 + p)
		else:
			updatedTime.append(p)
	
	updatedAthPressure = [np.average(athPressure[i:i+5]) for i in range(0, len(athPressure), 5)]
	updatedRH = [np.average(rh[i:i+5]) for i in range(0, len(rh), 5)]
	updatedMeanTemp = [np.average(meanTemp[i:i+5]) for i in range(0, len(meanTemp), 5)]
	updatedTime = updatedTime
	dayTime = fiveSecondTime
	
	return updatedAthPressure, updatedRH, updatedMeanTemp, updatedTime, dayTime
	
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
	
	dataset = Dataset("sgpmetavgE13.b1.20180101.000000.cdf", "w", format="NETCDF4")
	
	dataset.command_line = command_line
	dataset.process_version = process_version
	dataset.dod_version = dod_version
	dataset.input_source = input_source
	dataset.site_id = site_id
	dataset.platform_id = platform_id
	dataset.facility_id = facility_id
	dataset.data_level = data_level
	dataset.location_description = location_description
	dataset.datastream = datastream
	dataset.serial_number = serial_number
	dataset.sampling_interval = sampling_interval
	dataset.averaging_interval = averaging_interval
	dataset.averaging_interval_comment = averaging_interval_comment
	dataset.tbrg = tbrg
	dataset.pwd = pwd
	dataset.wind_speed_offset = wind_speed_offset
	dataset.wind_speed_slope = wind_speed_slope
	dataset.tbrg_precip_corr_info = tbrg_precip_corr_info
	dataset.qc_bit_comment = qc_bit_comment
	dataset.qc_bit_1_description = qc_bit_1_description
	dataset.qc_bit_1_assessment = qc_bit_1_assessment
	dataset.qc_bit_2_description = qc_bit_2_description
	dataset.qc_bit_2_assessment = qc_bit_2_assessment
	dataset.qc_bit_3_description = qc_bit_3_description
	dataset.qc_bit_3_assessment = qc_bit_3_assessment
	dataset.qc_bit_4_description = qc_bit_4_description
	dataset.qc_bit_4_assessment = qc_bit_4_assessment
	dataset.history = history
	
	time = dataset.createDimension("time", len(finalIntervalTime))
    
	atmospheric_pressure = dataset.createVariable('atmospheric_pressure', np.float32, ('time',))
	relative_humidity = dataset.createVariable('relative_humidity', np.float32,  ('time',))
	mean_temperature = dataset.createVariable('mean_temperature', np.float32,  ('time',))
	time_value = dataset.createVariable('time', np.float32,  ('time',))
	
	atmospheric_pressure[:] = finalIntervalAthPressure
	relative_humidity[:] = finalIntervalRH
	mean_temperature[:] = finalIntervalMeanTemp
	time_value[:] = finalIntervalTime
	#time[:] = finalIntervalTime
	
	dataset.close()
	
def readCDFData():
	finalData = Dataset("sgpmetavgE13.b1.20180101.000000.cdf", "r", format="NETCDF4")
			
	athPressure = finalData.variables["atmospheric_pressure"][:]
	rh = finalData.variables["relative_humidity"][:]
	meanTemp = finalData.variables["mean_temperature"][:]
	time = finalData.variables["time"][:]
	
	finalData.close()

def main():
	
	i = 1
	
	finalIntervalAthPressure = []
	finalIntervalRH = []
	finalIntervalMeanTemp = []
	finalIntervalTime = []
	finalDayTime = []
	
	while os.path.exists(os.getcwd() + "/sgpmetE13.b1.2018010%s.000000.cdf" % i):
		
		updatedAthPressure, updatedRH, updatedMeanTemp, updatedTime, dayTime = importDataFromFile("sgpmetE13.b1.2018010%s.000000.cdf" % i, finalIntervalTime)
		
		finalIntervalAthPressure.extend(updatedAthPressure)
		finalIntervalRH.extend(updatedRH)
		finalIntervalMeanTemp.extend(updatedMeanTemp)
		finalIntervalTime.extend(updatedTime)
		finalDayTime.extend(dayTime)
		
		i += 1
	
	createCDFFile(finalIntervalAthPressure, finalIntervalRH, finalIntervalMeanTemp, finalIntervalTime)
	
	#readCDFData()
	
	KMeanClustering(finalIntervalAthPressure, finalIntervalRH, finalIntervalMeanTemp, finalIntervalTime, finalDayTime)
	
	KMeanClusteringNormalized(finalIntervalAthPressure, finalIntervalRH, finalIntervalMeanTemp, finalIntervalTime, finalDayTime)
	
	
	
main()





