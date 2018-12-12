#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 15:20:17 2018

@author: Thomas
"""
#cd ~/desktop/Documents/project#only using iPython console

#import librarys
import numpy as np
from initialise import pyramidYearVect,pyramidMvalues,pyramidFvalues
import plots

#For a year plot the population pyramid 
year=2010
yearPos=sum((pyramidYearVect==year)*np.arange(pyramidYearVect.size))
x1=pyramidMvalues[yearPos,:]
y1=pyramidFvalues[yearPos,:]
plots.plotPyramid(x1,y1)

#Start projection
year=2000 #select starting year
pyramidYearPos=sum((pyramidYearVect==year)*np.arange(pyramidYearVect.size))+1
yearsHist=pyramidYearVect[0:pyramidYearPos:]
#historical data starting year included
pyramidHistM=pyramidMvalues[0:pyramidYearPos,:]
pyramidHistW=pyramidFvalues[0:pyramidYearPos,:]



nProj=10 #number of year of projetion
#Real data on the last year of projection
pyramidProjM=pyramidMvalues[pyramidYearPos+nProj-1,:]
pyramidProjW=pyramidFvalues[pyramidYearPos+nProj-1,:]

#Projection gives the pyramid for each year to the last year of projection
import projModule
(popAllM,popAllW)=projModule.projPop2(pyramidHistM,pyramidHistW,yearsHist,nProj)
#Plot the pyramid at the end of the projection with the real data
plots.plot2Pyramid(pyramidProjM,popAllM[-1,:],pyramidProjW,popAllW[-1,:])

""" Back up alternative
import projModule
import pandas
(PopM,PopW,yearVect,nProj)=(pyramidHistM,pyramidHistW,yearsHist,nProj)
(mortalityRatesM,life_aveM)=projModule.projMortalityRates(PopM,nProj)
(mortalityRatesW,life_aveW)=projModule.projMortalityRates(PopW,nProj)
'''Updated with link'''
indics = pandas.read_csv('indics2.csv')

for iProj in range(1,nProj+1):
        
        #Begin with the birth that happen during the first year
        
    #proj birth
    import statistics
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, LSTM
    
    np.random.seed(7)
    
    yearHist=PopM[:,1].size
    BirthRate=np.zeros((yearHist-1,2))#Container for the annual birth rate 
    MFratioTab=np.zeros((yearHist-1))#Container for the ratio between male and female
    for iYear in range(1,yearHist):
        #Birth rate as a % of population between 18 to 45
        BirthN = PopM[iYear,0]+PopW[iYear,0]
        AvgMF = (sum(PopW[iYear-1,18:41])+sum(PopM[iYear-1,18:41]))/2
        BirthRateY = BirthN/AvgMF
        MFratioTab[iYear-1] = PopM[iYear,0]/(PopW[iYear,0]+PopM[iYear,0])
        if iYear>1:
            BirthRate[iYear-1,1]=(AvgMF/PrevAvg)-1
            BirthRate[iYear-1,0] = BirthRateY/BirthRateP-1
        PrevAvg = AvgMF
        BirthRateP=BirthRateY
        
    BirthMFratio=statistics.mean(MFratioTab)
    #plots.plotFunL(yearsHist[0:-1],BirthRate)
            
    d=np.zeros((yearHist-2,3))
    d[:,0]=yearVect[1:-1]
    d[:,1]=BirthRate[1:,1]
    d[:,2]=BirthRate[1:,0]
        
    
    birthToMerge=pandas.DataFrame(d,columns=['Year','pop_evo', 'birthRate'])
    #birthToMerge=pandas.DataFrame(d,columns=['Year', 'birthRate'])
        
        
    all_data=indics.set_index('year').join(birthToMerge.set_index('Year'))
        
    #transform tab in order to have all inputs vary from 0 to 1
    all_data=all_data[~np.isnan(all_data["birthRate"])]#To remove columns with missing data
        
    #all_data['pp_evo'] = (all_data['pp_evo']-min(all_data['pp_evo']))/(max(all_data['pp_evo'])-min(all_data['pp_evo']))
    #all_data['unemp'] = (all_data['unemp']-min(all_data['unemp']))/(max(all_data['unemp'])-min(all_data['unemp']))
    #all_data['pop_evo'] = (all_data['pop_evo']-min(all_data['pop_evo']))/(max(all_data['pop_evo'])-min(all_data['pop_evo']))
        #For birth rate save transforming parameters
    #minB=min(all_data['birthRate'])
    #maxB=max(all_data['birthRate'])
    #all_data['birthRate'] = (all_data['birthRate']-minB)/(maxB-minB)
    all_data['pp_evo'] = all_data['pp_evo'] / 100
    all_data['unemp'] = all_data['unemp'] / 100
    
    
    
    birthRateVect=all_data['birthRate'].values
    seqInAll=[]
    seqOut=[]
    seq_size=3
    for i in range(len(all_data)-seq_size):
        seqInAll.append(all_data.values[i:i+seq_size])
        seqOut.append(birthRateVect[i+seq_size])
        
        
    n_patterns = len(seqInAll)
    param_length=len(all_data.columns)
    seqInAll=np.array(seqInAll)
    network_input = np.reshape(seqInAll, (n_patterns, seq_size, param_length))
        
        
    # create and fit the LSTM network
    # fix random seed for reproducibility
        
    model = Sequential()
    model.add(LSTM(6, input_shape=(network_input.shape[1], network_input.shape[2])))
    model.add(Dropout(0.3))
    model.add(Dense(6))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
        
    model.fit(network_input, seqOut, epochs=10, batch_size=4, verbose=2)
        
    #predict
    seqPredict=all_data.values[-seq_size:]
    seqInPredict = np.reshape(seqPredict, (1, seq_size, param_length))
    Out=model.predict(seqInPredict)
    birthRateYear=(Out[0][0]+1)*BirthRateY
    
    #end proj birth
    AvgMF=(sum(PopM[-1,18:41])+sum(PopW[-1,18:41]))/2
    birthYear=AvgMF*birthRateYear
        
    #Take the last vector in the pyramid and project
    popYearM=projModule.projDeathAndAging('M',PopM[-1,:],mortalityRatesM[iProj-1],birthYear,BirthMFratio)
    popYearW=projModule.projDeathAndAging('F',PopW[-1,:],mortalityRatesW[iProj-1],birthYear,BirthMFratio)
    #Add proj to past
    popYearM = np.reshape(popYearM,(1,len(popYearM)))
    popYearW = np.reshape(popYearW,(1,len(popYearW)))
        
    PopM=np.append(PopM,popYearM,axis=0)
    PopW=np.append(PopW,popYearW,axis=0)
    #Add year in vect
    yearVect=np.append(yearVect,yearVect[-1:]+1)
"""
