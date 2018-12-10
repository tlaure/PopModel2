#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 19:02:47 2018

@author: Thomas
"""

def projPop2(PopM,PopW,yearVect,nProj): 
    #To project the population this function use the historical pyramid
    #Evaluate the death rate function at each age using regressions
    #Evaluate the number of newborn based on a regression of each years birth according to n# of woment between 16to50
    import pandas
    import numpy as np
    #Call the function for the mortality rates
    
    (mortalityRatesM,life_aveM)=projMortalityRates(PopM,nProj)
    (mortalityRatesW,life_aveW)=projMortalityRates(PopW,nProj)
    '''Updated with link'''
    indics = pandas.read_csv('indics.csv')

    for iProj in range(1,nProj+1):
        
        #Begin with the birth that happen during the first year
        
        (birthRateYear,BirthMFratio)=projBirth(PopM,PopW,yearVect,indics)
        AvgMF=(sum(PopM[-1,24:34])+sum(PopW[-1,24:34]))/2
        birthYear=AvgMF*birthRateYear
            
        #Take the last vector in the pyramid and project
        popYearM=projDeathAndAging('M',PopM[-1,:],mortalityRatesM[iProj-1],birthYear,BirthMFratio)
        popYearW=projDeathAndAging('F',PopW[-1,:],mortalityRatesW[iProj-1],birthYear,BirthMFratio)
        #Add proj to past
        popYearM = np.reshape(popYearM,(1,len(popYearM)))
        popYearW = np.reshape(popYearW,(1,len(popYearW)))
        
        PopM=np.append(PopM,popYearM,axis=0)
        PopW=np.append(PopW,popYearW,axis=0)
        #Add year in vect
        yearVect=np.append(yearVect,yearVect[-1:]+1)
    

    return(PopM,PopW)


def projDeathAndAging(S,pop0,deathRates,birthYear,BirthMFratio): 
    
    import numpy as np
    
    
    if S=='M':
        newBorn = np.array([birthYear*BirthMFratio])
    else:
        newBorn = np.array([birthYear*(1-BirthMFratio)])
    
    popBeforeNewBorn=np.zeros((pop0.size))
    for i in range(0,pop0.size-2):
        popBeforeNewBorn[i+1]=pop0[i]-pop0[i]*deathRates[i]
        
    popBeforeNewBorn[0]=newBorn
    return(popBeforeNewBorn)
    
    
def projMortalityRates(popHist,n):
    import numpy as np
    #import plots
    yearHist=popHist[:,1].size
    nAgePyramid=popHist[1,:].size
    ages=np.arange(0,nAgePyramid)
    mortalityRatesHist=np.zeros((yearHist-1,nAgePyramid))#Container for the mortality rates, ordered from most recent to older rates
    mortalityRatesHistClean=np.zeros((yearHist-1,nAgePyramid))#Container for the mortality rates, ordered from most recent to older rates
    life_ave=[]
    for iYear in range(1,yearHist):
        for iAge in range(1,nAgePyramid):
            #1-popAgeN@t/popAgeN-1@t is the mortality rate for N-1@t-1 as is the % of pop which haven't survived
            if popHist[iYear-1,iAge-1]==0:
                mortalityRatesHist[iYear-1,iAge-1]=0
            else:
                mortalityRatesHist[iYear-1,iAge-1]=1-popHist[iYear,iAge]/popHist[iYear-1,iAge-1]
        #Fit a polynomal regtession from year 0 to last survival year -2 as the last two value may lead to some issues
        #start by cleaning the end of the vector including the two last data
        yearEnd=nAgePyramid-1
        
        while (mortalityRatesHist[iYear-1,yearEnd]==0):
            yearEnd=yearEnd-1
        #Perform regression
        x=ages[0:yearEnd-2]
        y=mortalityRatesHist[iYear-1,0:yearEnd-2]
        z = np.polyfit(x, y, 5)
        p=np.poly1d(z)#Polynom for the mortality rates at year iYear
        mortalityRatesHistClean[iYear-1,:]=p(ages)
        intP = np.polyint(p, 1)-0.5 # integrate polymon to obtain the life expectancy
        life_exp_year=40
        
        while intP(life_exp_year)<0.5 and life_exp_year<120:
            life_exp_year+=0.5
        life_ave.append(life_exp_year)
        
        
        
    #Plot mortality rate across ages for fixed period and evolution of mortality rate by age during the past
    #plots.plotNvert(mortalityRatesHistClean,10)
    #plots.plotNhor(mortalityRatesHistClean,10)
    #Project mortality rates for upcomming periods 
    #following the graph use a linear regression across the periods
    mortalityRatesProj=np.zeros((n,nAgePyramid))
    vectorHist=np.arange(0,yearHist-1)
    vectorProj=np.arange(yearHist+1,yearHist+1+n)
    for iAge in range(0,nAgePyramid):
        y=mortalityRatesHistClean[:,iAge]
        x=vectorHist
        z = np.polyfit(x, y, 1)
        p=np.poly1d(z)
        mortalityRatesProj[:,iAge]=p(vectorProj)
        
    #plots.plotNhor(mortalityRatesProj,1)
    #plots.plotNhor(mortalityRatesHistClean,1)
    y=life_ave
    x=vectorHist
    z = np.polyfit(x, y, 1)
    p=np.poly1d(z)
    add_life=p(vectorProj)
    life_ave=np.array(life_ave)
    life_ave=np.append(life_ave,add_life)
    return(mortalityRatesProj,life_ave)
    
def projBirth(PopM,PopW,yearVect,indics):
    import pandas
    import statistics
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, LSTM
    import numpy as np
    
    np.random.seed(7)
    
    yearHist=PopM[:,1].size
    BirthRate=np.zeros((yearHist-1,2))#Container for the annual birth rate 
    MFratioTab=np.zeros((yearHist-1))#Container for the ratio between male and female
    for iYear in range(1,yearHist):
        #Birth rate as a % of population between 18 to 45
        BirthN = PopM[iYear,0]+PopW[iYear,0]
        AvgMF = (sum(PopW[iYear-1,25:36])+sum(PopM[iYear-1,25:36]))/2
        BirthRate[iYear-1,0] = BirthN/AvgMF
        MFratioTab[iYear-1] = PopM[iYear,0]/(PopW[iYear,0]+PopM[iYear,0])
        if iYear>1:
            BirthRate[iYear-1,1]=(AvgMF-PrevAvg)
        PrevAvg = AvgMF
        
    BirthMFratio=statistics.mean(MFratioTab)
    #plots.plotFunL(yearsHist[0:-1],BirthRate)
            
    d=np.zeros((yearHist-1,2))
    d[:,0]=yearVect[0:-1]
    #d[:,1]=BirthRate[:,1]
    d[:,1]=BirthRate[:,0]
        
            
        #birthToMerge=pandas.DataFrame(d,columns=['Year','pop_evo', 'birthRate'])
    birthToMerge=pandas.DataFrame(d,columns=['Year', 'birthRate'])
        
        
    all_data=indics.set_index('year').join(birthToMerge.set_index('Year'))
        
    #transform tab in order to have all inputs vary from 0 to 1
    all_data=all_data[~np.isnan(all_data["birthRate"])]#To remove columns with missing data
        
    all_data['pp_evo'] = (all_data['pp_evo']-min(all_data['pp_evo']))/(max(all_data['pp_evo'])-min(all_data['pp_evo']))
    all_data['unemp'] = (all_data['unemp']-min(all_data['unemp']))/(max(all_data['unemp'])-min(all_data['unemp']))
        #all_data['pop_evo'] = (all_data['unemp']-min(all_data['unemp']))/(max(all_data['unemp'])-min(all_data['unemp']))
        #For birth rate save transforming parameters
    minB=min(all_data['birthRate'])
    maxB=max(all_data['birthRate'])
    all_data['birthRate'] = (all_data['birthRate']-minB)/(maxB-minB)
        
    birthRateVect=all_data['birthRate'].values
    seqInAll=[]
    seqOut=[]
    seq_size=6
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
    model.add(LSTM(8, input_shape=(network_input.shape[1], network_input.shape[2])))
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
    birthRateYear=Out[0][0]*(maxB-minB)+minB
    return(birthRateYear,BirthMFratio)
    