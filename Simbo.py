#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import random as rd
from scipy import constants
from scipy.linalg import lstsq

k_B = constants.value(u'Boltzmann constant')
h = constants.value(u'Planck constant')
c = constants.value(u'speed of light in vacuum')
nu = 500e2

#automatic generation of levels
def generate_levels(en, nop):

    levels = np.zeros(nop)

    for i in range(0, nop, 1):
        levels[i] = int(rd.random()*en*2+0.5)
    
    return(levels)

#set everything zero
def setzero(levels):
    maxlev = int(max(levels)+1)
    idist = np.zeros(maxlev)
    dist_sum = np.zeros(maxlev)
    nstep = 0

    return(maxlev, idist, dist_sum, nstep)

#get random molecules
def getrand(nop):
    ito = int(np.ceil(rd.random()*nop-1))
    ifrom = int(np.ceil(rd.random()*nop-1))

    while (ito == ifrom):
        ito = int(np.ceil(rd.random()*nop-1))

    return(ifrom, ito)

#exchange energies between levels
def exchange(ifrom, ito, nop, levels, idist, dist_sum, istep):
    if (ito >= 0 and ito < nop) and (ifrom >= 0 and ifrom < nop) and levels[ifrom] > 0:
        levels[ifrom]-=1
        levels[ito]+=1
        istep+=1

    #new limit for levels
    maxlev = int(max(levels)+1)
    add = maxlev - len(idist)
    if add > 0:
        addarray = np.zeros(add)
        idist = np.concatenate((idist, addarray))
        dist_sum = np.concatenate((dist_sum, addarray))

    return(levels, maxlev, idist, dist_sum, istep)

#calculate distribution over energy levels
def recdist(nop, levels, maxlev, all_levels):
    distr = np.zeros(maxlev)

    for i in range(0, nop):
        if levels[i] <= maxlev:
            distr[int(levels[i])] += 1
        else:
            print('error')
    
    new_levels = np.copy(levels)
    new_levels[new_levels==0] = 0.05

    cum_levels = []
    for element in new_levels:
        cum_levels.append(element)
    all_levels.append(cum_levels)

    return(distr, all_levels)

#calculate accumulated distribution
def accum(maxlev, idist, distr, dist_sum, nstep):
    for i in range(0, maxlev):
            idist[i] = distr[i]+idist[i]
            if nstep != 0:
                dist_sum[i] = idist[i]/nstep

    return(idist, dist_sum)

#plot energy levels and distribution
def plot_AB(nop, levels, distr, maxlev, x):
    new_levels = np.copy(levels)
    new_levels[new_levels==0] = 0.05

    ax1.bar(x, new_levels)
    xticks = np.arange(0, nop+1, 2)
    ax1.set_xticks(xticks)
    ax1.set_xlabel('molecule')
    ax1.set_ylabel('energy')
    ax1.set_ylim(-0.2, max(levels)+1)

    new_distr = np.copy(distr)
    new_distr[new_distr==0] = 0.05

    x = np.arange(0, len(new_distr), 1)
    ax2.bar(x, new_distr)
    ax2.set_xticks(x)
    ax2.set_ylabel('nr. of molecules')
    ax2.set_xlabel('energy level')
    ax2.set_xlim(-1, max(x)+1)
    ax2.set_ylim(-0.2, max(new_distr)+1)
