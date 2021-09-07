#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import random as rd
from scipy import constants
from scipy.linalg import lstsq
from matplotlib.animation import FuncAnimation
plt.rcParams.update({'font.size': 6.5})

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
    maxlev = int(sum(levels)+1)
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
def exchange(ifrom, ito, nop, levels, istep):
    if (ito >= 0 and ito < nop) and (ifrom >= 0 and ifrom < nop) and levels[ifrom] > 0:
        levels[ifrom]-=1
        levels[ito]+=1
        istep+=1

    return(levels, istep)

#calculate distribution over energy levels
def recdist(nop, levels, maxlev, all_levels, all_distr):
    distr = np.zeros(maxlev)

    for i in range(0, nop):
        if levels[i] <= maxlev-1:
            distr[int(levels[i])] += 1
        else:
            print('error')
    
    new_levels = np.copy(levels)
    new_levels[new_levels==0] = 0.05

    cum_levels = []
    for element in new_levels:
        cum_levels.append(element)
    all_levels.append(cum_levels)

    new_distr = np.copy(distr)
    new_distr[new_distr==0] = 0.05

    cum_distr = []
    for element in new_distr:
        cum_distr.append(element)
    all_distr.append(cum_distr)

    return(distr, all_levels, all_distr)

#calculate accumulated distribution
def accum(nop, maxlev, idist, distr, dist_sum, nstep, all_dist_sum):
    for i in range(0, maxlev):
        idist[i] = distr[i]+idist[i]
        if nstep != 0:
            dist_sum[i] = idist[i]/nstep
        else:
            dist_sum[i] = idist[i]

    new_dist_sum = np.copy(dist_sum)
    new_dist_sum[new_dist_sum==0] = 0.05

    cum_dist_sum = []
    for element in new_dist_sum:
        cum_dist_sum.append(element)
    all_dist_sum.append(cum_dist_sum)

    return(idist, dist_sum, all_dist_sum, nstep)

#calculate average energy from distribution
def calc_Eav(maxlev, distr, nop):
    U = 0

    for j in range(0, maxlev):
        U+=j*distr[j]

    return(U/nop)

#calculate Boltzmann energy on the fly
def calc_Bolt_ent(nop, distr):
    num = np.math.factorial(nop)
    denom = 1

    for i in distr:
        denom*=np.math.factorial(i)

    W = num/denom
    S_w = k_B * np.log(W)

    return(W, S_w)

#calculate Boltzmann entropy for average distribution
def calc_av_ent(nop, dist_sum):
    num = nop*np.log(nop)-nop #Stirling approximation
    denom = 0

    for i in dist_sum:
        if i != 0:
            denom+=(i*np.log(i)-i) #Stirling approximation

    lnW=num-denom
    S_a = k_B * lnW

    return(S_a)

#find maxima for plotting animation
def find_max(all_levels, all_distr, all_dist_sum):

    max_levels = []
    for levels in all_levels:
        max_levels.append(max(levels))

    max_level = max(max_levels)

    max_distr = []
    for distr in all_distr:
        max_distr.append(max(distr))

    max_dist = max(max_distr)

    max_dist_sum = []
    for dist_sum in all_dist_sum:
        max_dist_sum.append(max(dist_sum))

    max_distr_sum = max(max_dist_sum)

    return(max_level, max_dist, max_distr_sum)

#find skips
def find_skips(maximum):
    skips = [1, 2, 5]
    if maximum >= 15 and maximum <= 25:
        skip = skips[1]
    elif maximum > 25:
        skip = skips[2]
    else:
        skip = skips[0]

    return(skip)

#calculate probability distribution and fit temperature
def calc_prob_temp(dist_sum, nop):
    #probability distribution
    prdist = np.zeros(len(dist_sum))
    lnprdist = np.zeros(len(dist_sum))

    for i in range(len(dist_sum)):
        prdist[i] = dist_sum[i]/nop
        if prdist[i] != 0:
           lnprdist[i] = np.log(prdist[i])
        else:
            lnprdist[i] = 0

    y = lnprdist[lnprdist != 0]

    #least square fit to get temperature

    x = np.arange(0, len(y))
    M = x[:, np.newaxis]**[0, 1]
    p, res, rnk, s = lstsq(M, y)
    yy = p[0] + p[1]*x
    if p[1] > 0:
        temp = 1/p[1]
    elif p[1] < 0:
        temp = -1/p[1]
    elif p[1] < 1:
        temp = np.nan
    realtemp = temp*((h*c*nu)/k_B)
    exp = (h*c*nu*1e23)/(k_B*1e23)

#    print('Average energy: %.4fe-21 J'%((h*c*nu)/(np.exp(exp*(1/realtemp))-1)*1e21))

    return(temp)

#plot prob. distribution and lstq fit
def plot_prob_temp(dist_sum, nop, max_level):
    #probability distribution
    prdist = np.zeros(len(dist_sum))
    lnprdist = np.zeros(len(dist_sum))

    for i in range(len(dist_sum)):
        prdist[i] = dist_sum[i]/nop
        if prdist[i] != 0:
           lnprdist[i] = np.log(prdist[i])
        else:
            lnprdist[i] = 0

    y = lnprdist[lnprdist != 0]

    #least square fit to get temperature

    x = np.arange(0, len(y))
    M = x[:, np.newaxis]**[0, 1]
    p, res, rnk, s = lstsq(M, y)
    yy = p[0] + p[1]*x
    if p[1] > 0:
        temp = 1/p[1]
    elif p[1] < 0:
        temp = -1/p[1]
    else:
        temp = np.nan
    realtemp = temp*((h*c*nu)/k_B)
    exp = (h*c*nu*1e23)/(k_B*1e23)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))

    x1 = np.arange(0, len(dist_sum), 1)
    xticks = np.arange(0, max_level+1, find_skips(max_level))
    x2 = np.arange(0, len(dist_sum), 0.1)
    ax1.bar(x1, prdist, color='b')
    ax1.set_xticks(xticks)
    ax1.set_xlabel('energy level')
    ax1.set_ylabel('$\\rho (n)$')
    ax1.set_xlim(-0.8, max_level+1-0.2)
    q = np.sum(np.exp(-(x1)/temp))
    bd = np.exp(-(x2)/temp)/q
    ax1.plot(x2, bd, color='k')

    ax2.scatter(x, y, color='b')
    ax2.set_xticks(x)
    ax2.set_xlabel('energy level')
    ax2.set_ylabel('ln($\\rho (n)$)')
    ax2.plot(x, yy, '--', color='gray')

#append to stored data
def store_data(all_uav, maxlev, nop, distr, all_wbolt, all_sw, all_sa, dist_sum, all_temp):
    all_uav.append(calc_Eav(maxlev, distr, nop))
    W, S_w = calc_Bolt_ent(nop, distr)
    all_wbolt.append(W)
    all_sw.append(S_w)
    all_sa.append(calc_av_ent(nop, dist_sum))
    all_temp.append(calc_prob_temp(dist_sum, nop))

    return(all_uav, all_wbolt, all_sw, all_sa, all_temp)

#append equilibration data
def store_eq_data(zero_array, all_dist_sum, all_uav, maxlev, distr, nop, all_wbolt, all_sw, all_sa, all_temp):
    all_dist_sum.append(zero_array)
    all_uav.append(calc_Eav(maxlev, distr, nop))
    W, S_w = calc_Bolt_ent(nop, distr)
    all_wbolt.append(W)
    all_sw.append(S_w)
    all_sa.append('undefined')
    all_temp.append('undefined')

    return(all_dist_sum, all_uav, all_wbolt, all_sw, all_sa, all_temp)

#auxiliary functions

def print_w_sw(all_wbolt, all_sw, start, end):
   for i, x, y in zip(np.arange(len(all_wbolt[start:end])), all_wbolt[start:end], all_sw[start:end]):
       print('step %d\t' %(i), 'W = %6s\t' %(x), 'S_w = %.4fe-21 J/K' %(y*1e21))

def get_w_sw(nop, en):
    levels = generate_levels(en, nop)
    W, S_w = calc_Bolt_ent(nop, levels)

    plt.rcParams['figure.figsize'] = [4, 3]

    max_level = max(levels)+3
    labels1 = np.arange(1, nop+1, 1)
    trimmed_dist1 = levels

    xrange_ = np.arange(0, nop+1, find_skips(nop+1))
    xticks = xrange_[1:]
    xticks = [1, *xticks]
    plt.xticks(xticks)
    plt.yticks(np.arange(0, max_level+1, find_skips(max_level)))
    plt.xlabel('molecule')
    plt.ylabel('energy')
    plt.ylim(-0.2, max_level+1-0.6)
    plt.bar(x=labels1, height=trimmed_dist1, color='b')

    plt.tight_layout()
    plt.show()

    print('Statistical weight: %.0f' %(W) + '    ' + 'Boltzmann entropy: %.4fe-21 J/K' %(S_w*1e21))
