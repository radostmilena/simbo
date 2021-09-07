#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import random as rd
from scipy import constants
from scipy.linalg import lstsq
from matplotlib.animation import FuncAnimation
plt.rcParams.update({'font.size': 6.5})

class run_simbo:
    
    #constructor
    def __init__(self, en, nop, nu):

        self.en = en
        self.nop = nop
        self.nu = nu

	self.k_B = constants.value(u'Boltzmann constant')
	self.h = constants.value(u'Planck constant')
	self.c = constants.value(u'speed of light in vacuum')

        self.nstot = self.nop*40
        self.nseqv = self.nop*20 

    #helpful functions
    def find_max(all_levels, all_distr, all_dist_sum):
        """
        Finds maxima of stored data.   
        """
        max_levels = []
        max_distr = []
        max_dist_sum = []

        for levels, distr, dist_sum in zip(all_levels, all_distr, all_dist_sum):
            max_levels.append(max(levels))
            max_distr.append(max(distr))
            max_dist_sum.append(max(dist_sum))

        max_level = max(max_levels)
        max_dist = max(max_distr)
        max_distr_sum = max(max_dist_sum)

	return(max_level, max_dist, max_distr_sum)

    def find_skips(maximum):
        """
        Finds spacing for optimal axis labels.
        """
        skips = [1, 2, 5]
        if maximum >= 15 and maximum <= 25:
            skip = skips[1]
        elif maximum > 25:
            skip = skips[2]
        else:
	    skip = skips[0]

        return(skip)

    #functions for initialization
    def generate_levels(self):
        """
        Automatic generation of levels.
        """
        levels = np.zeros(self.nop)

        for i in range(0, self.nop, 1):
            levels[i] = int(rd.random()*self.en*2+0.5)
    
        return(levels)

    def setzero(self, levels):
        """
        Set everything zero. 
        """
        maxlev = int(sum(levels)+1)
        idist = np.zeros(maxlev)
        dist_sum = np.zeros(maxlev)
        all_levels = []
        all_distr = []
        all_dist_sum = []
        all_uav = []
        all_wbolt = []
        all_sw = []
        all_sa = []
        all_temp = [] 
        zero_array = []
        istep = 0
        nstep = 0

        return(maxlev, idist, dist_sum, nstep, istep, all_levels, all_distr, all_dist_sum, all_uav, all_wbolt, all_sw, all_sa, all_temp, zero_array)

    #functions for running simulation
    def getrand(self):
        """
        Get random molecules.
        """
        ito = int(np.ceil(rd.random()*nop-1))
        ifrom = int(np.ceil(rd.random()*nop-1))

        while (ito == ifrom):
            ito = int(np.ceil(rd.random()*nop-1))

        return(ifrom, ito)

    def exchange(self, ifrom, ito, levels, istep):
        """
        Exchange energy between levels.
        """
        if (ito >= 0 and ito < nop) and (ifrom >= 0 and ifrom < nop) and levels[ifrom] > 0:
            levels[ifrom]-=1
            levels[ito]+=1
            istep+=1

        return(levels, istep)

    def recdist(self, levels, maxlev, all_levels, all_distr):
        """
        Calculate distribution over levels.
        """
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

    def accum(self, maxlev, idist, distr, dist_sum, nstep, all_dist_sum):
        """
        Calculate accumulated distribution.
        """
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

    def calc_Eav(self, maxlev, distr):
        """
        Calculate average energy from distribution.
        """
        U = 0

        for j in range(0, maxlev):
            U+=j*distr[j]

        return(U/nop)

    def calc_Bolt_ent(self, distr):
        """
        Calculate Boltzmann entropy.
        """
        num = np.math.factorial(nop)
        denom = 1

        for i in distr:
            denom*=np.math.factorial(i)

        W = num/denom
        S_w = k_B * np.log(W)

        return(W, S_w)

    def calc_av_ent(self, dist_sum):
        """
        Calculate Boltzmann entropy for accumulated distribution (uses Stirling approximation for N!).
        """
        num = nop*np.log(nop)-nop 
        denom = 0

        for i in dist_sum:
            if i != 0:
                denom+=(i*np.log(i)-i) 

        lnW=num-denom
        S_a = k_B * lnW

        return(S_a)

    def calc_prob_temp(self, dist_sum):
        """
        Calculate probability distribution and do least square fit to get temperature.
        """
        prdist = np.zeros(len(dist_sum))
        lnprdist = np.zeros(len(dist_sum))

        for i in range(len(dist_sum)):
            prdist[i] = dist_sum[i]/nop
            if prdist[i] != 0:
               lnprdist[i] = np.log(prdist[i])
            else:
	       lnprdist[i] = 0

        y = lnprdist[lnprdist != 0]

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

        return(temp)

    def store_data(self, all_uav, maxlev, distr, all_wbolt, all_sw, all_sa, dist_sum, all_temp):
        """
        Appends to stored data.
        """
        all_uav.append(calc_Eav(maxlev, distr, nop))
        W, S_w = calc_Bolt_ent(nop, distr)
        all_wbolt.append(W)
        all_sw.append(S_w)
        all_sa.append(calc_av_ent(nop, dist_sum))
        all_temp.append(calc_prob_temp(dist_sum, nop))

        return(all_uav, all_wbolt, all_sw, all_sa, all_temp)

    def store_eq_data(self, all_dist_sum, all_uav, maxlev, distr, all_wbolt, all_sw, all_sa, all_temp, levels):
        """
        Appends to stored data from equilibration.
        """
        for i in range(int(sum(levels))+1):
            zero_array.append(0.05)

        all_dist_sum.append(zero_array)
        all_uav.append(calc_Eav(maxlev, distr, nop))
        W, S_w = calc_Bolt_ent(nop, distr)
        all_wbolt.append(W)
        all_sw.append(S_w)
        all_sa.append('undefined')
        all_temp.append('undefined')

        return(all_dist_sum, all_uav, all_wbolt, all_sw, all_sa, all_temp)

    def run(self, istep, levels, maxlev, idist, dist_sum, nstep):
        """
        Runs simulations after initialization.    
        """
        while istep < nstot:
    
            bef = istep
    
            ifrom, ito = getrand(nop)
    
            levels, istep = exchange(ifrom, ito, nop, levels, istep)
    
            if istep > bef: #exchange not possible sometimes (level[ifrom] = 0)
        
                distr, all_levels, all_distr = recdist(nop, levels, maxlev, all_levels, all_distr)
    
                if istep > nseqv: #check if equilibrated
            
                    idist, dist_sum, all_dist_sum, nstep = accum(nop, maxlev, idist, distr, dist_sum, nstep, all_dist_sum)
                    all_uav, all_wbolt, all_sw, all_sa, all_temp = store_data(all_uav, maxlev, nop, distr, all_wbolt, all_sw, all_sa, dist_sum, all_temp)
                    nstep+=1
            
                else:
            
                    all_dist_sum, all_uav, all_wbolt, all_sw, all_sa, all_temp = store_eq_data(zero_array, all_dist_sum, all_uav, maxlev, distr, nop, all_wbolt, all_sw, all_sa, all_temp)

        return(dist_sum, all_levels, all_distr, all_uav, all_wbolt, all_sw, all_sa, all_temp)

    def show(self, all_levels, all_distr, all_dist_sum, units, time_per_step):
        """
        Shows animation of stored data from simulation run.
        """
        max_level, max_dist, max_distr_sum = find_max(all_levels, all_distr, all_dist_sum)

        def animation_frame(i):
            plt.rcParams.update({'font.size': 10})
            ax1.cla()

	    labels1 = np.arange(1, nop+1, 1)
	    trimmed_dist1 = all_levels[i]

	    xrange_ = np.arange(0, nop+1, find_skips(nop+1))
	    xticks = xrange_[1:]
	    xticks = [1, *xticks]
	    ax1.set_xticks(xticks)
	    ax1.set_yticks(np.arange(0, max_level+1, find_skips(max_level)))
	    ax1.set_xlabel('molecule')
	    ax1.set_ylabel('energy')
	    ax1.set_ylim(-0.2, max_level+1-0.6)
	    ax1.bar(x=labels1, height=trimmed_dist1, color='b')
	    ax1.set_title(f'Energy levels \n step = {i}')

	    ax2.cla()

	    labels2 = np.arange(0, maxlev, 1)
	    trimmed_dist2 = all_distr[i]
	    ax2.set_xticks(np.arange(0, max_level+1, find_skips(max_level)))
	    ax2.set_yticks(np.arange(0, max_dist+1, find_skips(max_dist)))
	    ax2.set_ylabel('nr. of molecules')
	    ax2.set_xlabel('energy level')
	    ax2.set_ylim(-0.2, max_dist+1-0.2)
	    ax2.set_xlim(-0.8, max_level+1-0.2)
	    ax2.bar(x=labels2, height=trimmed_dist2, color='b')
	    ax2.set_title(f'Distribution of levels \n step = {i}')

	    ax3.cla()

	    labels3 = np.arange(0, maxlev, 1)
	    trimmed_dist3 = all_dist_sum[i]
	    ax3.set_xticks(np.arange(0, max_level+1, find_skips(max_level)))
	    ax2.set_yticks(np.arange(0, max_distr_sum, find_skips(max_distr_sum)))
	    ax3.set_ylabel('average nr. of molecules')
	    ax3.set_xlabel('energy level')
	    ax3.set_ylim(-0.2, max_distr_sum+1-0.2)
	    ax3.set_xlim(-0.8, max_level+1-0.2)
	    ax3.bar(x=labels3, height=trimmed_dist3, color='b')
    
	    xv = (max_level+1)*1.1
	    yv = max_distr_sum+1
        
	    ax3.text(xv, yv*(7/9), 'Current distribution', fontweight='bold')
	    ax3.text(xv, yv*(5/9), 'Statistical weight: \n %.1f' %(all_wbolt[i]))
	    ax3.text(xv, yv*(2/9), 'Accum. distribution', fontweight='bold')
            
            if units == 'default':
	        ax3.text(xv, yv*(8/9), 'Average energy: %.4fe-21 J'%((all_uav[i])*h*c*nu*1e21))
	        ax3.text(xv, yv*(3/9), 'Boltzmann entropy: \n %.4fe-21 J/K' %(all_sw[i]*1e21))
            else:
	        ax3.text(xv, yv*(8/9), 'Average energy: %.4f red. un.'%(all_uav[i]))
	        ax3.text(xv, yv*(3/9), 'Boltzmann entropy: \n %.4f red. un.' %(all_sw[i]))

	    if i > nseqv:
	       ax3.set_title(f'Accum. distribution \n accum. steps = {i-nseqv}')

               if units == 'default':
		     ax3.text(xv, yv*(1/9), 'Average entropy: %.4fe-21 J/K' %(all_sa[i]*1e21))
		     ax3.text(xv, 0, 'Temperature: %.0f K' %(all_temp[i]*((h*c*nu)/k_B)))
               else:
		     ax3.text(xv, yv*(1/9), 'Average entropy: %.4f red. un.' %(all_sa[i]))
		     ax3.text(xv, 0, 'Temperature: %.4f red. un.' %(all_temp[i]))
	    else:
	       ax3.set_title(f'Accum. distribution \n accum. steps = 0')
	       ax3.text(xv, yv*(1/9), 'Average entropy: %s' %(all_sa[i]))
	       ax3.text(xv, 0, 'Temperature: %s' %(all_temp[i]))

        frames_no = len(all_levels)
        animation = FuncAnimation(fig, animation_frame, frames=frames_no, interval=time_per_step, repeat=False)

