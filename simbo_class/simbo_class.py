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
        self.nu = nu*1e2

        self.k_B = constants.value(u'Boltzmann constant')
        self.h = constants.value(u'Planck constant')
        self.c = constants.value(u'speed of light in vacuum')
        self.N_A = constants.value(u'Avogadro constant')

        self.nseqv = self.nop*20 

    #helpful functions
    def find_max(self, all_levels, all_distr, all_dist_sum):
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

    def find_skips(self, maximum):
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
            levels[i] = int(rd.uniform(0.0, 1.0)*self.en*2+0.5)
    
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
    def getrand(self, levels):
        """
        Get random molecules.
        """
        #get ifrom
        Etot = sum(levels)
        ien = int(np.ceil(rd.random()*Etot))
        ifrom = 0
        iq = levels[ifrom]

        while iq < ien and ifrom < self.nop-1:
            ifrom+=1
            iq+=levels[ifrom]

        #get nr of quanta
        itr = iq-ien+1

        #get ito
        ito = ifrom

        while (ito == ifrom):
            ito = rd.randint(0, self.nop-1)

        return(ifrom, ito, itr)

    def exchange(self, ifrom, ito, itr, levels, istep):
        """
        Exchange energy between levels.
        """
        if levels[ifrom] > 0:

            levels[ifrom]-=itr
            levels[ito]+=itr
            istep+=1

        else:

            istep+=1

        return(levels, istep)

    def recdist(self, levels, maxlev):
        """
        Calculate distribution over levels.
        """
        distr = np.zeros(maxlev)

        for i in range(0, self.nop):
            if levels[i] <= maxlev-1:
                distr[int(levels[i])] += 1
            else:
                print('error')
    
        return(distr)

    def accum(self, maxlev, idist, distr, dist_sum, nstep, all_dist_sum):
        """
        Calculate accumulated distribution.
        """
        for i in range(0, maxlev):
            idist[i] = distr[i]+idist[i]
            if nstep != 0:
                dist_sum[i] = idist[i]/(nstep+1)
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

        return(U/self.nop)

    def calc_Bolt_ent(self, distr):
        """
        Calculate Boltzmann entropy.
        """
        num = np.math.factorial(self.nop)
        denom = 1

        for i in distr:
            denom*=np.math.factorial(i)

        W = num/denom
        S_w = np.log(W)

        return(W, S_w)

    def calc_av_ent(self, dist_sum):
        """
        Calculate Boltzmann entropy for accumulated distribution (uses Stirling approximation for N!).
        """
        num = self.nop*np.log(self.nop)-self.nop 
        denom = 0

        for i in dist_sum:
            if i != 0:
                denom+=(i*np.log(i)-i) 

        lnW=num-denom
        S_a = lnW

        return(S_a)

    def calc_prob_temp(self, dist_sum):
        """
        Calculate probability distribution and do least square fit to get temperature.
        """
        prdist = np.zeros(len(dist_sum))
        lnprdist = np.zeros(len(dist_sum))

        for i in range(len(dist_sum)):
            prdist[i] = dist_sum[i]/self.nop
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

    def store_data(self, levels, all_levels, all_distr, all_uav, maxlev, distr, all_wbolt, all_sw, all_sa, dist_sum, all_temp):
        """
        Appends to stored data.
        """
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

        all_uav.append(self.calc_Eav(maxlev, distr))
        W, S_w = self.calc_Bolt_ent(distr)
        all_wbolt.append(W)
        all_sw.append(S_w)
        all_sa.append(self.calc_av_ent(dist_sum))
        all_temp.append(self.calc_prob_temp(dist_sum))

        return(all_levels, all_distr, all_uav, all_wbolt, all_sw, all_sa, all_temp)

    def run(self, levels, nstot):
        """
        Runs simulations after initialization.    
        """
        nstot = self.nseqv + nstot

        maxlev, idist, dist_sum, nstep, istep, all_levels, all_distr, all_dist_sum, all_uav, all_wbolt, all_sw, all_sa, all_temp, zero_array = self.setzero(levels)

        while istep < nstot:
    
            ifrom, ito, itr = self.getrand(levels)
    
            levels, istep = self.exchange(ifrom, ito, itr, levels, istep)
    
            distr = self.recdist(levels, maxlev)
    
            if istep > self.nseqv: #check if equilibrated
            
                idist, dist_sum, all_dist_sum, nstep = self.accum(maxlev, idist, distr, dist_sum, nstep, all_dist_sum)
                all_levels, all_distr, all_uav, all_wbolt, all_sw, all_sa, all_temp = self.store_data(levels, all_levels, all_distr, all_uav, maxlev, distr, all_wbolt, all_sw, all_sa, dist_sum, all_temp)
                nstep+=1

        max_level, max_dist, max_distr_sum = self.find_max(all_levels, all_distr, all_dist_sum)
            
        return(dist_sum, all_levels, all_distr, all_dist_sum, all_uav, all_wbolt, all_sw, all_sa, all_temp, max_level, max_dist, max_distr_sum)

    def show(self, all_levels, all_distr, all_dist_sum, all_uav, all_wbolt, all_sw, all_sa, all_temp, units):
        """
        Shows animation of stored data from simulation run.
        """
        max_level, max_dist, max_distr_sum = self.find_max(all_levels, all_distr, all_dist_sum)

        fig = plt.figure(constrained_layout=True, figsize=(8, 6))
        gs = fig.add_gridspec(2, 4)
        gs.update(wspace=0.5)
        ax1 = fig.add_subplot(gs[0, :2], )
        ax2 = fig.add_subplot(gs[0, 2:])
        ax3 = fig.add_subplot(gs[1, 1:3])

        def animation_frame(i):
            plt.rcParams.update({'font.size': 10})
            ax1.cla()

            labels1 = np.arange(1, self.nop+1, 1)
            trimmed_dist1 = all_levels[i]

            xrange_ = np.arange(0, self.nop+1, self.find_skips(self.nop+1))
            xticks = xrange_[1:]
            xticks = [1, *xticks]
            ax1.set_xticks(xticks)
            ax1.set_yticks(np.arange(0, max_level+1, self.find_skips(max_level)))
            ax1.set_xlabel('molecule')
            ax1.set_ylabel('energy')
            ax1.set_ylim(-0.2, max_level+1-0.6)
            ax1.bar(x=labels1, height=trimmed_dist1, color='b')
            ax1.set_title(f'Energy levels \n step = {i+self.nseqv}')

            ax2.cla()

            labels2 = np.arange(0, len(all_distr[i]), 1)
            trimmed_dist2 = all_distr[i]
            ax2.set_xticks(np.arange(0, max_level+1, self.find_skips(max_level)))
            ax2.set_yticks(np.arange(0, max_dist+1, self.find_skips(max_dist)))
            ax2.set_ylabel('nr. of molecules')
            ax2.set_xlabel('energy level')
            ax2.set_ylim(-0.2, max_dist+1-0.2)
            ax2.set_xlim(-0.8, max_level+1-0.2)
            ax2.bar(x=labels2, height=trimmed_dist2, color='b')
            ax2.set_title(f'Distribution of levels \n step = {i+self.nseqv}')

            ax3.cla()

            labels3 = np.arange(0, len(all_dist_sum[i]), 1)
            trimmed_dist3 = all_dist_sum[i]
            ax3.set_xticks(np.arange(0, max_level+1, self.find_skips(max_level)))
            ax2.set_yticks(np.arange(0, max_distr_sum, self.find_skips(max_distr_sum)))
            ax3.set_ylabel('average nr. of molecules')
            ax3.set_xlabel('energy level')
            ax3.set_ylim(-0.2, max_distr_sum+1-0.2)
            ax3.set_xlim(-0.8, max_level+1-0.2)
            ax3.bar(x=labels3, height=trimmed_dist3, color='b')
    
            xv = (max_level+1)*1.1
            yv = max_distr_sum+1
        
            ax3.text(xv, yv*(7/9), 'Current distribution', fontweight='bold')
            ax3.text(xv, yv*(5/9), 'Statistical weight: \n %.2E' %(all_wbolt[i]))
            ax3.text(xv, yv*(2/9), 'Accum. distribution', fontweight='bold')
            
            ax3.set_title(f'Accum. distribution \n accum. steps = {i}')

            if units == 'default':
                ax3.text(xv, yv*(1/9), 'Average entropy: %.2f kJ/mol K' %(all_sa[i]*self.k_B*self.N_A))
                ax3.text(xv, 0, 'Temperature: %.1f K' %(all_temp[i]*((self.h*self.c*self.nu)/self.k_B)))
                ax3.text(xv, yv*(8/9), 'Average energy: %.2f kJ/mol'%((all_uav[i])*self.h*self.c*self.nu*self.N_A))
                ax3.text(xv, yv*(3/9), 'Boltzmann entropy: \n %.2f kJ/mol K' %(all_sw[i]*self.k_B*self.N_A))
            else:
                ax3.text(xv, yv*(1/9), 'Average entropy: %.4f red. un.' %(all_sa[i]))
                ax3.text(xv, 0, 'Temperature: %.4f red. un.' %(all_temp[i]))
                ax3.text(xv, yv*(8/9), 'Average energy: %.4f red. un.'%(all_uav[i]))
                ax3.text(xv, yv*(3/9), 'Boltzmann entropy: \n %.4f red. un.' %(all_sw[i]))

        frames_no = len(all_levels)

        return(fig, animation_frame, frames_no)

