#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import random as rd
from scipy import constants
from scipy.linalg import lstsq
from matplotlib.animation import FuncAnimation
from simbo_class import run_simbo
plt.rcParams.update({'font.size': 6.5})

#auxiliary functions
class extra_functions(run_simbo): #inherits objects and some methods from run_simbo
    """
    :class: Extra functions for plotting etc.

    Methods
    -------
    __init__
    find_skips
    generate_levels
    calc_Bolt_ent
    plot_prob_temp
    print_w_sw
    get_w_sw
    plot_eq

    """
    def __init__(self, en, nop, nu):
        """
        Constructor (inherits from run_simbo class)
        """
        super(extra_functions, self).__init__(en, nop, nu)     

    def find_skips(self, maximum):
        """
        Finds spacing for optimal axis labels. (Could have been inherited from run_simbo, but it didn't work for some reason).
        """
        skips = [1, 2, 5]
        if maximum >= 15 and maximum <= 25:
            skip = skips[1]
        elif maximum > 25:
            skip = skips[2]
        else:
            skip = skips[0]

        return(skip)

    def generate_levels(self):
        """
        Automatic generation of levels (inherited from run_simbo).
        """
        run_simbo.generate_levels(self)

    def calc_Bolt_ent(self, levels):
        """
        Calculate Boltzmann entropy (inherited from run_simbo).
        """
        run_simbo.calc_Bolt_ent(self, levels)

    def plot_prob_temp(self, dist_sum, max_level):
        """
        Calculates probability distribution/temperature and plots Boltzmann distribution and fit.
        (Kind of the same as calc_prob_temp in run_simbo, but with plots.)
        """
        prdist = np.zeros(len(dist_sum))
        lnprdist = np.zeros(len(dist_sum))
        for i in range(len(dist_sum)):
            prdist[i] = dist_sum[i]/self.nop
            if prdist[i] != 0:
               lnprdist[i] = np.log(prdist[i])
            else:
                lnprdist[i] = 0

        last_levels = prdist[prdist < 0.005]
        max_level = int((len(prdist)-1)-(len(last_levels)-1))

        y = lnprdist[lnprdist != 0]

        x = np.arange(0, len(y))
        M = x[:, np.newaxis]**[0, 1]
        p, res, rnk, s = lstsq(M, y)
        yy = p[0]+p[1]*x
        if p[1] > 0:
            temp = 1/p[1]
        elif p[1] < 0:
            temp = -1/p[1]
        else:
            temp = np.nan

        stderr = np.sqrt(res)**(1/np.sqrt(len(lnprdist)-1))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
        plt.rcParams.update({'font.size': 12})

        x1 = np.arange(0, len(dist_sum), 1)
        xticks = np.arange(0, max_level, self.find_skips(max_level))
        x2 = np.arange(0, len(dist_sum)-0.9, 0.1)
        ax2.bar(x1, prdist, color='b')
        ax2.set_xticks(xticks)
        ax2.set_xlabel('energy level')
        ax2.set_ylabel('$\\rho (n)$')
        ax2.set_xlim(-0.8, max_level-0.2)
        ax2.set_ylim(-min(prdist)*0.1, max(prdist)+max(prdist)*0.5)
        if np.isnan(temp) == False:
            q = np.sum(np.exp(-(x1)/temp))
            bd = np.exp(-(x2)/temp)/q

        else:
            bd = x2*0 + 1/max_level 

        ax2.plot(x2, bd, color='k', label='Fitted distribution')
        ax2.legend()

        ax1.scatter(x, y, color='b')
        xticks1 = np.arange(0, len(y), self.find_skips(len(y)))
        ax1.set_xticks(xticks1)
        ax1.set_ylim(min(lnprdist)+min(lnprdist)*0.2, max(lnprdist)-max(lnprdist)*0.2)
        ax1.text(0, min(lnprdist)-min(lnprdist)*0.1, 'T = %.2f ± %.2f red. un. \n T = %.1f ± %.1f K' %(temp, stderr, temp*((self.h*self.c*self.nu)/self.k_B), stderr*((self.h*self.c*self.nu)/self.k_B))) 
        ax1.set_xlabel('energy level')
        ax1.set_ylabel('ln($\\rho (n)$)')
        ax1.plot(x, yy, '--', color='gray')

        plt.tight_layout()

    def print_w_sw(self, all_wbolt, all_sw, start, end, units):
        """
        Prints W and S_w for selected interval and units.  (Not used in tasks)
        """
        for i, x, y in zip(np.arange(len(all_wbolt[start:end])), all_wbolt[start:end], all_sw[start:end]):
            if units == 'default':
               print('step %d\t' %(i), 'W = %.2E\t' %(x), 'S_w = %.2f kJ/mol K' %(y*self.k_B*self.N_A))
            else:
               print('step %d\t' %(i), 'W = %.2E\t' %(x), 'S_w = %.4f red. un.' %(y))

    def get_w_sw(self, en, nop, units):
        """
        Allows user to play around with W and S_w for Task 6. en and nop not belonging to class!!!  
        """
        levels = np.zeros(nop)

        for i in range(0, nop, 1):
            levels[i] = int(rd.random()*en*2+0.5)

        maxlev = int(max(levels))+1
        distr = np.zeros(maxlev)

        for i in range(0, nop):
            if levels[i] <= maxlev-1:
                distr[int(levels[i])] += 1
            else:
                print('error')

        num = np.math.factorial(nop)
        denom = 1

        for i in distr:
            denom*=np.math.factorial(i)

        W = num/denom
        S_w = np.log(W)

        plt.rcParams['figure.figsize'] = [6, 4.5]

        max_level = max(levels)+3
        labels1 = np.arange(1, nop+1, 1)
        trimmed_dist1 = levels

        xrange_ = np.arange(0, nop+1, self.find_skips(nop+1))
        xticks = xrange_[1:]
        xticks = [1, *xticks]
        plt.xticks(xticks)
        plt.yticks(np.arange(0, max_level+1, self.find_skips(max_level)))
        plt.xlabel('molecule')
        plt.ylabel('energy')
        plt.ylim(-0.2, max_level+1-0.6)
        plt.bar(x=labels1, height=trimmed_dist1, color='b')

        plt.tight_layout()
        plt.show()

        if units == 'default':
            print('Statistical weight: %.2E' %(W) + '    ' + 'Boltzmann entropy: %.2f J/mol K' %(S_w*self.k_B*self.N_A))
        else:
            print('Statistical weight: %.2E' %(W) + '    ' + 'Boltzmann entropy: %.4f red. un.' %(S_w))

    def plot_eq(self, all_wbolt, all_sw):
        """
        Plots equilibration data generated by run_eq.
        :param all_wbolt: stored data for statistical weight from equilibration run.
        :param all_sw: stored data for Boltzmann entropy from equilibration run.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

        x = np.arange(len(all_wbolt))
        all_wbolt = np.array(all_wbolt)
        ax1.plot(x, all_wbolt*1e-4, color='darkblue')
        ax1.set_xlabel('step')
        ax1.set_ylabel(r'$W \; \; / \; \; 10^4$')
        all_sw = np.array(all_sw)
        ax2.plot(x, all_sw*self.k_B*self.N_A)
        ax2.set_xlabel('step')
        ax2.set_ylabel(r'$S \; \; / \; \; \mathrm{kJ \cdot {mol}^{-1} \cdot K^{-1}}$')
        plt.tight_layout()
        plt.show()
