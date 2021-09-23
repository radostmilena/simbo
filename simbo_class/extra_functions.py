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
  
    def __init__(self, en, nop, nu):
        super(extra_functions, self).__init__(en, nop, nu)     

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

    def generate_levels(self):
        run_simbo.generate_levels(self)

    def calc_Bolt_ent(self, levels):
        run_simbo.calc_Bolt_ent(self, levels)

    def plot_prob_temp(self, dist_sum, max_level):
        """
        Calculates probability distribution/temperature and plots Boltzmann distribution and fit.
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
        yy = p[0] + p[1]*x
        if p[1] > 0:
            temp = 1/p[1]
        elif p[1] < 0:
            temp = -1/p[1]
        else:
            temp = np.nan

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,3))

        x1 = np.arange(0, len(dist_sum), 1)
        xticks = np.arange(0, max_level, self.find_skips(max_level))
        x2 = np.arange(0, len(dist_sum)-0.9, 0.1)
        ax1.bar(x1, prdist, color='b')
        ax1.set_xticks(xticks)
        ax1.set_xlabel('energy level')
        ax1.set_ylabel('$\\rho (n)$')
        ax1.set_xlim(-0.8, max_level-0.2)
        ax1.set_ylim(-min(prdist)*0.1, max(prdist)+max(prdist)*0.5)
        q = np.sum(np.exp(-(x1)/temp))
        bd = np.exp(-(x2)/temp)/q
        ax1.plot(x2, bd, color='k', label='Boltzmann distribution')
        ax1.legend()

        ax2.scatter(x, y, color='b')
        xticks1 = np.arange(0, len(y), self.find_skips(len(y)))
        ax2.set_xticks(xticks1)
        ax2.set_ylim(min(lnprdist)+min(lnprdist)*0.2, max(lnprdist)-max(lnprdist)*0.2)
        ax2.text(0, min(lnprdist)-min(lnprdist)*0.1, 'T = %.2f red. un. \n T = %.1f K' %(temp, temp*((self.h*self.c*self.nu)/self.k_B))) 
        ax2.set_xlabel('energy level')
        ax2.set_ylabel('ln($\\rho (n)$)')
        ax2.plot(x, yy, '--', color='gray')

        plt.tight_layout()

    def print_w_sw(self, all_wbolt, all_sw, start, end, units):
        """
        Prints W and S_w for selected interval and units.  
        """
        for i, x, y in zip(np.arange(len(all_wbolt[start:end])), all_wbolt[start:end], all_sw[start:end]):
            if units == 'default':
               print('step %d\t' %(i), 'W = %.2E\t' %(x), 'S_w = %.2f kJ/mol K' %(y*self.k_B*self.N_A))
            else:
               print('step %d\t' %(i), 'W = %.2E\t' %(x), 'S_w = %.4f red. un.' %(y))

    def get_w_sw(self, en, nop, units):# en and nop not belonging to class!!!
        """
        Allows user to play around with W and S_w.      .  
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

        plt.rcParams['figure.figsize'] = [4, 3]

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
