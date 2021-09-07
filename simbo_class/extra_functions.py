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
class extra_functions(run_simbo): #inherits objects and methods from run_simbo
  
    def __init__(self):
	super(extra_functions, self).__init__()     


    def plot_prob_temp(self, dist_sum, max_level):
        """
        Calculates probability distribution/temperature and plots Boltzmann distribution and fit.
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
        else:
            temp = np.nan

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

    def print_w_sw(self, all_wbolt, all_sw, start, end, units):
        """
        Prints W and S_w for selected interval and units.  
        """
	for i, x, y in zip(np.arange(len(all_wbolt[start:end])), all_wbolt[start:end], all_sw[start:end]):
            if units == 'default':
	       print('step %d\t' %(i), 'W = %6s\t' %(x), 'S_w = %.4fe-21 J/K' %(y*1e21))
            else:
	       print('step %d\t' %(i), 'W = %6s\t' %(x), 'S_w = %.4f red. un.' %(y))

    def get_w_sw(en, nop, units):# en and nop not belonging to class!!!
        """
        Allows user to play around with W and S_w.      .  
        """
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

        if units == 'default':
	    print('Statistical weight: %.0f' %(W) + '    ' + 'Boltzmann entropy: %.4fe-21 J/K' %(S_w*1e21))
        else:
	    print('Statistical weight: %.0f' %(W) + '    ' + 'Boltzmann entropy: %.4f red. un.' %(S_w))

