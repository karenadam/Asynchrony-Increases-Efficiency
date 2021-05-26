import numpy as np
import os
import sys
import time
import pickle
import scipy
import scipy.stats

Figure_Path = os.path.split(os.path.realpath(__file__))[0] + "/Figures/"
Data_Path = os.path.split(os.path.realpath(__file__))[0] + "/Data/"

graphical_import = True

if graphical_import:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rc

To_Svg = True

if To_Svg:
    figure_filename = Figure_Path + "mixing_comparison.svg"
    plt.rc('text', usetex=False)
    plt.rc('text.latex', unicode = False)
    plt.rc('svg',fonttype = 'none')

else:
    figure_filename = Figure_Path + "mixing_comparison.png"



with open(Data_Path + "Figure3_no_mixing.pkl", "rb") as f:  # Python 3: open(..., 'wb')
    obj = pickle.load(f, encoding="latin1")

n_spikes_total = obj[0]
n_spikes_constrained = obj[1]
results = obj[2]
# num_signals = results.shape[1]
total_deg_freedom = obj[3]
apparent_deg_freedom = obj[4]


clr = plt.rcParams["axes.prop_cycle"].by_key()["color"]

no_mixing_median = np.median(results[:, 0, :], 0)
no_mixing_q1 = np.quantile(results[:,0,:], 0.25, 0)
no_mixing_q3 = np.quantile(results[:,0,:], 0.75, 0)



with open(Data_Path + "Figure3_known_mixing.pkl", "rb") as f:  # Python 3: open(..., 'wb')
    obj = pickle.load(f, encoding="latin1")

n_spikes_total = obj[0]

n_spikes_total_known = obj[0]
n_spikes_constrained = obj[1]
results = obj[2]


known_mixing_median = np.median(results[:, 0, :], 0)
known_mixing_q1 = np.quantile(results[:,0,:], 0.25, 0)
known_mixing_q3 = np.quantile(results[:,0,:], 0.75, 0)

with open(Data_Path + "Figure3_test.pkl", "rb") as f:  # Python 3: open(..., 'wb')
    obj = pickle.load(f, encoding="latin1")

n_spikes_total = obj[0]
n_spikes_constrained = obj[1]
results = obj[2]




unknown_mixing_median = np.median(results[:, 0, :], 0)
unknown_mixing_q1 = np.quantile(results[:,0,:], 0.25, 0)
unknown_mixing_q3 = np.quantile(results[:,0,:], 0.75, 0)
plt.plot(n_spikes_total,no_mixing_median, label = 'Full Rank Assumption', color = clr[0])
plt.plot(n_spikes_total,no_mixing_q1, color = clr[0], alpha = 0.3, linestyle = '--')
plt.plot(n_spikes_total,no_mixing_q3, color = clr[0], alpha = 0.3, linestyle = '--')
plt.gca().fill_between(n_spikes_total, no_mixing_q1, no_mixing_q3, color = clr[0], alpha =0.2)


plt.plot(n_spikes_total,known_mixing_median, label = 'Known Low Rank Factorization', color = clr[1])
plt.plot(n_spikes_total,known_mixing_q1, color = clr[1], alpha = 0.3, linestyle = '--')
plt.plot(n_spikes_total,known_mixing_q3, color = clr[1], alpha = 0.3, linestyle = '--')
plt.gca().fill_between(n_spikes_total, known_mixing_q1, known_mixing_q3, color = clr[1], alpha = 0.2)


plt.plot(n_spikes_total,unknown_mixing_median, label = 'Unknown Low Rank Factorization', color = clr[2])
plt.plot(n_spikes_total,unknown_mixing_q1, color = clr[2], alpha = 0.3, linestyle = '--')
plt.plot(n_spikes_total,unknown_mixing_q3, color = clr[2], alpha = 0.3, linestyle = '--')
plt.gca().fill_between(n_spikes_total, unknown_mixing_q1, unknown_mixing_q3,  color = clr[2], alpha = 0.2)


ax = plt.gca()
ax.set_xticks(n_spikes_total[0::2])
ax.set_xticklabels(n_spikes_total[0::2])
# plt.ylim(1e-5,10)
plt.ylim(1e-22, 1e5)
ax.set_yscale("log")
plt.legend(loc = 'upper right')
plt.xlabel("Total number of spikes")
plt.ylabel("Reconstruction Error")
ax.axvline(total_deg_freedom, color=clr[3], linestyle = '--')
ax.axvline(apparent_deg_freedom, color=clr[4], linestyle = '--')

ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', alpha = 0.2)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


plt.tight_layout()
plt.savefig(figure_filename)