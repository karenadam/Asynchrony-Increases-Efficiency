import matplotlib
import matplotlib.pyplot as plt

import pickle

To_Svg = True

if To_Svg:
    figure_spike_sweep = "n_spikes_sweep.svg"
    figure_tem_sweep ="n_tems_sweep.svg"
    plt.rc('text', usetex=False)
    plt.rc('text.latex', unicode = False)
    plt.rc('svg',fonttype = 'none')

else:
    figure_spike_sweep =  "n_spikes_sweep.png"
    figure_tem_sweep = "n_tems_sweep.png"



clr = plt.rcParams["axes.prop_cycle"].by_key()["color"]

def set_x_y_lims():
    plt.xlim(3,36)
    plt.ylim(1e-14,10)
    pass

plt.figure(figsize = (6,6))
plt.subplot(3,1,1)

data_filename = "nspike_sweep_9x15_spacing.pkl"
with open(data_filename, "rb") as f:  # Python 3: open(..., 'wb')
    obj = pickle.load(f, encoding="latin1")
n_spikes, error = obj
plt.title("Mean-Squared Reconstruction Error")

plt.plot(n_spikes-1.5, error)
ax = plt.gca()
ax.set_yscale("log")
plt.axvline(6, color = clr[1], linewidth = 3)
plt.axvline(6, linestyle = (0, (5,5)), color = clr[8], linewidth = 3)
plt.ylabel("9x15 TEMs")#, rotation = 0, ha = 'right', va = 'center')
set_x_y_lims()

plt.subplot(3,1,2)

data_filename = "nspike_sweep_9x9_spacing.pkl"
with open(data_filename, "rb") as f:  # Python 3: open(..., 'wb')
    obj = pickle.load(f, encoding="latin1")
n_spikes, error = obj

plt.plot(n_spikes-1.5, error)
ax = plt.gca()
ax.set_yscale("log")
plt.axvline(9, color = clr[1], linewidth = 3)
plt.axvline(9, linestyle = (0, (5,5)), color = clr[8], linewidth = 3)
plt.ylabel("9x9 TEMs")#, rotation = 0, ha = 'right', va = 'center')
set_x_y_lims()


plt.subplot(3,1,3)

data_filename = "nspike_sweep_9x5_spacing.pkl"
with open(data_filename, "rb") as f:  # Python 3: open(..., 'wb')
    obj = pickle.load(f, encoding="latin1")
n_spikes, error = obj

plt.plot(n_spikes-1.5, error)
ax = plt.gca()
ax.set_yscale("log")
plt.axvline(17, linestyle = (0, (5,5)), color = clr[8], linewidth = 3)
plt.ylabel("9x5 TEMs")#, rotation = 0, ha = 'right', va = 'center')
plt.xlim(3, 36)
plt.ylim(0.1,1)
# print(plt.gca().get_yticks())
plt.gca().set_yticks([], minor = True)
plt.gca().set_yticks([0.1,1], minor = False)

print(plt.gca().get_yticks())

plt.xlabel("Number of Spikes per Machine")
plt.tight_layout()

plt.savefig(figure_spike_sweep)


plt.figure(figsize = (6,6))
def set_x_y_lims():
    plt.xlim(25,14*14)
    plt.ylim(1e-14,10)
    pass

plt.subplot(3,1,1)


data_filename = "ntem_sweep_9_5_spikes.pkl"
with open(data_filename, "rb") as f:  # Python 3: open(..., 'wb')
    obj = pickle.load(f, encoding="latin1")
n_tems, error = obj

plt.title("Mean-Squared Reconstruction Error")
plt.plot([n_t**2 for n_t in n_tems], error)
ax = plt.gca()
ax.set_yscale("log")
plt.axvline(169, color = clr[1], linewidth = 3)
plt.axvline(169, linestyle = (0, (5,5)), color = clr[8], linewidth = 3)
yl = plt.ylabel("8 spikes\n per TEM")#, ha = 'right', va = 'center')
set_x_y_lims()

plt.subplot(3,1,2)
data_filename = "ntem_sweep_9_9_spikes.pkl"
with open(data_filename, "rb") as f:  # Python 3: open(..., 'wb')
    obj = pickle.load(f, encoding="latin1")
n_tems, error = obj

plt.plot([n_t**2 for n_t in n_tems], error)
ax = plt.gca()
ax.set_yscale("log")
plt.axvline(81, color = clr[1], linewidth = 3)
plt.axvline(81, linestyle = (0, (5,5)), color = clr[8], linewidth = 3)
yl = plt.ylabel("15 spikes\n per TEM")#, rotation = 0, ha = 'right', va = 'center')

set_x_y_lims()

plt.subplot(3,1,3)
data_filename = "ntem_sweep_9_15_spikes.pkl"
with open(data_filename, "rb") as f:  # Python 3: open(..., 'wb')
    obj = pickle.load(f, encoding="latin1")
n_tems, error = obj

plt.plot([n_t**2 for n_t in n_tems], error)
ax = plt.gca()
ax.set_yscale("log")
plt.axvline(81, color = clr[1], linewidth = 3)
plt.axvline(49, linestyle = (0, (5,5)), color = clr[8], linewidth = 3)
yl = plt.ylabel("25 spikes\n per TEM")#, rotation = 0, ha = 'right', va = 'center')
# plt.xlim(25,15*15)

plt.xlabel("Number of TEMs")
set_x_y_lims()

plt.tight_layout()

plt.savefig(figure_tem_sweep)


















