import os
import sys
import numpy as np
from tqdm import tqdm
from enum import Enum
import pickle

graphical_import = True

if graphical_import:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rc

sys.path.insert(
    0, os.path.split(os.path.realpath(__file__))[0] + "/../Multi-Channel-Time-Encoding"
)
from src import *


class RecoveryType(Enum):
    no_mixing = 1
    known_mixing = 2
    unknown_mixing = 3

def create_signal(num_signals, t, delta_t, Omega, sinc_padding):
    x = np.zeros((num_signals, len(t)))
    x_param = Signal.bandlimitedSignals(Omega)
    for n in range(num_signals):
        next_signal = Signal.bandlimitedSignal(Omega)
        next_signal.random(t, padding = sinc_padding)
        x_param.add(next_signal)
    return x_param

def get_params_for_spike_rate(x_param, t, A, end_time, num_spikes):
    n_signals = x_param.n_signals
    x_ints = np.zeros((n_signals, 1))
    for n in range(n_signals):
        x_ints[n] = x_param.get_signal(n).get_precise_integral(0, end_time)
    y_ints = np.array(A).dot(x_ints)
    n_channels = len(A)
    b = [1]*n_channels
    kappa = [1] * n_channels
    delta = [1] * n_channels
    kappadelta = [2 * kappa[l] * delta[l] for l in range(n_channels)]
    for m in range(n_channels):
        needed_integral = (num_spikes[m] +0.6) * kappadelta[m]
        b[m] = (needed_integral - y_ints[m, 0]) / end_time
    return b, kappa, delta


def get_results(x_param, A, t, end_time, Omega, kappa, delta, b, recovType = RecoveryType.no_mixing):
    start_index = int(len(t) / 5)
    end_index = int(len(t) * 4 / 5)
    num_channels = len(A)
    delta_t = t[1] - t[0]
    tem_mult = TEMParams(kappa, delta, b, A)
    spikes = Encoder.ContinuousEncoder(tem_mult).encode(x_param, end_time, tol=1e-10, with_start_time = True)
    if recovType == RecoveryType.unknown_mixing:
        rec_mult = Decoder.UnknownMixingDecoder(tem_mult).decode(spikes, t, 2, sinc_locs = x_param.get_sinc_locs(), Omega = Omega)
    elif recovType == RecoveryType.known_mixing:
        rec_mult = Decoder.MSignalMChannelDecoder(tem_mult).decode(spikes, t, x_param.get_sinc_locs(), Omega, Delta_t = None)
        rec_mult = A.dot(rec_mult)
    else:
        tem_mult = TEMParams(kappa, delta, b, np.eye(num_channels))
        rec_mult = Decoder.MSignalMChannelDecoder(tem_mult).decode(spikes, t, x_param.get_sinc_locs(), Omega, Delta_t = None)


    x_0 = np.array(A[0,:]).dot(x_param.sample(t))
    x_1 = np.array(A[1,:]).dot(x_param.sample(t))
    res1 = np.mean(((rec_mult[0, :] - x_0) ** 2)[start_index:end_index]) / np.mean(
        x_0[start_index:end_index] ** 2
    )
    res2 = np.mean(((rec_mult[1, :] - x_1) ** 2)[start_index:end_index]) / np.mean(
        x_1[start_index:end_index] ** 2
    )
    print(res1, res2)
    return res1, res2


def GetData(data_filename, recovType = RecoveryType.no_mixing, num_spikes = None, num_trials = None, seed = 0):
    # Settings for x_param
    end_time = 25
    sinc_padding = 0
    delta_t = 1e-2
    t = np.arange(0, end_time + delta_t, delta_t)
    Omega = np.pi
    np.random.seed(int(seed))
    num_signals = 2
    num_sincs = end_time - sinc_padding * 2
    total_deg_freedom = num_sincs * num_signals

    # Settings for time encoding machine
    num_channels = 20
    apparent_deg_freedom = num_sincs * num_channels
    A = np.random.normal(size = (num_channels,num_signals))

    if num_spikes is None:
        num_spikes_range = np.arange(2, 28, 1)
    else:
        num_spikes_range = np.array([num_spikes])


    n_spikes_total = np.zeros_like(num_spikes_range)
    n_spikes_constrained = np.zeros_like(num_spikes_range)
    for n_s_r in range(len(num_spikes_range)):
        n_spikes_constrained[n_s_r] = (
            min(num_spikes_range[n_s_r], num_sincs)*num_channels
        )
        n_spikes_total[n_s_r] = (
            num_spikes_range[n_s_r]*num_channels
        )

    # Settings for Simulation
    if num_trials is None:
        num_trials = 10

    results = np.zeros((num_trials, num_signals, len(num_spikes_range)))

    for n_t in tqdm(range(num_trials)):
        x_param = create_signal(num_signals, t, delta_t, Omega, sinc_padding)
        for n_s_r in range(len(num_spikes_range)):
            b, kappa, delta = get_params_for_spike_rate(
                x_param,
                t,
                A,
                end_time,
                [num_spikes_range[n_s_r]]*num_channels,
            )
            res1, res2 = get_results(x_param, A, t, end_time, Omega, kappa, delta, b, recovType)
            results[n_t, 0, n_s_r] = res1
            results[n_t, 1, n_s_r] = res2
            with open(data_filename+".part", "wb") as f:  # Python 3: open(..., 'wb')
                pickle.dump(
                    [n_spikes_total, n_spikes_constrained, results, total_deg_freedom, apparent_deg_freedom], f
                )

    with open(data_filename, "wb") as f:  # Python 3: open(..., 'wb')
        pickle.dump(
            [n_spikes_total, n_spikes_constrained, results, total_deg_freedom, apparent_deg_freedom], f
        )


if __name__ =="__main__":
    n_trials = 25
    seed = 0
    Figure_Path = os.path.split(os.path.realpath(__file__))[0] + "/../Figures/"
    Data_Path = os.path.split(os.path.realpath(__file__))[0] + "/../Data/"
    if len(sys.argv)>1:
        n_trials = int(sys.argv[1])
    filename = "known_mixing"

    data_filename = Data_Path + filename+".pkl"

    if not os.path.isfile(data_filename):
        GetData(data_filename, recovType = RecoveryType.known_mixing, num_trials = n_trials, seed = seed)

    filename = "unknown_mixing"
    data_filename = Data_Path + filename+".pkl"

    if not os.path.isfile(data_filename):
        GetData(data_filename, recovType = RecoveryType.unknown_mixing, num_trials = n_trials, seed = seed)

    filename = "no_mixing"
    data_filename = Data_Path + filename+".pkl"

    if not os.path.isfile(data_filename):
            GetData(data_filename, recovType = RecoveryType.no_mixing, num_trials = n_trials, seed = seed)


