import cv2
import sys
import os
import matplotlib.pyplot as plt

sys.path.insert(
    0, os.path.split(os.path.realpath(__file__))[0] + "/../Multi-Channel-Time-Encoding"
)
from src import *

Input_Data_Path = (
    os.path.split(os.path.realpath(__file__))[0] + "/../Video Data/kid_swing/"
)


def time_encode_video(video, TEM_locations, rank=10, num_spikes=None, plot=False):

    signals = Signal.periodicBandlimitedSignals(period=video.periods[-1])
    deltas = []
    if num_spikes is None:
        num_spikes = video.periods[-1] + 8.5

    for TEM_l in TEM_locations:
        signal_l = video.get_time_signal(TEM_l)
        signals.add(signal_l)
        deltas.append(
            signal_l.get_precise_integral(0, video.periods[-1]) / (2 * num_spikes)
        )

    kappa, b = 1, 0
    tem_mult = TEMParams(kappa, deltas, b, np.eye(len(TEM_locations)))
    end_time = video.periods[-1]
    spikes = Encoder.ContinuousEncoder(tem_mult).encode(
        signals, end_time, tol=1e-14, with_start_time=False
    )

    integrals, integral_start_coordinates, integral_end_coordinates = Decoder.MSignalMChannelDecoder(
        tem_mult
    ).get_vid_constraints(
        spikes, TEM_locations
    )
    coefficients = video.get_coefficients_from_integrals(
        integral_start_coordinates, integral_end_coordinates, integrals
    )

    if plot:
        for channel in range(len(TEM_locations)):
            plt.figure(figsize=(4, 1.5))
            plt.stem(
                spikes.get_spikes_of(channel),
                np.ones_like(spikes.get_spikes_of(channel)),
                markerfmt="^",
            )
            ax = plt.gca()
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_xlabel("Time")
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.set_ylim([0.01, 1.1])
            plt.savefig("Simulations/Figures/Temp/ch" + str(channel) + ".svg")
            plt.savefig("Simulations/Figures/Temp/ch" + str(channel) + ".png")

    return (
        np.linalg.norm((coefficients.flatten() - video.freq_domain_samples.flatten()))
        / np.linalg.norm((video.fft.flatten())),
        spikes.get_const_num_constraints(video.periods[-1]),
    )


if __name__ == "__main__":

    VID_WIDTH = 9
    VID_HEIGHT = 9
    num_images = 9

    im = cv2.imread(Input_Data_Path + "00001.jpg")
    h_sep, v_sep = 2, 2

    im_width, im_height = int(im.shape[0] / h_sep), int(im.shape[1] / v_sep)
    video = np.zeros((int(im_width / h_sep), int(im_height / v_sep), num_images, 3))

    for i in range(num_images):
        filename = Input_Data_Path + "{:05d}.jpg".format(i + 1)
        im = cv2.imread(filename)
        im = cv2.resize(im, (int(im.shape[1] / h_sep), int(im.shape[0] / v_sep)))
        video[:, :, i, :] = im[:im_width:h_sep, :im_height:v_sep, :]

    subvideo = video[88 : 88 + VID_HEIGHT, 155 : 155 + VID_WIDTH, ::-1, 0]
    opt = {"time_domain_samples": subvideo}
    signal = MultiDimPeriodicSignal(opt)

    do_spike_sweep = False
    spacings = ["9x15", "9x9", "9x5"]
    spacing_n = 2
    spike_rates = ["9_5", "9_9", "9_15"]
    spike_rate_n = 0
    do_n_TEM_sweep = True

    if do_spike_sweep:
        # N SPIKE SWEEP
        TEM_locations = []

        if spacing_n == 0:
            hor_separation, ver_separation = 1, 9 / 15.0
            num_spikes = 1.5 + np.arange(3, 12, 1)
            num_spikes = np.concatenate((num_spikes, 1.5 + np.arange(12, 37, 4)))
        elif spacing_n == 1:
            hor_separation, ver_separation = 1, 1
            num_spikes = 1.5 + np.arange(3, 16, 1)
            num_spikes = np.concatenate((num_spikes, 1.5 + np.arange(16, 37, 4)))
        elif spacing_n == 2:
            hor_separation, ver_separation = 1, 9 / 5.0
            num_spikes = 1.5 + np.arange(3, 37, 3)

        hor_loc_range = np.arange(0.1, VID_WIDTH, hor_separation)
        ver_loc_range = np.arange(0.1, VID_HEIGHT, ver_separation)
        for h in hor_loc_range:
            for v in ver_loc_range:
                TEM_locations.append([v, h])

        errors = []
        for n_s in num_spikes:
            error, n_constraints = time_encode_video(
                signal, TEM_locations, num_spikes=n_s
            )
            print(
                "Horizontal Separation:",
                hor_separation,
                ", Vertical Separation:",
                ver_separation,
                ", Error: ",
                error,
                ", Number of constraints:",
                n_constraints,
                ", Number of TEMs: ",
                len(TEM_locations),
            )
            errors.append(error)

        import pickle

        data_filename =(
            os.path.split(os.path.realpath(__file__))[0]
            + "/../Data/nspike_sweep_" + spacings[spacing_n] + "_spacing.pkl"
            )
        with open(data_filename, "wb") as f: 
            pickle.dump([num_spikes, errors], f)

    if do_n_TEM_sweep:

        n_TEMs = np.arange(4, 20)
        n_TEMs = np.arange(5, 15)
        errors = []


        if spike_rate_n == 0:
            spike_rate = 5
        elif spike_rate_n == 1:
            spike_rate = 9
        elif spike_rate_n == 2:
            spike_rate = 15

        for n_t in n_TEMs:

            TEM_locations = []
            hor_separation, ver_separation = VID_WIDTH / (n_t), VID_HEIGHT / n_t
            hor_loc_range = np.arange(0.1, VID_WIDTH, hor_separation)
            ver_loc_range = np.arange(0.1, VID_HEIGHT, ver_separation)
            for h in hor_loc_range:
                for v in ver_loc_range:
                    TEM_locations.append([h, v])

            error, n_constraints = time_encode_video(
                signal, TEM_locations, num_spikes=spike_rate + 1.5
            )
            print(
                "Horizontal Separation:",
                hor_separation,
                ", Vertical Separation:",
                ver_separation,
                ", Error: ",
                error,
                ", Number of constraints:",
                n_constraints,
                ", Number of TEMs: ",
                len(TEM_locations),
            )
            errors.append(error)

        import pickle

        data_filename = (
            os.path.split(os.path.realpath(__file__))[0]
            + "/../Data/ntem_sweep_"
            + spike_rates[spike_rate_n]
            + "_spikes.pkl"
        )
        with open(data_filename, "wb") as f:  # Python 3: open(..., 'wb')
            pickle.dump([n_TEMs, errors], f)
