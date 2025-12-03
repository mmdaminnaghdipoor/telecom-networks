GFSK
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, hilbert

# ---------------------------------------
# GFSK MODEM CLASS
# ---------------------------------------
class GFSKModem:
    def __init__(self, sps=20, BT=0.3, freq_dev=500, fs=20000, fc=2000):
        self.sps = sps
        self.BT = BT
        self.freq_dev = freq_dev
        self.fs = fs
        self.dt = 1 / fs
        self.fc = fc
        self.gaussian_filter = self._gaussian_filter()

    # Gaussian Pulse Shaping Filter
    def _gaussian_filter(self, span_symbols=4):
        sigma = 0.44 / self.BT
        t = np.linspace(-span_symbols, span_symbols, 2 * span_symbols * self.sps + 1)
        g = np.exp(-0.5 * (t / sigma) ** 2)
        return g / np.sum(g)

    # -----------------------------
    # MODULATION
    # -----------------------------
    def modulate(self, bits):
        symbols = 2 * bits - 1
        upsampled = np.repeat(symbols, self.sps)

        shaped = fftconvolve(upsampled, self.gaussian_filter, mode='same')

        inst_freq = shaped * self.freq_dev
        phase = 2 * np.pi * np.cumsum(inst_freq) * self.dt

        t = np.arange(len(phase)) * self.dt
        signal = np.cos(2 * np.pi * self.fc * t + phase)

        return signal, shaped, inst_freq

    # -----------------------------
    # DEMODULATION
    # -----------------------------
    def demodulate(self, signal, num_bits):
        analytic = hilbert(signal)
        phase = np.unwrap(np.angle(analytic))

        inst_freq = np.diff(phase) / (2 * np.pi * self.dt)
        inst_freq = np.concatenate(([inst_freq[0]], inst_freq))

        smooth = np.convolve(inst_freq, np.ones(self.sps//2)/ (self.sps//2), mode='same')

        sample_points = np.arange(num_bits) * self.sps + self.sps//2
        detected = (smooth[sample_points] > 0).astype(int)

        return detected, smooth

    # -----------------------------
    # PLOT
    # -----------------------------
    def plot(self, shaped, inst_freq, signal, inst_freq_demod, bits, detected):
        fig, axs = plt.subplots(4, 1, figsize=(12, 12))

        axs[0].plot(shaped)
        axs[0].set_title("Gaussian Filtered Baseband (Shaped Symbols)")
        axs[0].grid()

        axs[1].plot(inst_freq)
        axs[1].set_title("Instantaneous Frequency of Modulated Signal")
        axs[1].grid()

        axs[2].plot(signal)
        axs[2].set_title("GFSK Modulated Passband Signal")
        axs[2].grid()

        axs[3].plot(inst_freq_demod)
        axs[3].plot(np.arange(len(bits))*self.sps + self.sps//2,
                    detected * np.max(inst_freq_demod), 'ro', label="Detected")
        axs[3].set_title("Demodulated Instantaneous Frequency")
        axs[3].grid()

        plt.tight_layout()
        plt.show()


# ---------------------------------------
# EXAMPLE USE IN JUPYTER
# ---------------------------------------

# Create modem
modem = GFSKModem()

# Random bitstream
bits = np.random.randint(0, 2, 100)

# Modulate
signal, shaped, inst_freq = modem.modulate(bits)

# Demodulate
detected, inst_freq_demod = modem.demodulate(signal, len(bits))

# Bit error rate
ber = np.mean(bits != detected)
print("BER =", ber)

# Plot results
modem.plot(shaped, inst_freq, signal, inst_freq_demod, bits, detected)
