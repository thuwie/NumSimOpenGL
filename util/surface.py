import numpy as np


class Surface(object):
    def __init__(self, size=(150, 150), waves=7):
        self._size = size
        self._wave_vector = 5 * np.random.randn(waves, 2)
        self._angular_frequency = np.random.randn(waves)
        self._phase = 2 * np.pi * np.random.rand(waves)
        self._amplitude = np.random.rand(waves) / waves

    def position(self):
        pos = np.empty(self._size + (2,), dtype=np.float32)
        pos[:, :, 0] = np.linspace(-1, 1, self._size[0])[:, None]
        pos[:, :, 1] = np.linspace(-1, 1, self._size[1])[None, :]
        return pos

    def height(self, t):
        x = np.linspace(-1, 1, self._size[0])[:, None]
        y = np.linspace(-1, 1, self._size[1])[None, :]
        z = np.zeros(self._size, dtype=np.float32)
        for n in range(self._amplitude.shape[0]):
            z[:, :] += self._amplitude[n] *\
                       np.cos(self._phase[n] +
                              x * self._wave_vector[n, 0] +
                              y * self._wave_vector[n, 1] +
                              t * self._angular_frequency[n])
        return z
