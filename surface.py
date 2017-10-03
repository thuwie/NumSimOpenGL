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

    def wireframe(self):
        # Возвращаем координаты всех вершин, кроме крайнего правого столбца
        left = np.indices((self._size[0] - 1, self._size[1]))
        # Пересчитываем в координаты всех точек, кроме крайнего левого столбца
        right = left + np.array([1, 0])[:, None, None]
        # Преобразуем массив точек в список (одномерный массив)
        left_r = left.reshape((2, -1))
        right_r = right.reshape((2, -1))
        # Заменяем многомерные индексы линейными индексами
        left_l = np.ravel_multi_index(left_r, self._size)
        right_l = np.ravel_multi_index(right_r, self._size)
        # собираем массив пар точек
        horizontal = np.concatenate((left_l[..., None], right_l[..., None]), axis=-1)
        # делаем то же самое для вертикальных отрезков
        bottom = np.indices((self._size[0], self._size[1] - 1))
        top = bottom + np.array([0, 1])[:, None, None]
        bottom_r = bottom.reshape((2, -1))
        top_r = top.reshape((2, -1))
        bottom_l = np.ravel_multi_index(bottom_r, self._size)
        top_l = np.ravel_multi_index(top_r, self._size)
        vertical = np.concatenate((bottom_l[..., None], top_l[..., None]), axis=-1)
        return np.concatenate((horizontal, vertical), axis=0).astype(np.uint32)
