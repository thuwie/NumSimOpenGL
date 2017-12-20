import numpy as np


class RungeKutta():
    def __init__(self, size=(100, 100), nwave=10, max_height=0.2):
        self._size = size
        self._wave_vector = 5 * (2 * np.random.rand(nwave, 2) - 1)
        self._angular_frequency = 2 * np.random.rand(nwave)
        self._phase = 2 * np.pi * np.random.rand(nwave)
        self._amplitude = max_height * (1 + np.random.rand(nwave)) / 2 / nwave
        self.t = 0
        self.speed = 1
        self.step = 0.0202
        # self.h_before = self.height_old(0)
        self.p = self.init_height()
        self.h_before = self.p[0]
        self.h_now = self.h_before
        self.h_diff = self.p[1]
        # self.h_now = self.init_height2()
        # self.speed = 0
        self.dt = 0.0001
        self.dt = 1

    def init_height(self):
        height = np.zeros(self._size, dtype=np.float32)
        v = np.zeros(self._size, dtype=np.float32)
        x = np.linspace(-1, 1, self._size[0])[:, None]
        height[:, :] = (np.cos(2 * x * np.pi) * np.cos(2 * np.pi * self.speed * self.t)) / 10
        return height, v

    def init_height2(self):
        height = np.zeros(self._size, dtype=np.float32)
        x = np.linspace(-1, 1, self._size[0])[:, None]
        height[:, :] = np.sin(x)
        return height

    def position(self):
        xy = np.empty(self._size + (2,), dtype=np.float32)
        xy[:, :, 0] = np.linspace(-1, 1, self._size[0])[:, None]
        xy[:, :, 1] = np.linspace(-1, 1, self._size[1])[None, :]
        return xy

    def propagate(self, dt):
        self.t += dt

    def height_nowie(self):
        height_now = self.h_now  # cкопирует укзатель

        delta = (height_now[2:, 1:-1] + height_now[:-2, 1:-1] + height_now[1:-1, :-2] + height_now[1:-1, 2:]) - 4 * height_now[1:-1, 1:-1]
        coe = self.speed ** 2 / 2

        energy = (np.power(self.h_now, 2) * self.step * self.step) + (np.power(self.h_now, 2) * self.step) + (np.power(self.h_now, 2) * self.step)
        print('Energy = ', np.sum(energy))

        k1_first = self.h_diff
        k1_second = np.zeros(self._size, dtype=np.float32)
        k1_second[1: -1, 1: -1] = coe * delta

        P2_first = height_now + (self.dt / 2) * k1_first
        P2_second = self.h_diff + (self.dt / 2) * k1_second

        k2_first = P2_second
        k2_second = np.zeros(self._size, dtype=np.float32)
        delta = (P2_first[2:, 1:-1] + P2_first[:-2, 1:-1] + P2_first[1:-1, :-2] + P2_first[1:-1, 2:]) - 4 * P2_first[1:-1, 1:-1]
        k2_second[1: -1, 1: -1] = coe * delta

        P3_first = height_now + (self.dt / 2) * k2_first
        P3_second = self.h_diff + (self.dt / 2) * k2_second

        k3_first = P3_second
        k3_second = np.zeros(self._size, dtype=np.float32)
        delta = (P3_first[2:, 1:-1] + P3_first[:-2, 1:-1] + P3_first[1:-1, :-2] + P3_first[1:-1, 2:]) - 4 * P3_first[1:-1, 1:-1]
        k3_second[1: -1, 1: -1] = coe * delta

        P4_first = height_now + self.dt * k3_first
        P4_second = self.h_diff + self.dt * k3_second

        k4_first = P4_second
        k4_second = np.zeros(self._size, dtype=np.float32)
        delta = (P4_first[2:, 1:-1] + P4_first[:-2, 1:-1] + P4_first[1:-1, :-2] + P4_first[1:-1, 2:]) - 4 * P4_first[1:-1, 1:-1]
        k4_second[1: -1, 1: -1] = coe * delta

        P_first = height_now + (self.dt / 6) * (k1_first + 2 * k2_first + 2 * k3_first + k4_first)
        P_second = self.h_diff + (self.dt / 6) * (k1_second + 2 * k2_second + 2 * k3_second + k4_second)

        P_first[0, :] = P_first[1, :]
        P_first[self._size[0] - 1, :] = P_first[self._size[0] - 2, :]
        P_first[:, 0] = P_first[:, 1]
        P_first[:, self._size[1] - 1] = P_first[:, self._size[1] - 2]

        P_second[0, :] = P_second[1, :]
        P_second[self._size[0] - 1, :] = P_second[self._size[0] - 2, :]
        P_second[:, 0] = P_second[:, 1]
        P_second[:, self._size[1] - 1] = P_second[:, self._size[1] - 2]

        self.h_now = P_first
        self.h_diff = P_second

        return self.h_now

    def height(self):
        x = np.linspace(-1, 1, self._size[0])[:, None]
        y = np.linspace(-1, 1, self._size[1])[None, :]
        z = np.zeros(self._size, dtype=np.float32)
        z[:, :] = np.sin(self.t * np.pi * x) * np.cos(self.t * np.pi * y) / 15
        return z

    def height_old(self, t):
        x = np.linspace(-1, 1, self._size[0])[:, None]
        y = np.linspace(-1, 1, self._size[1])[None, :]
        z = np.zeros(self._size, dtype=np.float32)
        for n in range(self._amplitude.shape[0]):
            arg = self._phase[n] + x * self._wave_vector[n, 0] + y * self._wave_vector[n, 1] + \
                  t * self._angular_frequency[n]
            z[:, :] += self._amplitude[n] * np.cos(arg)
        return z

    def normal_an(self, h, delta):
        x = np.linspace(-1, 1, self._size[0])[:, None]
        y = np.linspace(-1, 1, self._size[1])[None, :]
        grad_x = self.t * np.pi * np.cos(self.t * np.pi * x) * np.cos(self.t * np.pi * y) / 15
        grad_y = -self.t * np.pi * np.sin(self.t * np.pi * x) * np.sin(self.t * np.pi * y) / 15
        grad = np.zeros(self._size + (2,), dtype=np.float32)
        grad[:, :, 0] = grad_x
        grad[:, :, 1] = grad_y
        return grad

    def normal(self, h, delta):
        grad = np.zeros(self._size + (2,), dtype=np.float32)
        h_first_col = h[:, 0]
        h_last_col = h[:, -1]
        h_first_row = h[0, :]
        h_last_row = h[-1, :]
        hx = np.c_[h_first_col, h, h_last_col]
        hy = np.r_[[h_first_row], h, [h_last_row]]
        nx = (hx[:, 2:] - hx[:, :-2]) / (2 * delta)
        ny = (hy[2:, :] - hy[:-2, :]) / (2 * delta)
        grad[:, :, 0] = nx
        grad[:, :, 1] = ny
        return grad

    def height_and_normal(self):
        x = np.linspace(-1, 1, self._size[0])[:, None]
        y = np.linspace(-1, 1, self._size[1])[None, :]
        z = np.zeros(self._size, dtype=np.float32)
        grad = np.zeros(self._size + (2,), dtype=np.float32)
        for n in range(self._amplitude.shape[0]):
            arg = self._phase[n] + x * self._wave_vector[n, 0] + y * self._wave_vector[n, 1] + \
                  self.t * self._angular_frequency[n]
            z[:, :] += self._amplitude[n] * np.cos(arg)
            dcos = -self._amplitude[n] * np.sin(arg)
            grad[:, :, 0] += self._wave_vector[n, 0] * dcos
            grad[:, :, 1] += self._wave_vector[n, 1] * dcos
        return z, grad

    def triangulation(self):
        a = np.indices((self._size[0] - 1, self._size[1] - 1))
        b = a + np.array([1, 0])[:, None, None]
        c = a + np.array([1, 1])[:, None, None]
        d = a + np.array([0, 1])[:, None, None]
        a_r = a.reshape((2, -1))
        b_r = b.reshape((2, -1))
        c_r = c.reshape((2, -1))
        d_r = d.reshape((2, -1))
        a_l = np.ravel_multi_index(a_r, self._size)
        b_l = np.ravel_multi_index(b_r, self._size)
        c_l = np.ravel_multi_index(c_r, self._size)
        d_l = np.ravel_multi_index(d_r, self._size)
        abc = np.concatenate((a_l[..., None], b_l[..., None], c_l[..., None]), axis=-1)
        acd = np.concatenate((a_l[..., None], c_l[..., None], d_l[..., None]), axis=-1)
        return np.concatenate((abc, acd), axis=0).astype(np.uint32)


class Euler:
    def __init__(self, size=(100, 100), nwave=10, max_height=0.2):
        self._size = size
        self._wave_vector = 5 * (2 * np.random.rand(nwave, 2) - 1)
        self._angular_frequency = 2 * np.random.rand(nwave)
        self._phase = 2 * np.pi * np.random.rand(nwave)
        self._amplitude = max_height * (1 + np.random.rand(nwave)) / 2 / nwave
        self.t = 0
        self.speed = 1
        self.step = 0.0202
        # self.h_before = self.height_old(0)
        self.p = self.init_height()
        self.h_before = self.p[0]
        self.h_now = self.h_before
        self.h_diff = self.p[1]
        # self.h_now = self.init_height2()
        # self.speed = 0
        self.dt = 0.0001
        self.dt = 1

    def init_height(self):
        height = np.zeros(self._size, dtype=np.float32)
        v = np.zeros(self._size, dtype=np.float32)
        x = np.linspace(-1, 1, self._size[0])[:, None]
        height[:, :] = (np.cos(2 * x * np.pi) * np.cos(2 * np.pi * self.speed * self.t)) / 10
        return height, v

    def init_height2(self):
        height = np.zeros(self._size, dtype=np.float32)
        x = np.linspace(-1, 1, self._size[0])[:, None]
        height[:, :] = np.sin(x)
        return height

    def position(self):
        xy = np.empty(self._size + (2,), dtype=np.float32)
        xy[:, :, 0] = np.linspace(-1, 1, self._size[0])[:, None]
        xy[:, :, 1] = np.linspace(-1, 1, self._size[1])[None, :]
        return xy

    def propagate(self, dt):
        self.t += dt

    def height_nowie(self):
        height_now = self.h_now
        height_before = self.h_before
        height_future = np.zeros(self._size, dtype=np.float32)

        delta = (height_now[2:, 1:-1] + height_now[:-2, 1:-1] + height_now[1:-1, :-2] + height_now[1:-1, 2:]) - 4 * height_now[1:-1, 1:-1]

        coe = self.speed ** 2 / 2

        energy = (np.power(self.h_now, 2) * self.step * self.step) + (np.power(self.h_now, 2) * self.step) + (
        np.power(self.h_now, 2) * self.step)
        print('Energy = ', np.sum(energy))

        P_tilda_second = np.zeros(self._size, dtype=np.float32)

        P_tilda_first = height_now + self.dt * self.h_diff
        P_tilda_second[1: -1, 1: -1] = self.h_diff[1: -1, 1: -1] + self.dt * coe * delta

        P_eler_first = np.zeros(self._size, dtype=np.float32)
        P_eler_second = np.zeros(self._size, dtype=np.float32)

        P_eler_first = height_now + (coe / 2) * (self.h_diff + P_tilda_second)
        P_eler_second[1: -1, 1: -1] = self.h_diff[1: -1, 1: -1] + (coe / 2) * (coe * delta + coe * ((P_tilda_first[2:, 1:-1] + P_tilda_first[:-2, 1:-1] + P_tilda_first[1:-1, :-2] + P_tilda_first[1:-1, 2:]) - 4 * P_tilda_first[1:-1, 1:-1]))

        P_eler_first[0, :] = P_eler_first[1, :]
        P_eler_first[self._size[0] - 1, :] = P_eler_first[self._size[0] - 2, :]
        P_eler_first[:, 0] = P_eler_first[:, 1]
        P_eler_first[:, self._size[1] - 1] = P_eler_first[:, self._size[1] - 2]

        P_eler_second[0, :] = P_eler_second[1, :]
        P_eler_second[self._size[0] - 1, :] = P_eler_second[self._size[0] - 2, :]
        P_eler_second[:, 0] = P_eler_second[:, 1]
        P_eler_second[:, self._size[1] - 1] = P_eler_second[:, self._size[1] - 2]

        self.h_now = P_eler_first
        self.h_diff = P_eler_second
        return self.h_now

    def height(self):
        x = np.linspace(-1, 1, self._size[0])[:, None]
        y = np.linspace(-1, 1, self._size[1])[None, :]
        z = np.zeros(self._size, dtype=np.float32)
        z[:, :] = np.sin(self.t * np.pi * x) * np.cos(self.t * np.pi * y) / 15
        return z

    def height_old(self, t):
        x = np.linspace(-1, 1, self._size[0])[:, None]
        y = np.linspace(-1, 1, self._size[1])[None, :]
        z = np.zeros(self._size, dtype=np.float32)
        for n in range(self._amplitude.shape[0]):
            arg = self._phase[n] + x * self._wave_vector[n, 0] + y * self._wave_vector[n, 1] + \
                  t * self._angular_frequency[n]
            z[:, :] += self._amplitude[n] * np.cos(arg)
        return z

    def normal_an(self, h, delta):
        x = np.linspace(-1, 1, self._size[0])[:, None]
        y = np.linspace(-1, 1, self._size[1])[None, :]
        grad_x = self.t * np.pi * np.cos(self.t * np.pi * x) * np.cos(self.t * np.pi * y) / 15
        grad_y = -self.t * np.pi * np.sin(self.t * np.pi * x) * np.sin(self.t * np.pi * y) / 15
        grad = np.zeros(self._size + (2,), dtype=np.float32)
        grad[:, :, 0] = grad_x
        grad[:, :, 1] = grad_y
        return grad

    def normal(self, h, delta):
        grad = np.zeros(self._size + (2,), dtype=np.float32)
        h_first_col = h[:, 0]
        h_last_col = h[:, -1]
        h_first_row = h[0, :]
        h_last_row = h[-1, :]
        hx = np.c_[h_first_col, h, h_last_col]
        hy = np.r_[[h_first_row], h, [h_last_row]]
        nx = (hx[:, 2:] - hx[:, :-2]) / (2 * delta)
        ny = (hy[2:, :] - hy[:-2, :]) / (2 * delta)
        grad[:, :, 0] = nx
        grad[:, :, 1] = ny
        return grad

    def height_and_normal(self):
        x = np.linspace(-1, 1, self._size[0])[:, None]
        y = np.linspace(-1, 1, self._size[1])[None, :]
        z = np.zeros(self._size, dtype=np.float32)
        grad = np.zeros(self._size + (2,), dtype=np.float32)
        for n in range(self._amplitude.shape[0]):
            arg = self._phase[n] + x * self._wave_vector[n, 0] + y * self._wave_vector[n, 1] + \
                  self.t * self._angular_frequency[n]
            z[:, :] += self._amplitude[n] * np.cos(arg)
            dcos = -self._amplitude[n] * np.sin(arg)
            grad[:, :, 0] += self._wave_vector[n, 0] * dcos
            grad[:, :, 1] += self._wave_vector[n, 1] * dcos
        # print(grad[:,:, 0][1][1:6])
        return z, grad

    def triangulation(self):
        a = np.indices((self._size[0] - 1, self._size[1] - 1))
        b = a + np.array([1, 0])[:, None, None]
        c = a + np.array([1, 1])[:, None, None]
        d = a + np.array([0, 1])[:, None, None]
        a_r = a.reshape((2, -1))
        b_r = b.reshape((2, -1))
        c_r = c.reshape((2, -1))
        d_r = d.reshape((2, -1))
        a_l = np.ravel_multi_index(a_r, self._size)
        b_l = np.ravel_multi_index(b_r, self._size)
        c_l = np.ravel_multi_index(c_r, self._size)
        d_l = np.ravel_multi_index(d_r, self._size)
        abc = np.concatenate((a_l[..., None], b_l[..., None], c_l[..., None]), axis=-1)
        acd = np.concatenate((a_l[..., None], c_l[..., None], d_l[..., None]), axis=-1)
        return np.concatenate((abc, acd), axis=0).astype(np.uint32)

    def bed_uneven(size, max_val):
        begin_arr = np.linspace(0, max_val, size / 2)
        end_arr = np.linspace(max_val, 0, size / 2)
        line = np.concatenate((begin_arr, end_arr))
        stack_line = np.stack((line, line))
        for i in range(0, size - 2):
            stack_line = np.vstack([stack_line, line])
        stack_line = stack_line + 1
        return stack_line


class PlaneWaves(object):
    def __init__(self, size=(100, 100), waves=1, max_height=0.2):
        self._size = size
        self._wave_vector = 5 * (2 * np.random.rand(waves, 2) - 1)
        self._angular_frequency = 2 * np.random.rand(waves)
        self._phase = 2 * np.pi * np.random.rand(waves)
        self._amplitude = max_height * (1 + np.random.rand(waves)) / 2 / waves
        self.t = 0

    def position(self):
        xy = np.empty(self._size + (2,), dtype=np.float32)
        xy[:, :, 0] = np.linspace(-1, 1, self._size[0])[:, None]
        xy[:, :, 1] = np.linspace(-1, 1, self._size[1])[None, :]
        return xy

    def propagate(self, dt):
        self.t += dt

    def height_and_normal(self):
        x = np.linspace(-1, 1, self._size[0])[:, None]
        y = np.linspace(-1, 1, self._size[1])[None, :]
        z = np.zeros(self._size, dtype=np.float32)
        grad = np.zeros(self._size + (2,), dtype=np.float32)
        for n in range(self._amplitude.shape[0]):
            arg = self._phase[n] + x * self._wave_vector[n, 0] + y * self._wave_vector[n, 1] + self.t * \
                                                                                               self._angular_frequency[
                                                                                                   n]
            z[:, :] += self._amplitude[n] * np.cos(arg)
            dcos = -self._amplitude[n] * np.sin(arg)
            grad[:, :, 0] += self._wave_vector[n, 0] * dcos
            grad[:, :, 1] += self._wave_vector[n, 1] * dcos
        return z, grad

    def triangulation(self):
        a = np.indices((self._size[0] - 1, self._size[1] - 1))
        b = a + np.array([1, 0])[:, None, None]
        c = a + np.array([1, 1])[:, None, None]
        d = a + np.array([0, 1])[:, None, None]
        a_r = a.reshape((2, -1))
        b_r = b.reshape((2, -1))
        c_r = c.reshape((2, -1))
        d_r = d.reshape((2, -1))
        a_l = np.ravel_multi_index(a_r, self._size)
        b_l = np.ravel_multi_index(b_r, self._size)
        c_l = np.ravel_multi_index(c_r, self._size)
        d_l = np.ravel_multi_index(d_r, self._size)
        abc = np.concatenate((a_l[..., None], b_l[..., None], c_l[..., None]), axis=-1)
        acd = np.concatenate((a_l[..., None], c_l[..., None], d_l[..., None]), axis=-1)
        return np.concatenate((abc, acd), axis=0).astype(np.uint32)


class CircularWaves(PlaneWaves):
    def __init__(self, size=(100, 100), max_height=0.1, wave_length=0.3, center=(0., 0.), speed=3):
        self._size = size
        self._amplitude = max_height
        self._omega = 2 * np.pi / wave_length
        self._center = np.asarray(center, dtype=np.float32)
        self._speed = speed
        self.t = 0

    def height_and_normal(self):
        x = np.linspace(-1, 1, self._size[0])[:, None]
        y = np.linspace(-1, 1, self._size[1])[None, :]
        z = np.empty(self._size, dtype=np.float32)
        grad = np.zeros(self._size + (2,), dtype=np.float32)
        d = np.sqrt((x - self._center[0]) ** 2 + (y - self._center[1]) ** 2)
        arg = self._omega * d - self.t * self._speed
        z[:, :] = self._amplitude * np.cos(arg)
        dcos = -self._amplitude * self._omega * np.sin(arg)
        grad[:, :, 0] += (x - self._center[0]) * dcos / d
        grad[:, :, 1] += (y - self._center[1]) * dcos / d
        return z, grad


class Surface(PlaneWaves):
    pass
