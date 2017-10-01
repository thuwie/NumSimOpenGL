import numpy as np
from vispy import gloo
from vispy import app

from render2.surface import Surface

vert = ("""
attribute vec2 dot_position;
attribute float dot_height;
 void main(){
    float height=(1-dot_height)*0.5;
    gl_Position = vec4(dot_position.xy, height, height);
 }
""")

frag = ("""
void main(){
    gl_FragColor = vec4(0, 0.4, 1, 1);
}
""")


class Window(app.Canvas):
    def __init__(self, surface):
        app.Canvas.__init__(self, size=(400, 400), title='Render suface')
        gloo.set_state(clear_color=(0, 0, 0, 1), depth_test=False, blend=False)
        self.program = gloo.Program(vert, frag)

        self.surface = surface
        self.program["dot_position"] = self.surface.position()
        self.time = 0
        self._timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.activate_zoom()
        self.show()

    def activate_zoom(self):
        self.width, self.height = self.size
        gloo.set_viewport(0, 0, *self.physical_size)

    def on_draw(self, event):
        gloo.clear()
        self.program["dot_height"] = self.surface.height(self.time)
        self.program.draw('points')

    def on_timer(self, event):
        self.time += 0.008
        self.update()

    def on_resize(self, event):
        self.activate_zoom()


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


if __name__ == '__main__':
    appWindow = Window(Surface(size=(150, 150), waves=7))
    app.run()
