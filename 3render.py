from vispy import app
from vispy import gloo

from surface import Surface

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
    gl_FragColor = vec4(0, 0.7, 1, 1);
}
""")


class Window(app.Canvas):
    def __init__(self, surface):
        self.time = 0
        app.Canvas.__init__(self, size=(400, 400), title='Render suface')
        gloo.set_state(clear_color=(0, 0, 0, 1), depth_test=False, blend=False)
        self.program = gloo.Program(vert, frag)

        self.surface = surface
        self.program["dot_position"] = self.surface.position()

        self.segments = gloo.IndexBuffer(self.surface.wireframe())

        self._timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.activate_zoom()
        self.show()

    def activate_zoom(self):
        self.width, self.height = self.size
        gloo.set_viewport(0, 0, *self.physical_size)

    def on_draw(self, event):
        gloo.clear()
        self.program["dot_height"] = self.surface.height(self.time)
        self.program.draw('lines', self.segments)

    def on_timer(self, event):
        self.time += 0.008
        self.update()

    def on_resize(self, event):
        self.activate_zoom()


if __name__ == '__main__':
    appWindow = Window(Surface(size=(150, 150), waves=7))
    app.run()
