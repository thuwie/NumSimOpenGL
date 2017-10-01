import numpy as np
from vispy import app
from vispy import gloo

vert = ("""
attribute vec2 dot_position;
 void main(){
    gl_Position = vec4(dot_position.xy, 1, 1);
 }
""")

frag = ("""
void main(){
    gl_FragColor = vec4(0, 0.4, 1, 1);
}
""")

class Window(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, title='Render dot', size=(350, 350))
        gloo.set_state(clear_color=(0, 0, 0, 1), depth_test=False, blend=False)
        self.program = gloo.Program(vert, frag)
        self.program['dot_position'] = np.array([[0, 0]], dtype=np.float32)
        self.width, self.height = self.size
        gloo.set_viewport(0, 0, *self.physical_size)
        self.show()

    def on_draw(self, event):
        gloo.clear()
        self.program.draw('points')


if __name__ == '__main__':
    appWindow = Window()
    app.run()
