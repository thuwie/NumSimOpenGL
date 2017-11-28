from vispy import gloo, app, io

from surface import *
from shaders import vertex_shader as vertex_shader_file
from shaders import fragment_shader as fragment_shader_file
from shaders import point_fragment_shader as point_fragment_shader_file

vertex_shader = vertex_shader_file.shader
fragment_shader = fragment_shader_file.shader
point_fragment_shader = point_fragment_shader_file.shader


def normalize(vec):
    vec = np.asarray(vec, dtype=np.float32)
    return vec / np.sqrt(np.sum(vec * vec, axis=-1))[..., None]


class Canvas(app.Canvas):
    def __init__(self, surface, sky="textures/fluffy_clouds.png", bed="textures/seabed.png"):
        # store parameters
        self.surface = surface
        # read textures
        self.sky = io.read_png(sky)
        self.bed = io.read_png(bed)
        # create GL context
        app.Canvas.__init__(self, size=(400, 400), title="water")
        # Compile shaders and set constants
        self.program = gloo.Program(vertex_shader, fragment_shader)
        self.program_point = gloo.Program(vertex_shader, point_fragment_shader)
        pos = self.surface.position()
        self.camera_height = 1.0
        self.program["a_position"] = pos
        self.program_point["a_position"] = pos
        self.program['u_sky_texture'] = gloo.Texture2D(self.sky, wrapping='repeat', interpolation='linear')
        self.program['u_bed_texture'] = gloo.Texture2D(self.bed, wrapping='repeat', interpolation='linear')
        self.program_point['u_camera_height'] = self.program['u_camera_height'] = self.camera_height
        self.program_point["u_eye_height"] = self.program["u_eye_height"] = 1
        self.program["u_alpha"] = 0.9
        self.program["u_bed_depth"] = 0
        self.program["u_sun_direction"] = normalize([0, 1, 0.1])
        self.program["u_sun_diffused_color"] = [1, 0.8, 1]
        self.program["u_sun_reflected_color"] = [1, 0.8, 0.6]
        self.program["u_water_ambient_color"] = [0.9, 0.7, 0.7]
        self.program["u_water_depth_color"] = [0, 0.1, 0.1]
        self.triangles = gloo.IndexBuffer(self.surface.triangulation())
        # Set up GUI
        self.camera = np.array([0, 0, 1])
        self.up = np.array([0, 1, 0])
        self.set_camera()
        self.are_points_visible = False
        self.drag_start = None
        self.diffused_flag = True;
        self.reflected_flag = True;
        self.bed_flag = True;
        self.depth_flag = True;
        self.sky_flag = True;
        self.apply_flags();
        # Run everything
        self._timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.activate_zoom()
        self.show()

    def apply_flags(self):
        self.program["u_diffused_mult"] = 1.0 if self.diffused_flag else 0;
        self.program["u_reflected_mult"] = 1.0 if self.reflected_flag else 0;
        self.program["u_bed_mult"] = 1 if self.bed_flag else 0;
        self.program["u_depth_mult"] = 1 if self.depth_flag else 0;
        self.program["u_sky_mult"] = 1 if self.sky_flag else 0;

    def set_camera(self):
        rotation = np.zeros((4, 4), dtype=np.float32)
        rotation[3, 3] = 1
        rotation[0, :3] = np.cross(self.up, self.camera)
        rotation[1, :3] = self.up
        rotation[2, :3] = self.camera
        world_view = rotation
        self.program['u_world_view'] = world_view.T
        self.program_point['u_world_view'] = world_view.T

    def rotate_camera(self, shift):
        right = np.cross(self.up, self.camera)
        new_camera = self.camera - right * shift[0] + self.up * shift[1]
        new_up = self.up - self.camera * shift[0]
        self.camera = normalize(new_camera)
        self.up = normalize(new_up)
        self.up = np.cross(self.camera, np.cross(self.up, self.camera))

    def activate_zoom(self):
        self.width, self.height = self.size
        gloo.set_viewport(0, 0, *self.physical_size)

    def on_draw(self, event):
        gloo.set_state(clear_color=(0, 0, 0, 1), blend=False)
        gloo.clear()
        h, grad = self.surface.height_and_normal()
        self.program["a_height"] = h
        self.program["a_normal"] = grad
        gloo.set_state(depth_test=True)
        self.program.draw('triangles', self.triangles)
        if self.are_points_visible:
            self.program_point["a_height"] = h
            gloo.set_state(depth_test=False)
            self.program_point.draw('points')

    def on_timer(self, event):
        self.surface.propagate(0.01)
        self.update()

    def on_resize(self, event):
        self.activate_zoom()

    def on_key_press(self, event):
        if event.key == 'Escape':
            self.close()
        elif event.key == ' ':
            self.are_points_visible = not self.are_points_visible
            print("Show lattice vertices:", self.are_points_visible)
        elif event.key == '1':
            self.diffused_flag = not self.diffused_flag;
            print("Show sun diffused light:", self.diffused_flag)
            self.apply_flags();
        elif event.key == '2':
            self.bed_flag = not self.bed_flag;
            print("Show refracted image of seabed:", self.bed_flag)
            self.apply_flags();
        elif event.key == '3':
            self.depth_flag = not self.depth_flag;
            print("Show ambient light in water:", self.depth_flag)
            self.apply_flags();
        elif event.key == '4':
            self.sky_flag = not self.sky_flag;
            print("Show reflected image of sky:", self.sky_flag)
            self.apply_flags();
        elif event.key == '5':
            self.reflected_flag = not self.reflected_flag;
            print("Show reflected image of sun:", self.reflected_flag)
            self.apply_flags();
        elif event.key == 'p':
            print("path" + self.program["v_position"]);
        elif event.key == '+':
            if self.program["u_eye_height"] > 1:
                self.program["u_eye_height"] -= 0.2
                print("Camera height: ", self.program["u_eye_height"])
            else:
                print("Can't scroll closer")

        elif event.key == '-':
            self.program["u_eye_height"] += 0.2
            print("Camera height: ", self.program["u_eye_height"])

    def screen_to_gl_coordinates(self, pos):
        return 2 * np.array(pos) / np.array(self.size) - 1

    def on_mouse_press(self, event):
        self.drag_start = self.screen_to_gl_coordinates(event.pos)

    def on_mouse_move(self, event):
        if not self.drag_start is None:
            pos = self.screen_to_gl_coordinates(event.pos)
            self.rotate_camera(pos - self.drag_start)
            self.drag_start = pos
            self.set_camera()
            self.update()

    def on_mouse_release(self, event):
        self.drag_start = None


if __name__ == '__main__':
    # surface=Surface(size=(100,100), waves=5, max_height=0.2)
    surface = CircularWaves(size=(100, 100), max_height=0.1)
    c = Canvas(surface)
    c.measure_fps()
    app.run()
