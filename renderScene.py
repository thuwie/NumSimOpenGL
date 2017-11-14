import numpy as np
from vispy import gloo, app, io

from surface import *

VS = ("""
#version 130

uniform float u_eye_height;
uniform mat4 u_world_view;
attribute vec2 a_position;
attribute float a_height;
attribute vec2 a_normal;
varying vec3 v_normal;
varying vec3 v_position;
varying vec3 v_from_eye;
vec4 to_clipspace(vec3 position) {
    vec4 position_view=u_world_view*vec4(v_position,1);
    float z=1.0-(1.0+position_view.z)/(1.0+u_eye_height);
    return vec4(position_view.xy,-position_view.z*z/2.0,z);
}
vec3 from_water_to_eye(vec3 position) {
    vec4 eye_view=vec4(0.0,0.0,u_eye_height,1.0);
    vec4 eye=eye_view*u_world_view;
    return position-eye.xyz;
}
void main (void) {
    // aggregate input data (in world coordiantes)
    v_position=vec3(a_position.xy,a_height); // point on the water surface
    v_normal=normalize(vec3(a_normal, -1.0)); // inner normal to the water surface
    // compute position of vertices for trianges rendering
    gl_Position=to_clipspace(v_position);

    // compute reflected and refracted lights
    v_from_eye=from_water_to_eye(v_position);
}
""")

FS_triangle = ("""
#version 130
uniform sampler2D u_sky_texture;
uniform sampler2D u_bed_texture;
uniform  vec3 u_sun_direction;
uniform  vec3 u_sun_diffused_color;
uniform  vec3 u_sun_reflected_color;
uniform  vec3 u_water_ambient_color;
uniform  float u_alpha;
uniform  float u_bed_depth;
uniform  float u_reflected_mult;
uniform  float u_diffused_mult;
uniform  float u_bed_mult;
uniform  float u_depth_mult;
uniform  float u_sky_mult;
varying  vec3 v_normal;
varying  vec3 v_position;
varying  vec3 v_from_eye;
 float reflection_refraction(in  vec3 from_eye, in  vec3 outer_normal,
in  float alpha, in  float c1, out  vec3 reflected, out  vec3 refracted) {
    reflected=normalize(from_eye-2.0*outer_normal*c1);
     float k=max(0.0, 1.0-alpha*alpha*(1.0-c1*c1));
    refracted=normalize(alpha*from_eye-(alpha*c1+sqrt(k))*outer_normal);
     float c2=dot(refracted,outer_normal);
     float reflectance_s=pow((alpha*c1-c2)/(alpha*c1+c2),2.0);
     float reflectance_p=pow((alpha*c2-c1)/(alpha*c2+c1),2.0);
    return (reflectance_s+reflectance_p)/2.0;
}
 vec2 get_sky_texcoord( vec3 position,  vec3 direction) {
    return 0.05*direction.xy/direction.z+vec2(0.5,0.5);
}
 vec3 bed_intersection( vec3 position,  vec3 direction) {
     float t=(-u_bed_depth-position.z)/direction.z;
    return position+t*direction;
}
 vec2 get_bed_texcoord( vec3 point_on_bed) {
    return point_on_bed.xy+vec2(0.5,0.5);
}
 vec3 sun_contribution( vec3 direction,  vec3 normal) {
     float diffused_intensity=u_diffused_mult*max(-dot(normal, u_sun_direction), 0.0);
     float cosphi=max(dot(u_sun_direction,direction), 0.0);
     float reflected_intensity=u_reflected_mult*pow(cosphi,100.0);
    return diffused_intensity*u_sun_diffused_color+reflected_intensity*u_sun_reflected_color;
}
 vec3 water_decay( vec3 color,  float distance) {
     float mask=exp(-distance*u_depth_mult);
    return mix(u_water_ambient_color, color, mask);
}
void main() {
    // normalize directions
     vec3 normal=normalize(v_normal);
     float distance_to_eye=length(v_from_eye);
     vec3 from_eye=v_from_eye/distance_to_eye;
    // compute reflection and refraction
     float c=dot(v_normal,from_eye);
     vec3 reflected;
     vec3 refracted;
     vec2 sky_texcoord;
     vec2 bed_texcoord;
     float reflectance;
     float path_in_water;
    if(c>0.0) { // looking from air to water
        reflectance=reflection_refraction(from_eye, -normal, u_alpha, -c, reflected, refracted);
        sky_texcoord=get_sky_texcoord(v_position, reflected);
         vec3 point_on_bed=bed_intersection(v_position, refracted);
        bed_texcoord=get_bed_texcoord(point_on_bed);
        path_in_water=length(point_on_bed-v_position);
    } else { // looking from water to air
        reflectance=reflection_refraction(from_eye, normal, 1.0/u_alpha, c, reflected, refracted);
        sky_texcoord=get_sky_texcoord(v_position, refracted);
         vec3 point_on_bed=bed_intersection(v_position, reflected);
        bed_texcoord=get_bed_texcoord(point_on_bed);
        path_in_water=length(point_on_bed-v_position);
    };
    // fetch texture
     vec3 sky_color=texture2D(u_sky_texture, sky_texcoord).rgb;
     vec3 bed_color=texture2D(u_bed_texture, bed_texcoord).rgb;
    // compute colors
     vec3 rgb;
     vec3 sky=u_sky_mult*sky_color;
    if(c>0.0) { // in the air
        sky+=sun_contribution(reflected, normal);
         vec3 bed=water_decay(bed_color*u_bed_mult, path_in_water);
        rgb=mix(bed, sky, reflectance);
    } else { // under water
        sky+=sun_contribution(refracted, normal);
         vec3 bed=water_decay(bed_color*u_bed_mult, path_in_water);
        rgb=water_decay(mix(sky, bed, reflectance),distance_to_eye);
    };
    gl_FragColor.rgb = clamp(rgb,0.0,1.0);
    gl_FragColor.a = 1.0;
}
""")

FS_point = """
#version 120

void main() {
    #gl_FragColor = vec4(1,0,0,1);
}
"""


def normalize(vec):
    vec = np.asarray(vec, dtype=np.float32)
    return vec / np.sqrt(np.sum(vec * vec, axis=-1))[..., None]


class Canvas(app.Canvas):
    def __init__(self, surface, sky="fluffy_clouds.png", bed="sand.png"):
        # store parameters
        self.surface = surface
        # read textures
        self.sky = io.read_png(sky)
        self.bed = io.read_png(bed)
        # create GL context
        app.Canvas.__init__(self, size=(600, 600), title="water")
        # Compile shaders and set constants
        self.program = gloo.Program(VS, FS_triangle)
        self.program_point = gloo.Program(VS, FS_point)
        pos = self.surface.position()
        self.program["a_position"] = pos
        self.program_point["a_position"] = pos
        self.program['u_sky_texture'] = gloo.Texture2D(self.sky, wrapping='repeat', interpolation='linear')
        self.program['u_bed_texture'] = gloo.Texture2D(self.bed, wrapping='repeat', interpolation='linear')
        self.program_point["u_eye_height"] = self.program["u_eye_height"] = 3;
        self.program["u_alpha"] = 0.9;
        self.program["u_bed_depth"] = 1;
        self.program["u_sun_direction"] = normalize([0, 1, 0.1]);
        self.program["u_sun_diffused_color"] = [1, 1, 1];
        self.program["u_sun_reflected_color"] = [1, 1, 1];
        self.program["u_water_ambient_color"] = [0.0, 0.0, 0.0]
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
        self.program["u_diffused_mult"] = 0.5 if self.diffused_flag else 0;
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
    surface = Surface(size=(100, 100), waves=5, max_height=0.2)
    # surface = CircularWaves(size=(100, 100), max_height=0.01)
    c = Canvas(surface)
    c.measure_fps()
    app.run()
