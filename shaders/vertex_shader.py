shader = ("""

uniform float u_eye_height;
uniform mat4 u_world_view;

attribute vec2 a_position;
attribute float a_height;
attribute float u_camera_height;
attribute vec2 a_normal;

varying vec3 v_normal;
varying vec3 v_position;
varying vec3 v_from_eye;
vec4 to_clipspace(vec3 position) {
    vec4 position_view=u_world_view*vec4(v_position,1);
    float z=u_eye_height-(1.0+position_view.z)/(1.0+u_eye_height);
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
