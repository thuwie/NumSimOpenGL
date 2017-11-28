shader = ("""
#version 130
uniform sampler2D u_sky_texture;
uniform sampler2D u_bed_texture;
uniform vec3 u_sun_direction;
uniform vec3 u_sun_diffused_color;
uniform vec3 u_sun_reflected_color;
uniform vec3 u_water_ambient_color;
uniform vec3 u_water_depth_color;
uniform float u_alpha;
uniform float u_bed_depth;
uniform float u_reflected_mult;
uniform float u_diffused_mult;
uniform float u_bed_mult;
uniform float u_depth_mult;
uniform float u_sky_mult;
varying vec3 v_normal;
varying vec3 v_position;
varying vec3 v_from_eye;
float reflection_refraction(in vec3 from_eye, in vec3 outer_normal,
in  float alpha, in float c1, out vec3 reflected, out vec3 refracted) {
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
 vec3 sun_contribution(vec3 direction, vec3 normal) {
    float diffused_intensity=u_diffused_mult*max(-dot(normal, u_sun_direction), 0.0);
    float cosphi=max(dot(u_sun_direction,direction), 0.0);
    float reflected_intensity=u_reflected_mult*pow(cosphi,100.0);
    return diffused_intensity*u_sun_diffused_color+reflected_intensity*u_sun_reflected_color;
}
 vec3 water_decay(vec3 color, float distance) {
    float mask=exp(-distance*u_depth_mult/2);
    return mix(u_water_ambient_color, color, mask);
}
 vec3 water_depth_mix(vec3 color, vec3 position) {
    float mask=exp(position.z*2);
    return mix(u_water_depth_color, color, mask);
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
         vec3 bed=water_decay( water_depth_mix(bed_color*u_bed_mult, v_position), path_in_water);
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