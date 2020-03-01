#version 450

layout(location=0) in vec3 a_position;
layout(location=1) in vec2 a_tex_coords;
layout(location=2) in vec3 a_normal;

layout(location=0) out vec2 v_tex_coords;
layout(location=1) out vec3 v_normal;

layout(set=1, binding=0)
uniform Uniforms {
	mat4 u_view_proj;
};

void main() {
    v_tex_coords = a_tex_coords;
    v_normal = mat3(u_view_proj) * a_normal;
    gl_Position = u_view_proj * vec4(a_position, 1.0);
}