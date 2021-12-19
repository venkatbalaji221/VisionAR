uniform mat4 model;
uniform mat4 view;
uniform mat4 perspective;
uniform mat4 m_model;
uniform mat4 m_view;
uniform mat4 m_projection;
uniform mat4 m_normal;
attribute vec3 position;
attribute vec3 normal;
varying vec3 v_normal;
varying vec3 v_position;

attribute vec2 texcoord;
varying vec2 v_texcoord;

void main()
{
    v_texcoord  = texcoord;
//    gl_Position = <transform(position)>;
//    gl_Position =  m_projection * m_view * m_model * vec4(position, 1.0);
    gl_Position = perspective * view * model * vec4(position, 1.0);
//    gl_Position = m_projection * view * model * m_model * vec4(position, 1.0);
    vec4 P = view * model * vec4(position, 1.0);
    v_position = P.xyz / P.w;
    v_normal = vec3(m_normal * vec4(normal, 0.0));
}