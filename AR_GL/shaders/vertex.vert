uniform mat4   model;         // Model matrix
uniform mat4   view;          // View matrix
uniform mat4   projection;    // Projection matrix
uniform mat4 perspective;
attribute vec4 color;         // Vertex color
attribute vec3 position;      // Vertex position
varying vec3   v_position;    // Interpolated vertex position (out)
varying vec4   v_color;       // Interpolated fragment color (out)
void main()
{
    v_color = color;
    v_position = position;
//    gl_Position = projection * view * model * vec4(position,1.0);
    gl_Position = perspective * view  * model * vec4(position,1.0);
//    gl_Position = projection * view * model * vec4(position,1.0);

}