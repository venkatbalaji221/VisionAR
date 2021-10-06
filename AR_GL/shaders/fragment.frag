varying vec4 v_color;    // Interpolated fragment color (in)
varying vec3 v_position; // Interpolated vertex position (in)
void main()
{
    float xy = min( abs(v_position.x), abs(v_position.y));
    float xz = min( abs(v_position.x), abs(v_position.z));
    float yz = min( abs(v_position.y), abs(v_position.z));
    float b = 0.26;

//    if ((xy > b) || (xz > b) || (yz > b))
    if (0 > 1)
        gl_FragColor = vec4(0,0,0,1);
    else
        gl_FragColor = v_color;
}