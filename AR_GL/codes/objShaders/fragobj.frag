varying vec3 v_normal;
varying vec3 v_position;

const vec3 light_position = vec3(1.0,1.0,1.0);
const vec3 ambient_color = vec3(0.1, 0.0, 0.0);
const vec3 diffuse_color = vec3(0.75, 0.125, 0.125);
const vec3 specular_color = vec3(1.0, 1.0, 1.0);
const float shininess = 128.0;
const float gamma = 2.2;

void main()
{
    vec3 normal= normalize(v_normal);
    vec3 light_direction = normalize(light_position - v_position);
    float lambertian = max(dot(light_direction,normal), 0.0);
    float specular = 0.0;
    if (lambertian > 0.0)
    {
        vec3 view_direction = normalize(-v_position);
        vec3 half_direction = normalize(light_direction + view_direction);
        float specular_angle = max(dot(half_direction, normal), 0.0);
        specular = pow(specular_angle, shininess);
    }
    vec3 color_linear = ambient_color +
                        lambertian * diffuse_color +
                        specular * specular_color;
    vec3 color_gamma = pow(color_linear, vec3(1.0/gamma));
    gl_FragColor = vec4(color_gamma, 1.0);
}


//uniform vec4 color;
//void main()
//{
//    gl_FragColor = color;
//}
