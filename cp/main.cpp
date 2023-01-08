// вариант 5 Тетраэдр, Октаэдр, Икосаэдр
#include <stdlib.h>
#include <stdio.h> 
#include <math.h>
#include <string>
#include <string.h>
#include <fstream>
#include <vector>
#include <limits>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <chrono>

const float CPU_INTENCITY_THRESHOLD = 1e-4;
uint CPU_MAX_BOUNCES = 1;
const float CPU_RAY_OFFSET = 1e-4;

unsigned long long cpu_rays_count = 0;

struct cpu_uchar4{
    u_char x; 
    u_char y; 
    u_char z; 
    u_char w; 
};

struct cpu_vec3 {
	float x;
	float y;
	float z;
};


class cpu_matr3{
    public:
        float elems[3][3];

        cpu_matr3(cpu_vec3 a, cpu_vec3 b, cpu_vec3 c){
            elems[0][0] = a.x;
            elems[0][1] = b.x;
            elems[0][2] = c.x;

            elems[1][0] = a.y;
            elems[1][1] = b.y;
            elems[1][2] = c.y;

            elems[2][0] = a.z;
            elems[2][1] = b.z;
            elems[2][2] = c.z;
        }
};

// операции над векторами
    float dot(cpu_vec3 a, cpu_vec3 b) {
        return a.x*b.x+a.y*b.y+a.z*b.z;
    }

    cpu_vec3 prod(cpu_vec3 b, cpu_vec3 a) {
        return {a.y*b.z-a.z*b.y,
                a.z*b.x-a.x*b.z,
                a.x*b.y-a.y*b.x};
    }

    cpu_vec3 norm(cpu_vec3 v) {
        float l = sqrt(dot(v, v));
        return {v.x/l, v.y/l, v.z/l};
    }

    float length(cpu_vec3 v){
        return sqrt(dot(v, v));
    }

    cpu_vec3 diff(cpu_vec3 a, cpu_vec3 b) {
        return {a.x-b.x, a.y-b.y, a.z-b.z};
    }

    cpu_vec3 add(cpu_vec3 a, cpu_vec3 b) {
        return {a.x+b.x, a.y+b.y, a.z+b.z};
    }

    cpu_vec3 mult(cpu_matr3 matr, cpu_vec3 v) {
        return cpu_vec3{matr.elems[0][0]*v.x+matr.elems[0][1]*v.y+matr.elems[0][2]*v.z,
                    matr.elems[1][0]*v.x+matr.elems[1][1]*v.y+matr.elems[1][2]*v.z,
                    matr.elems[2][0]*v.x+matr.elems[2][1]*v.y+matr.elems[2][2]*v.z};
    }

    cpu_vec3 mult(float a, cpu_vec3 v){
        return cpu_vec3{a*v.x, a*v.y, a*v.z};
    }

    void print(cpu_vec3 v) {
        printf("%e %e %e\n", v.x, v.y, v.z);
    }

struct cpu_Material{
    cpu_vec3 color;
    float phong;
    float reflective;
    float transparency;
};

const float CPU_INF = std::numeric_limits<float>::max();
const cpu_vec3 CPU_BACKGOUND_COLOR{0.1, 0.1, 0.1};
const cpu_vec3 CPU_BACKGOUND_LIGHT_COLOR{0.0, 0.0, 0.0};

class cpu_Polygon{
    public:
        cpu_vec3 a;
        cpu_vec3 b;
        cpu_vec3 c;

        cpu_Polygon(cpu_vec3 init_a, cpu_vec3 init_b, cpu_vec3 init_c){
            a = init_a;
            b = init_b;
            c = init_c;
        }

        cpu_vec3 Normal(){
            return norm(prod(diff(b,a), diff(c,a)));
        }
};

struct cpu_PointLight{
    cpu_vec3 p;
    cpu_vec3 color;
    float intencity;
};

class cpu_Object{
    public:
        cpu_vec3 p;
        float scale;

        // адреса указывают на видеопамять
        cpu_Polygon* faces;
        uint faces_number;
        cpu_Material face_cpu_material;
        cpu_Polygon* bevels;
        uint bevels_number;
        cpu_Material bevel_cpu_material;
        cpu_Polygon* caps;
        uint caps_number;

        cpu_vec3* dev_lights;
        uint lights_number;
        float lights_radius;
        cpu_Material lights_cpu_material;

        void Import_obj(cpu_vec3 p, float scale, cpu_Polygon*& out_cpu_polygons, uint& out_cpu_polygons_number, std::string path){
            std::string word;
            std::vector<cpu_vec3> verts;
            std::vector<cpu_Polygon> cpu_polygons;
            std::ifstream faces_file;
            faces_file.open(path);
            while (faces_file >> word){
                if (word == "v"){
                    float a, b, c;
                    faces_file >> a >> b >> c;
                    verts.push_back(cpu_vec3{a, b, c});
                }

                if (word == "f"){
                    uint id1, id2, id3;
                    faces_file >> id1 >> id2 >> id3;
                    --id1;
                    --id2;
                    --id3;

                    cpu_Polygon cur_cpu_polygon = cpu_Polygon(add(p, mult(scale, cpu_vec3{verts[id1].x, verts[id1].y, verts[id1].z})),
                                                  add(p, mult(scale, cpu_vec3{verts[id2].x, verts[id2].y, verts[id2].z})),
                                                  add(p, mult(scale, cpu_vec3{verts[id3].x, verts[id3].y, verts[id3].z})));
                    cpu_polygons.push_back(cur_cpu_polygon);
                }   
            }
            out_cpu_polygons_number = cpu_polygons.size();
            out_cpu_polygons = (cpu_Polygon*)malloc(out_cpu_polygons_number*sizeof(cpu_Polygon));
            std::copy(cpu_polygons.begin(), cpu_polygons.end(), out_cpu_polygons);
            cpu_polygons.clear();

            cpu_Polygon* dev_out_cpu_polygons;
            dev_out_cpu_polygons = (cpu_Polygon*)malloc(out_cpu_polygons_number*sizeof(cpu_Polygon));
            memcpy(dev_out_cpu_polygons, out_cpu_polygons, out_cpu_polygons_number*sizeof(cpu_Polygon));
            free(out_cpu_polygons);
            out_cpu_polygons = dev_out_cpu_polygons;
        }

        void Import_obj_lights(cpu_vec3 p, float scale, uint bevel_lights_number, std::string path){
            std::string word;
            std::vector<cpu_vec3> verts;
            std::vector<std::pair<cpu_vec3, cpu_vec3>> verts_pairs;
            std::ifstream faces_file;
            faces_file.open(path);
            while (faces_file >> word){
                if (word == "v"){
                    float a, b, c;
                    faces_file >> a >> b >> c;
                    verts.push_back(cpu_vec3{a, b, c});
                }

                if (word == "l"){
                    uint id1, id2;
                    faces_file >> id1 >> id2;
                    --id1;
                    --id2;

                    std::pair<cpu_vec3, cpu_vec3> cur_pair(add(p, mult(scale, cpu_vec3{verts[id1].x, verts[id1].y, verts[id1].z})), 
                                                   add(p, mult(scale, cpu_vec3{verts[id2].x, verts[id2].y, verts[id2].z})));
                    verts_pairs.push_back(cur_pair);
                }
            }

            lights_number = bevel_lights_number*verts_pairs.size();
            cpu_vec3* lights = (cpu_vec3*)malloc(lights_number*sizeof(cpu_vec3));

            for (uint line_id = 0; line_id < verts_pairs.size(); ++line_id){
                cpu_vec3 step = diff(verts_pairs[line_id].second, verts_pairs[line_id].first);
                step = mult(1.0/(bevel_lights_number+1.0), step);

                for (int i = 0; i < bevel_lights_number; ++i){
                    lights[line_id*bevel_lights_number+i] = add(verts_pairs[line_id].first, mult(i+1.0, step));
                }
            }

            dev_lights = (cpu_vec3*)malloc(lights_number*sizeof(cpu_vec3));
            memcpy(dev_lights, lights, lights_number*sizeof(cpu_vec3));
            free(lights);
        }

        cpu_Object(cpu_vec3 init_p, float init_scale, std::string dir_path, cpu_Material init_face_cpu_material, cpu_Material init_bevel_cpu_material, cpu_Material init_lights_cpu_material, uint bevel_lights_number, float init_lights_radius){
            p = init_p;
            scale = init_scale;
            
            Import_obj(init_p, scale, faces, faces_number, dir_path+"/faces.obj");
            Import_obj(init_p, scale, bevels, bevels_number, dir_path+"/bevels.obj");
            Import_obj(init_p, scale, caps, caps_number, dir_path+"/caps.obj");
            face_cpu_material = init_face_cpu_material;
            bevel_cpu_material = init_bevel_cpu_material;

            Import_obj_lights(init_p, scale, bevel_lights_number, dir_path+"/bevel_lines.obj");
            lights_cpu_material = init_lights_cpu_material;
            lights_radius = init_lights_radius;
        }

        cpu_Object() = default;
};

class cpu_Textured_floor{
    public:
        cpu_vec3 a;
        cpu_vec3 b;
        cpu_vec3 c;
        cpu_vec3 d;
        cpu_Material floor_cpu_material;
        uint tex_w;
        uint tex_h;

        // ссылка на данные текстуры
        cpu_uchar4* tex_data;
        cpu_Polygon* cpu_polygons;

        cpu_Textured_floor(cpu_vec3 init_a, cpu_vec3 init_b, cpu_vec3 init_c, cpu_vec3 init_d, std::string image_path, cpu_Material init_floor_cpu_material){
            a = init_a;
            b = init_b;
            c = init_c;
            d = init_d;
            floor_cpu_material = init_floor_cpu_material;

            cpu_polygons = (cpu_Polygon*)malloc(2*sizeof(cpu_Polygon));
            cpu_polygons[0] = cpu_Polygon(init_a, init_c, init_b);
            cpu_polygons[1] = cpu_Polygon(init_d, init_b, init_c);

            // загрузка текстуры
                uint w, h;
                FILE* fp = fopen(image_path.c_str(), "rb");
                fread(&w, sizeof(uint), 1, fp);
                fread(&h, sizeof(uint), 1, fp);

                tex_w = w;
                tex_h = h;

                tex_data = (cpu_uchar4*)malloc(sizeof(cpu_uchar4) * w * h);
                
                for (uint y = 0; y < tex_h; ++y){
                    for (uint x = 0; x < tex_w; ++x){
                        fread(&tex_data[y*tex_w+x].x, sizeof(u_char), 1, fp);
                        fread(&tex_data[y*tex_w+x].y, sizeof(u_char), 1, fp);
                        fread(&tex_data[y*tex_w+x].z, sizeof(u_char), 1, fp);
                        fread(&tex_data[y*tex_w+x].w, sizeof(u_char), 1, fp);
                    }
                }
                fclose(fp);
        }

        cpu_vec3 Get_color(float u, float v, uint cpu_polygon_id){
            uint x = (uint)(u*(tex_w-1));
            uint y = (uint)(v*(tex_h-1));
            cpu_vec3 res;
            if (cpu_polygon_id == 0){
                cpu_uchar4 p = tex_data[y*tex_w+x];
                res = cpu_vec3{p.x/255.0f, p.y/255.0f, p.z/255.0f};
            }
            else if (cpu_polygon_id == 1){
                cpu_uchar4 p = tex_data[(tex_h-y)*tex_w+(tex_w-x)];
                res = cpu_vec3{p.x/255.0f, p.y/255.0f, p.z/255.0f};
            }

            return res;
        }
};

cpu_Object* cpu_scene_objects;
uint cpu_scene_objects_number;
cpu_PointLight* cpu_scene_lights;
uint cpu_scene_lights_number;

cpu_Textured_floor* cpu_scene_floor;

class cpu_Ray{
    public:
        cpu_vec3 p, d;
        uint parent;
        uint bounses;
        float intencity;
        cpu_vec3 color;

        cpu_Ray(cpu_vec3 init_p, cpu_vec3 init_d, uint init_bounces, float init_intencity){
            p = init_p;
            d = init_d;
            bounses = init_bounces;
            intencity = init_intencity;
            color = cpu_vec3{-1.0, -1.0, -1.0};
        }
};

float is_cpu_ray_hits_cpu_polygon(cpu_Ray cpu_ray, cpu_Polygon cpu_polygon){
    cpu_vec3 e1 = diff(cpu_polygon.b, cpu_polygon.a);
    cpu_vec3 e2 = diff(cpu_polygon.c, cpu_polygon.a);

    cpu_vec3 p = prod(cpu_ray.d, e2);
    double div = dot(p, e1);
    if (fabs(div) < 1e-10)
        return CPU_INF;
    cpu_vec3 t = diff(cpu_ray.p, cpu_polygon.a);
    double u = dot(p, t)/div;
    if (u<0.0 || u>1.0)
        return CPU_INF;
    cpu_vec3 q = prod(t, e1);
    double v = dot(q, cpu_ray.d)/div;
    if (v<0.0 || v+u>1.0)
        return CPU_INF;
    double ts = dot(q, e2)/div; 
    if (ts < 0.0)
        return CPU_INF;

    return ts;
}

float is_floor_cpu_ray_hits_cpu_polygon(float& x, float& y, cpu_Ray cpu_ray, cpu_Polygon cpu_polygon){
    cpu_vec3 e1 = diff(cpu_polygon.b, cpu_polygon.a);
    cpu_vec3 e2 = diff(cpu_polygon.c, cpu_polygon.a);

    cpu_vec3 p = prod(cpu_ray.d, e2);
    double div = dot(p, e1);
    if (fabs(div) < 1e-10)
        return CPU_INF;
    cpu_vec3 t = diff(cpu_ray.p, cpu_polygon.a);
    double u = dot(p, t)/div;
    if (u<0.0 || u>1.0)
        return CPU_INF;
    cpu_vec3 q = prod(t, e1);
    double v = dot(q, cpu_ray.d)/div;
    if (v<0.0 || v+u>1.0)
        return CPU_INF;
    double ts = dot(q, e2)/div; 
    if (ts < 0.0)
        return CPU_INF;

    x = u;
    y = v;

    return ts;
}

cpu_vec3 trace(cpu_Ray cpu_ray, cpu_Object* cpu_objects, uint cpu_objects_number, cpu_PointLight* lights, uint lights_number, cpu_Textured_floor* floor) {
    ++cpu_rays_count;
    
    float min_t = CPU_INF;
    cpu_Material hit_cpu_material;
    cpu_vec3 hit_normal;
    bool hit_bevel = false;
    bool no_shadow = false;
    uint hit_cpu_object_id;

    for (uint cpu_object_id = 0; cpu_object_id < cpu_objects_number; ++cpu_object_id){
        // грани
        for (uint cpu_polygon_id = 0; cpu_polygon_id < cpu_objects[cpu_object_id].faces_number; ++cpu_polygon_id){
            float t = is_cpu_ray_hits_cpu_polygon(cpu_ray, cpu_objects[cpu_object_id].faces[cpu_polygon_id]);
            if (min_t > t){
                min_t = t;
                hit_cpu_material = cpu_objects[cpu_object_id].face_cpu_material;
                hit_normal = cpu_objects[cpu_object_id].faces[cpu_polygon_id].Normal();
                hit_bevel = false;
            }
        }
        // ребра
        for (uint cpu_polygon_id = 0; cpu_polygon_id < cpu_objects[cpu_object_id].bevels_number; ++cpu_polygon_id){
            float t = is_cpu_ray_hits_cpu_polygon(cpu_ray, cpu_objects[cpu_object_id].bevels[cpu_polygon_id]);
            if (min_t > t){

                min_t = t;
                hit_cpu_material = cpu_objects[cpu_object_id].bevel_cpu_material;
                hit_normal = cpu_objects[cpu_object_id].bevels[cpu_polygon_id].Normal();
                hit_bevel = true;
                hit_cpu_object_id = cpu_object_id;
            }
        }
        // крышки
        for (uint cpu_polygon_id = 0; cpu_polygon_id < cpu_objects[cpu_object_id].caps_number; ++cpu_polygon_id){
            float t = is_cpu_ray_hits_cpu_polygon(cpu_ray, cpu_objects[cpu_object_id].caps[cpu_polygon_id]);
            if (min_t > t){
                min_t = t;
                hit_cpu_material = cpu_objects[cpu_object_id].bevel_cpu_material;
                hit_normal = cpu_objects[cpu_object_id].caps[cpu_polygon_id].Normal();
                hit_bevel = false;
            }
        }
    }

    cpu_vec3 hit_color;
    cpu_vec3 hit_p = add(cpu_ray.p, mult(min_t, cpu_ray.d));

    bool hit_floor = false;
    if (min_t == CPU_INF){

        // столкновение с полом
        uint hit_cpu_polygon_id;
        float u, v;
        for (uint cpu_polygon_id = 0; cpu_polygon_id < 2; ++cpu_polygon_id){
            float t = is_floor_cpu_ray_hits_cpu_polygon(u, v, cpu_ray, floor[0].cpu_polygons[cpu_polygon_id]);

            if (min_t > t){
                min_t = t;
                hit_cpu_material = floor[0].floor_cpu_material;
                hit_normal = floor[0].cpu_polygons[cpu_polygon_id].Normal();
                hit_cpu_polygon_id = cpu_polygon_id;
            }
        }

        // если попал
        if (min_t != CPU_INF){
            hit_floor = true;
            // hit_color = floor[0].Get_color(u, v, hit_cpu_polygon_id);
            hit_color = floor[0].Get_color(u, v, hit_cpu_polygon_id);
        }
    }
    else if (hit_bevel){
        hit_color = hit_cpu_material.color;
        // printf("%d\n", hit_cpu_object_id);
        for (uint light_id = 0; light_id < cpu_objects[hit_cpu_object_id].lights_number; ++light_id){
            if (dot(hit_normal, cpu_ray.d) > 0.0)
                hit_normal = mult(-1.0, hit_normal);
            if (dot(hit_normal, diff(hit_p, cpu_objects[hit_cpu_object_id].p)) < 0.0
                && cpu_objects[hit_cpu_object_id].lights_radius > length(diff(hit_p, cpu_objects[hit_cpu_object_id].dev_lights[light_id]))){
                hit_color = cpu_objects[hit_cpu_object_id].lights_cpu_material.color;    
                hit_cpu_material = cpu_objects[hit_cpu_object_id].lights_cpu_material;
                no_shadow = true;
            }
        }
    }
    else {
        hit_color = hit_cpu_material.color;
    }

    // двусторонние полигоны
    if (dot(hit_normal, cpu_ray.d) > 0.0)
        hit_normal = mult(-1.0, hit_normal);
    if (min_t == CPU_INF){
        cpu_ray.color = CPU_BACKGOUND_COLOR;
    }
    else {
        cpu_vec3 p = add(cpu_ray.p, mult(min_t, cpu_ray.d));
        for (uint light_id = 0; light_id < lights_number; ++light_id){
                cpu_vec3 l = diff(lights[light_id].p, p);
                
                // теневой луч
                    cpu_vec3 shadow_d = norm(l);
                    cpu_Ray shadow_cpu_ray = cpu_Ray(add(p, mult(-CPU_RAY_OFFSET, cpu_ray.d)), shadow_d, 0, 1.0);
                    cpu_vec3 shadow_color{1.0, 1.0, 1.0};
                    bool full_shadow = false;

                    if (!no_shadow){
                        for (uint cpu_object_id = 0; cpu_object_id < cpu_objects_number; ++cpu_object_id){
                            // грани
                            for (uint cpu_polygon_id = 0; cpu_polygon_id < cpu_objects[cpu_object_id].faces_number; ++cpu_polygon_id){
                                float t = is_cpu_ray_hits_cpu_polygon(shadow_cpu_ray, cpu_objects[cpu_object_id].faces[cpu_polygon_id]);
                                if (t != CPU_INF){
                                    cpu_Material shadow_hit_cpu_material = cpu_objects[cpu_object_id].face_cpu_material;

                                    if (shadow_hit_cpu_material.transparency > 0.0){
                                        cpu_vec3 subtractive_color{1.0f - shadow_hit_cpu_material.color.x,
                                                        1.0f - shadow_hit_cpu_material.color.y,
                                                        1.0f - shadow_hit_cpu_material.color.z};
                                        subtractive_color = mult((1.0f-shadow_hit_cpu_material.transparency), subtractive_color);
                                        shadow_color = diff(shadow_color, subtractive_color);
                                    }
                                    else{
                                        full_shadow = true;
                                        break;
                                    }
                                }
                            }
                            // ребра
                            for (uint cpu_polygon_id = 0; cpu_polygon_id < cpu_objects[cpu_object_id].bevels_number; ++cpu_polygon_id){
                                float t = is_cpu_ray_hits_cpu_polygon(shadow_cpu_ray, cpu_objects[cpu_object_id].bevels[cpu_polygon_id]);
                                if (t != CPU_INF){
                                    cpu_Material shadow_hit_cpu_material = cpu_objects[cpu_object_id].bevel_cpu_material;

                                    if (shadow_hit_cpu_material.transparency > 0.0){
                                        cpu_vec3 subtractive_color{1.0f - shadow_hit_cpu_material.color.x,
                                                            1.0f - shadow_hit_cpu_material.color.y,
                                                            1.0f - shadow_hit_cpu_material.color.z};
                                        subtractive_color = mult((1.0f-shadow_hit_cpu_material.transparency), subtractive_color);
                                        shadow_color = diff(shadow_color, subtractive_color);
                                    }
                                    else{
                                        full_shadow = true;
                                        break;
                                    }
                                }
                            }
                            // крышки
                            for (uint cpu_polygon_id = 0; cpu_polygon_id < cpu_objects[cpu_object_id].caps_number; ++cpu_polygon_id){
                                float t = is_cpu_ray_hits_cpu_polygon(shadow_cpu_ray, cpu_objects[cpu_object_id].caps[cpu_polygon_id]);
                                if (t != CPU_INF){
                                    cpu_Material shadow_hit_cpu_material = cpu_objects[cpu_object_id].bevel_cpu_material;

                                    if (shadow_hit_cpu_material.transparency > 0.0){
                                        cpu_vec3 subtractive_color{1.0f - shadow_hit_cpu_material.color.x,
                                                        1.0f - shadow_hit_cpu_material.color.y,
                                                        1.0f - shadow_hit_cpu_material.color.z};
                                        subtractive_color = mult((1.0f-shadow_hit_cpu_material.transparency), subtractive_color);
                                        shadow_color = diff(shadow_color, subtractive_color);
                                    }
                                    else{
                                        full_shadow = true;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    

                
                // затенение Фонга
                cpu_ray.color = cpu_vec3{0.0, 0.0, 0.0};
                if (!full_shadow){
                    float n_l_cos = dot(hit_normal, norm(l));
                    if (n_l_cos < 0.0)
                        n_l_cos = 0.0;
                    cpu_ray.color = mult(n_l_cos, hit_color);

                    cpu_vec3 r = diff(mult(2*dot(hit_normal, norm(l)), hit_normal), norm(l));
                    float v_r_cos = dot(mult(-1.0, cpu_ray.d), norm(r));
                    if (v_r_cos < 0){
                        v_r_cos = 0.0;
                    }
                    float specular_sharpness = 8.0;
                    cpu_ray.color = add(cpu_ray.color, mult(pow(v_r_cos, specular_sharpness), lights[light_id].color));

                    cpu_ray.color.x *= shadow_color.x;
                    cpu_ray.color.y *= shadow_color.y;
                    cpu_ray.color.z *= shadow_color.z;
                }
                cpu_ray.color = add(cpu_ray.color, mult(0.1, hit_color));
                cpu_ray.color = add(cpu_ray.color, CPU_BACKGOUND_LIGHT_COLOR);
            }
        if (cpu_ray.bounses != 0 and cpu_ray.intencity > CPU_INTENCITY_THRESHOLD){
            // уменьшить интесивность Фонга
                cpu_ray.color = mult(hit_cpu_material.phong, cpu_ray.color);
            // отраженный
            if (cpu_ray.intencity*hit_cpu_material.reflective > CPU_INTENCITY_THRESHOLD){
                cpu_vec3 r_dir = mult(-2.0*dot(cpu_ray.d, hit_normal), hit_normal);
                r_dir = add(r_dir, cpu_ray.d);

                cpu_Ray new_cpu_ray = cpu_Ray(add(p, mult(-CPU_RAY_OFFSET, cpu_ray.d)), r_dir, cpu_ray.bounses-1, cpu_ray.intencity*hit_cpu_material.reflective);
                cpu_vec3 reflected_color = trace(new_cpu_ray, cpu_objects, cpu_objects_number, lights, lights_number, floor);

                cpu_ray.color = add(cpu_ray.color, mult(new_cpu_ray.intencity, reflected_color));
            }
            // сквозной
            if (cpu_ray.intencity*hit_cpu_material.transparency > CPU_INTENCITY_THRESHOLD){
                cpu_Ray new_cpu_ray = cpu_Ray(add(p, mult(CPU_RAY_OFFSET, cpu_ray.d)), cpu_ray.d, cpu_ray.bounses-1, cpu_ray.intencity*hit_cpu_material.transparency);
                cpu_vec3 transparent_color = trace(new_cpu_ray, cpu_objects, cpu_objects_number, lights, lights_number, floor);
                cpu_ray.color = add(cpu_ray.color, mult(new_cpu_ray.intencity, transparent_color));
            }
        }
    }
    return cpu_ray.color;
}

class cpu_Camera{
    public:
        cpu_vec3 p, d;
        uint w, h;
        float fov;
        cpu_uchar4* data;

        cpu_Camera(cpu_vec3 init_p, cpu_vec3 init_d, uint init_w, uint init_h, float init_fov){
            p = init_p;
            d = init_d;
            w = init_w;
            h = init_h;
            fov = init_fov;
        }

        void Render(uint bounces){
            cpu_Ray* cpu_rays;
            cpu_rays = (cpu_Ray*)malloc(w*h*sizeof(cpu_Ray));

            float dw = 2.0/(w-1.0);
            float dh = 2.0/(h-1.0);
            float z_side = 1.0/tan(fov*(M_PI/360.0));

            cpu_vec3 basis_z = norm(diff(d, p));
            cpu_vec3 basis_x = norm(prod(basis_z, cpu_vec3{0.0, 0.0, 1.0}));
            cpu_vec3 basis_y = norm(prod(basis_x, basis_z));
            data = (cpu_uchar4*)malloc(w*h*sizeof(cpu_uchar4));
            cpu_rays_count = 0;
            for (int y = 0; y < h; ++y){
                for (int x = 0; x < w; ++x){
                    cpu_vec3 v = cpu_vec3{-1.0f+dw*x, -1.0f+dh*y* h / w, z_side};
                    cpu_matr3 matr(basis_x, basis_y, basis_z);
                    cpu_vec3 dir = norm(mult(matr, v));

                    unsigned long long cur_cpu_rays_count = 0;
                    cpu_vec3 traced_color = trace(cpu_Ray(p, dir, bounces, 1.0), cpu_scene_objects, cpu_scene_objects_number, cpu_scene_lights, cpu_scene_lights_number, cpu_scene_floor);

                    data[(h-1-y)*w+x].w = 255;
                    data[(h-1-y)*w+x].x = (uint)(std::clamp(traced_color.x, 0.0f, 1.0f)*255);
                    data[(h-1-y)*w+x].y = (uint)(std::clamp(traced_color.y, 0.0f, 1.0f)*255);
                    data[(h-1-y)*w+x].z = (uint)(std::clamp(traced_color.z, 0.0f, 1.0f)*255);
                }
            }
        }
};

int main(int argc, char **argv) {
    bool default_mode = true;
    bool gpu = true;

    if (argc == 1 or (argc == 2 and strcmp(argv[1], "--gpu")) == 0){
        default_mode = false;
    }
    if (argc == 2 and strcmp(argv[1], "--cpu") == 0){
        gpu = false;
        default_mode = false;
    }

    uint frame_number;
    std::string save_path;
    std::string pref_save_path;
    std::string suff_save_path;
    uint w, h;
    float fow;

    float r_0_c, z_0_c, f_0_c;
    float a_r_c, a_z_c;
    float w_r_c, w_z_c, w_f_c;
    float p_r_c, p_z_c;

    float r_0_n, z_0_n, f_0_n;
    float a_r_n, a_z_n;
    float w_r_n, w_z_n, w_f_n;
    float p_r_n, p_z_n;

    cpu_vec3 icosahedron_center;
    cpu_vec3 icosahedron_color;
    float icosahedron_size;
    float icosahedron_reflective, icosahedron_transparency;
    uint icosahedron_bevel_lights_number;

    cpu_vec3 octahedron_center;
    cpu_vec3 octahedron_color;
    float octahedron_size;
    float octahedron_reflective, octahedron_transparency;
    uint octahedron_bevel_lights_number;

    cpu_vec3 tetrahedron_center;
    cpu_vec3 tetrahedron_color;
    float tetrahedron_size;
    float tetrahedron_reflective, tetrahedron_transparency;
    uint tetrahedron_bevel_lights_number;

    cpu_vec3 floor_a, floor_b, floor_c, floor_d;

    std::string floor_texture_path;
        
    cpu_vec3 floor_color;
    float floor_reflective;

    uint max_bounces;    
    uint ssaa_scale;

    if (default_mode){
        frame_number = 126;
        save_path = "res/%d.data";
        pref_save_path = "res/";
        suff_save_path = ".data";

        w = 200;
        h = 300;
        fow = 30.0;

        r_0_c = 7.0; z_0_c = 3.0; f_0_c = 0.0;
        a_r_c = 2.0; a_z_c = 1.0;
        w_r_c = 2.0; w_z_c = 6.0; w_f_c = 1.0;
        p_r_c = 0.0; p_z_c = 0.0;

        r_0_n = 2.0; z_0_n = 0.0; f_0_n = 0.0;
        a_r_n = 0.5; a_z_n = 0.1;
        w_r_n = 1.0; w_z_n = 4.0; w_f_n = 1.0;
        p_r_n = 0.0; p_z_n = 0.0;

        // объекты
            icosahedron_center = cpu_vec3{0.0, 0.0, 0.0};
            icosahedron_color = cpu_vec3{1.0, 0.0, 0.0};
            icosahedron_size = 1.0;
            icosahedron_reflective = 0.3;
            icosahedron_transparency = 0.6;
            icosahedron_bevel_lights_number = 3;

            octahedron_center = cpu_vec3{-2.0, 0.0, 0.0};
            octahedron_color = cpu_vec3{0.0, 1.0, 0.0};
            octahedron_size = 1.0;
            octahedron_reflective = 0.3;
            octahedron_transparency = 0.6;
            octahedron_bevel_lights_number = 4;

            tetrahedron_center = cpu_vec3{2.0, 0.0, 0.0};
            tetrahedron_color = cpu_vec3{0.0, 0.0, 1.0};
            tetrahedron_size = 1.0;
            tetrahedron_reflective = 0.3;
            tetrahedron_transparency = 0.6;
            tetrahedron_bevel_lights_number = 5;

        // пол
            floor_a = cpu_vec3{-8.0, -8.0, -2.0};
            floor_b = cpu_vec3{-8.0, 8.0, -2.0};
            floor_c = cpu_vec3{8.0, -8.0, -2.0};
            floor_d = cpu_vec3{8.0, 8.0, -2.0};
            floor_texture_path = "images/shrek.data";
            floor_color = cpu_vec3{1.0, 1.0, 1.0};
            floor_reflective = 0.3;
        
        // свет
            cpu_scene_lights_number = 2;
            cpu_scene_lights = (cpu_PointLight*)malloc(cpu_scene_lights_number*sizeof(cpu_PointLight));

            cpu_scene_lights[0] = cpu_PointLight{cpu_vec3{3.0, 3.0, 3.0}, cpu_vec3{1.0, 1.0, 1.0}, 3.0};
            cpu_scene_lights[1] = cpu_PointLight{cpu_vec3{3.0, -3.0, 3.0}, cpu_vec3{0.3, 1.0, 0.3}, 0.4};

        max_bounces = 5;
        ssaa_scale = 1;
    }
    else {
        // ввод данных
        std::cin >> frame_number;
        std::cin >> save_path;
        for (uint i = 0; i < save_path.length(); ++i){
            if (save_path[i] == '%'){
                pref_save_path = save_path.substr(0, i);
                suff_save_path = save_path.substr(i+2, save_path.length()-(i+2));
            }
        }

        std::cin >> w >> h;
        std::cin >> fow;

        std::cin >> r_0_c >> z_0_c >> f_0_c;
        std::cin >> a_r_c, a_z_c;
        std::cin >> w_r_c >> w_z_c >> w_f_c;
        std::cin >> p_r_c >> p_z_c;

        std::cin >> r_0_n >> z_0_n >> f_0_n;
        std::cin >> a_r_n, a_z_n;
        std::cin >> w_r_n >> w_z_n >> w_f_n;
        std::cin >> p_r_n >> p_z_n;

        std::cin >> icosahedron_center.x >> icosahedron_center.y >> icosahedron_center.z;
        std::cin >> icosahedron_color.x >> icosahedron_color.y >> icosahedron_color.z;\
        std::cin >> icosahedron_size;
        std::cin >> icosahedron_reflective >> icosahedron_transparency;
        std::cin >> icosahedron_bevel_lights_number;

        std::cin >> octahedron_center.x >> octahedron_center.y >> octahedron_center.z;
        std::cin >> octahedron_color.x >> octahedron_color.y >> octahedron_color.z;
        std::cin >> octahedron_size;
        std::cin >> octahedron_reflective >> octahedron_transparency;
        std::cin >> octahedron_bevel_lights_number;

        std::cin >> tetrahedron_center.x >> tetrahedron_center.y >> tetrahedron_center.z;
        std::cin >> tetrahedron_color.x >> tetrahedron_color.y >> tetrahedron_color.z;
        std::cin >> tetrahedron_size;
        std::cin >> tetrahedron_reflective >> tetrahedron_transparency;
        std::cin >> tetrahedron_bevel_lights_number;

        std::cin >> floor_a.x >> floor_a.y >> floor_a.z;
        std::cin >> floor_b.x >> floor_b.y >> floor_b.z;
        std::cin >> floor_c.x >> floor_c.y >> floor_c.z;
        std::cin >> floor_d.x >> floor_d.y >> floor_d.z;

        std::cin >> floor_texture_path;

        std::cin >> floor_color.x >> floor_color.y >> floor_color.z;
        std::cin >> floor_reflective;

        std::cin >> cpu_scene_lights_number;
        cpu_scene_lights = (cpu_PointLight*)malloc(cpu_scene_lights_number*sizeof(cpu_PointLight));
        for (uint i = 0; i < cpu_scene_lights_number; ++i){
            cpu_vec3 cur_p;
            std::cin >> cur_p.x, cur_p.y, cur_p.z;
            cpu_vec3 cur_color;
            std::cin >> cur_color.x, cur_color.y, cur_color.z;
            cpu_scene_lights[i] = cpu_PointLight{cur_p, cur_color, 3.0};
        }

        std::cin >> max_bounces;
        std::cin >> ssaa_scale;
    }

    // материалы
        cpu_Material grey_bevels_cpu_material{cpu_vec3{0.1, 0.1, 0.1}, 1.0, -1.0, -1.0};
        cpu_Material lights_cpu_material{cpu_vec3{1.0, 1.0, 1.0}, 10.0, 0.6, -1.0};

    // загрузка объектов
        cpu_scene_objects_number = 3;
        cpu_scene_objects = (cpu_Object*)malloc(cpu_scene_objects_number*sizeof(cpu_Object));

        cpu_Material icosahedron_faces_cpu_material{icosahedron_color, 1.0f-icosahedron_reflective-icosahedron_transparency, icosahedron_reflective, icosahedron_transparency};
        cpu_scene_objects[0] = cpu_Object(icosahedron_center, icosahedron_size, "meshes/icosahedron", icosahedron_faces_cpu_material, grey_bevels_cpu_material, lights_cpu_material, icosahedron_bevel_lights_number, 0.02);
        cpu_Material octahedron_faces_cpu_material{octahedron_color, 1.0f-octahedron_reflective-octahedron_transparency, octahedron_reflective, octahedron_transparency};
        cpu_scene_objects[1] = cpu_Object(octahedron_center, octahedron_size, "meshes/octahedron", octahedron_faces_cpu_material, grey_bevels_cpu_material, lights_cpu_material, octahedron_bevel_lights_number, 0.02);
        cpu_Material tetrahedron_faces_cpu_material{tetrahedron_color, 1.0f-tetrahedron_reflective-tetrahedron_transparency, tetrahedron_reflective, tetrahedron_transparency};
        cpu_scene_objects[2] = cpu_Object(tetrahedron_center, tetrahedron_size, "meshes/tetrahedron", tetrahedron_faces_cpu_material, grey_bevels_cpu_material, lights_cpu_material, tetrahedron_bevel_lights_number, 0.02);
    // загрузка пола
        cpu_Material floor_faces_cpu_material{floor_color, 1.0f-floor_reflective, floor_reflective, -1.0};
        cpu_scene_floor = (cpu_Textured_floor*)malloc(sizeof(cpu_Textured_floor));
        cpu_scene_floor[0] = cpu_Textured_floor(floor_a, floor_b, floor_c, floor_d,
                                        floor_texture_path,
                                        floor_faces_cpu_material);

    // камера
        cpu_Camera cam(cpu_vec3{0.0, 0.0, 0.0}, cpu_vec3{0.0, 0.0, 0.0}, ssaa_scale*w, ssaa_scale*h, fow);

    // рендер
        for(uint k = 0; k < frame_number; k++) {
            float t = k*(2*M_PI/frame_number);

            cpu_vec3 cylindric_p{r_0_c+a_r_c*sin(w_r_c*t+p_r_c),
                             f_0_c+w_f_c*t,
                             z_0_c+a_z_c*sin(w_z_c*t+p_z_c)};

            cpu_vec3 cylindric_d{r_0_n+a_r_n*sin(w_r_n*t+p_r_n),
                             f_0_n+w_f_n*t,
                             z_0_n+a_z_n*sin(w_z_n*t+p_z_n)};

            cpu_vec3 cam_p{cylindric_p.x*cos(cylindric_p.y),
                       cylindric_p.x*sin(cylindric_p.y),
                       cylindric_p.z};
            cpu_vec3 cam_d{cylindric_d.x*cos(cylindric_d.y),
                       cylindric_d.x*sin(cylindric_d.y),
                       cylindric_d.z};

            cam.p = cam_p;
            cam.d = cam_d;

            auto start_time = std::chrono::high_resolution_clock::now();

            cam.Render(max_bounces);

            auto end_time = std::chrono::high_resolution_clock::now();
            auto time = end_time - start_time;

            std::string cur_save_path = pref_save_path+std::to_string(k)+suff_save_path;
            std::cout << k << '\t' << std::fixed << std::setprecision(3) << (1.0/1000.0)*(float)std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() << '\t' << cpu_rays_count << '\n';
            
            FILE* fp = fopen(cur_save_path.c_str(), "wb");
                fwrite(&w, sizeof(int), 1, fp);
                fwrite(&h, sizeof(int), 1, fp);

                cpu_uchar4* res_data = (cpu_uchar4*)malloc(w*h*sizeof(cpu_uchar4));

                for (uint y = 0; y < h; ++y){
                    for (uint x = 0; x < w; ++x){

                        cpu_vec3 mean_color{0.0, 0.0, 0.0};
                        for (uint ss_y = 0; ss_y < ssaa_scale; ++ss_y){
                            for (uint ss_x = 0; ss_x < ssaa_scale; ++ss_x){
                                mean_color.x += (float)cam.data[(ssaa_scale*y+ss_y)*cam.w+(ssaa_scale*x+ss_x)].x;
                                mean_color.y += (float)cam.data[(ssaa_scale*y+ss_y)*cam.w+(ssaa_scale*x+ss_x)].y;
                                mean_color.z += (float)cam.data[(ssaa_scale*y+ss_y)*cam.w+(ssaa_scale*x+ss_x)].z;
                            }
                        }
                        mean_color.x /= ssaa_scale*ssaa_scale;
                        mean_color.y /= ssaa_scale*ssaa_scale;
                        mean_color.z /= ssaa_scale*ssaa_scale;

                        // std::cout << (uint)(std::clamp(mean_color.x, 0.0f, 255.0f)) << '\n';

                        res_data[y*w+x].x = (uint)(std::clamp(mean_color.x, 0.0f, 255.0f));
                        res_data[y*w+x].y = (uint)(std::clamp(mean_color.y, 0.0f, 255.0f));
                        res_data[y*w+x].z = (uint)(std::clamp(mean_color.z, 0.0f, 255.0f));
                        res_data[y*w+x].w = 255;

                        fwrite(&res_data[y*w+x].x, sizeof(char), 1, fp);
                        fwrite(&res_data[y*w+x].y, sizeof(char), 1, fp);
                        fwrite(&res_data[y*w+x].z, sizeof(char), 1, fp);
                        fwrite(&res_data[y*w+x].w, sizeof(char), 1, fp);
                    }
                }

                free(res_data);
                free(cam.data);
                fclose(fp);
        }

    // завершение работы

        free(cpu_scene_lights);
        for (int i = 0; i < cpu_scene_objects_number; ++i){
            free(cpu_scene_objects[i].faces);
            free(cpu_scene_objects[i].bevels);
            free(cpu_scene_objects[i].caps);
            free(cpu_scene_objects[i].dev_lights);
        }
        free(cpu_scene_objects);
        free(cpu_scene_floor[0].cpu_polygons);
        free(cpu_scene_floor[0].tex_data);

	return 0;
}
