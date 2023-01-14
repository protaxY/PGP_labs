// вариант 5 Тетраэдр, Октаэдр, Икосаэдр
#include <stdlib.h>
#include <stdio.h> 
#include <math.h>
#include <string>
#include <string.h>
#include <fstream>
#include <vector>
#include <limits>
#include <iostream>
#include <iomanip>
#include <chrono>

#define CSC(call)																							\
do {																										\
	cudaError_t status = call;																				\
	if (status != cudaSuccess) {																			\
		fprintf(stderr, "ERROR is %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(status));	\
		exit(0);																							\
	}																										\
} while(0)

#define _index(i) ((i) + ((i) >> 5))

unsigned long long rays_count = 0;

const uint BLOCK_SIZE = 128;
const uint LOG2_BLOCK_SIZE = 7;
const uint GRID_SIZE = 128;

const float INTENCITY_THRESHOLD = 1e-4;
uint MAX_BOUNCES = 1;
const float RAY_OFFSET = 1e-4;

float clamp(float val, float l, float r){
    return std::max(l, std::min(val, r));
}

struct vec3 {
	float x;
	float y;
	float z;
};

class matr3{
    public:
        float elems[3][3];

        matr3(vec3 a, vec3 b, vec3 c){
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
    __host__ __device__ float dot(vec3 a, vec3 b) {
        return a.x*b.x+a.y*b.y+a.z*b.z;
    }

    __host__ __device__ vec3 prod(vec3 b, vec3 a) {
        return {a.y*b.z-a.z*b.y,
                a.z*b.x-a.x*b.z,
                a.x*b.y-a.y*b.x};
    }

    __host__ __device__ vec3 norm(vec3 v) {
        float l = sqrt(dot(v, v));
        return {v.x/l, v.y/l, v.z/l};
    }

    __host__ __device__ float length(vec3 v){
        return sqrt(dot(v, v));
    }

    __host__ __device__ vec3 diff(vec3 a, vec3 b) {
        return {a.x-b.x, a.y-b.y, a.z-b.z};
    }

    __host__ __device__ vec3 add(vec3 a, vec3 b) {
        return {a.x+b.x, a.y+b.y, a.z+b.z};
    }

    __host__ __device__ vec3 mult(matr3 matr, vec3 v) {
        return vec3{matr.elems[0][0]*v.x+matr.elems[0][1]*v.y+matr.elems[0][2]*v.z,
                    matr.elems[1][0]*v.x+matr.elems[1][1]*v.y+matr.elems[1][2]*v.z,
                    matr.elems[2][0]*v.x+matr.elems[2][1]*v.y+matr.elems[2][2]*v.z};
    }

    __host__ __device__ vec3 mult(float a, vec3 v){
        return vec3{a*v.x, a*v.y, a*v.z};
    }

    __host__ __device__ void print(vec3 v) {
        printf("%e %e %e\n", v.x, v.y, v.z);
    }

struct Material{
    vec3 color;
    float phong;
    float reflective;
    float transparency;
};

__constant__ float INF = std::numeric_limits<float>::max();
__constant__ vec3 BACKGOUND_COLOR{0.1, 0.1, 0.1};
__constant__ vec3 BACKGOUND_LIGHT_COLOR{0.0, 0.0, 0.0};

class Polygon{
    public:
        vec3 a;
        vec3 b;
        vec3 c;

        Polygon(vec3 init_a, vec3 init_b, vec3 init_c){
            a = init_a;
            b = init_b;
            c = init_c;
        }

        __host__ __device__ vec3 Normal(){
            return norm(prod(diff(b,a), diff(c,a)));
        }
};

struct PointLight{
    vec3 p;
    vec3 color;
    float intencity;
};

class Object{
    public:
        vec3 p;
        float scale;

        // адреса указывают на видеопамять
        Polygon* faces;
        uint faces_number;
        Material face_material;
        Polygon* bevels;
        uint bevels_number;
        Material bevel_material;
        Polygon* caps;
        uint caps_number;

        vec3* dev_lights;
        uint lights_number;
        float lights_radius;
        Material lights_material;

        void Import_obj(vec3 p, float scale, Polygon*& out_polygons, uint& out_polygons_number, std::string path){
            std::string word;
            std::vector<vec3> verts;
            std::vector<Polygon> polygons;
            std::ifstream faces_file;
            faces_file.open(path);
            while (faces_file >> word){
                if (word == "v"){
                    float a, b, c;
                    faces_file >> a >> b >> c;
                    verts.push_back(vec3{a, b, c});
                }

                if (word == "f"){
                    uint id1, id2, id3;
                    faces_file >> id1 >> id2 >> id3;
                    --id1;
                    --id2;
                    --id3;

                    Polygon cur_polygon = Polygon(add(p, mult(scale, vec3{verts[id1].x, verts[id1].y, verts[id1].z})),
                                                  add(p, mult(scale, vec3{verts[id2].x, verts[id2].y, verts[id2].z})),
                                                  add(p, mult(scale, vec3{verts[id3].x, verts[id3].y, verts[id3].z})));
                    polygons.push_back(cur_polygon);
                }   
            }
            out_polygons_number = polygons.size();
            out_polygons = (Polygon*)malloc(out_polygons_number*sizeof(Polygon));
            std::copy(polygons.begin(), polygons.end(), out_polygons);
            polygons.clear();

            Polygon* dev_out_polygons;
            CSC(cudaMalloc(&dev_out_polygons, out_polygons_number*sizeof(Polygon)));
            CSC(cudaMemcpy(dev_out_polygons, out_polygons, out_polygons_number*sizeof(Polygon), cudaMemcpyHostToDevice));
            free(out_polygons);
            out_polygons = dev_out_polygons;
        }

        void Import_obj_lights(vec3 p, float scale, uint bevel_lights_number, std::string path){
            std::string word;
            std::vector<vec3> verts;
            std::vector<std::pair<vec3, vec3>> verts_pairs;
            std::ifstream faces_file;
            faces_file.open(path);
            while (faces_file >> word){
                if (word == "v"){
                    float a, b, c;
                    faces_file >> a >> b >> c;
                    verts.push_back(vec3{a, b, c});
                }

                if (word == "l"){
                    uint id1, id2;
                    faces_file >> id1 >> id2;
                    --id1;
                    --id2;

                    std::pair<vec3, vec3> cur_pair(add(p, mult(scale, vec3{verts[id1].x, verts[id1].y, verts[id1].z})), 
                                                   add(p, mult(scale, vec3{verts[id2].x, verts[id2].y, verts[id2].z})));
                    verts_pairs.push_back(cur_pair);
                }
            }

            lights_number = bevel_lights_number*verts_pairs.size();
            vec3* lights = (vec3*)malloc(lights_number*sizeof(vec3));

            for (uint line_id = 0; line_id < verts_pairs.size(); ++line_id){
                vec3 step = diff(verts_pairs[line_id].second, verts_pairs[line_id].first);
                step = mult(1.0/(bevel_lights_number+1.0), step);

                for (int i = 0; i < bevel_lights_number; ++i){
                    lights[line_id*bevel_lights_number+i] = add(verts_pairs[line_id].first, mult(i+1.0, step));
                }
            }

            CSC(cudaMalloc(&dev_lights, lights_number*sizeof(vec3)));
            CSC(cudaMemcpy(dev_lights, lights, lights_number*sizeof(vec3), cudaMemcpyHostToDevice));
            free(lights);
        }

        Object(vec3 init_p, float init_scale, std::string dir_path, Material init_face_material, Material init_bevel_material, Material init_lights_material, uint bevel_lights_number, float init_lights_radius){
            p = init_p;
            scale = init_scale;
            
            Import_obj(init_p, scale, faces, faces_number, dir_path+"/faces.obj");
            Import_obj(init_p, scale, bevels, bevels_number, dir_path+"/bevels.obj");
            Import_obj(init_p, scale, caps, caps_number, dir_path+"/caps.obj");
            face_material = init_face_material;
            bevel_material = init_bevel_material;

            Import_obj_lights(init_p, scale, bevel_lights_number, dir_path+"/bevel_lines.obj");
            lights_material = init_lights_material;
            lights_radius = init_lights_radius;
        }

        Object() = default;
};

class Textured_floor{
    public:
        vec3 a;
        vec3 b;
        vec3 c;
        vec3 d;
        Material floor_material;
        uint tex_w;
        uint tex_h;

        // ссылка на данные текстуры
        cudaArray* dev_tex_data;

        // указывает на видеопамять
        Polygon* dev_polygons;
        cudaTextureObject_t texObj;

        Textured_floor(vec3 init_a, vec3 init_b, vec3 init_c, vec3 init_d, std::string image_path, Material init_floor_material){
            a = init_a;
            b = init_b;
            c = init_c;
            d = init_d;
            floor_material = init_floor_material;

            Polygon* polygons = (Polygon*)malloc(2*sizeof(Polygon));
            polygons[0] = Polygon(init_a, init_c, init_b);
            polygons[1] = Polygon(init_d, init_b, init_c);


            CSC(cudaMalloc(&dev_polygons, 2*sizeof(Polygon)));
            CSC(cudaMemcpy(dev_polygons, polygons, 2*sizeof(Polygon), cudaMemcpyHostToDevice));
            free(polygons);

            // загрузка текстуры
                uint w, h;
                FILE* fp = fopen(image_path.c_str(), "rb");
                fread(&w, sizeof(uint), 1, fp);
                fread(&h, sizeof(uint), 1, fp);

                tex_w = w;
                tex_h = h;

                uchar4* tex_data = (uchar4*)malloc(sizeof(uchar4) * w * h);
                fread(tex_data, sizeof(uchar4), w * h, fp);
                fclose(fp);

                cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
                CSC(cudaMallocArray(&dev_tex_data, &ch, w, h));

                CSC(cudaMemcpy2DToArray(dev_tex_data, 0, 0, tex_data, w * sizeof(uchar4), w * sizeof(uchar4), h, cudaMemcpyHostToDevice));

                struct cudaResourceDesc resDesc;
                memset(&resDesc, 0, sizeof(resDesc));
                resDesc.resType = cudaResourceTypeArray;
                resDesc.res.array.array = dev_tex_data;

                struct cudaTextureDesc texDesc;
                memset(&texDesc, 0, sizeof(texDesc));
                texDesc.addressMode[0] = cudaAddressModeClamp;
                texDesc.addressMode[1] = cudaAddressModeClamp;
                texDesc.filterMode = cudaFilterModePoint;
                texDesc.readMode = cudaReadModeElementType;
                texDesc.normalizedCoords = 0;

                texObj = 0;
                cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
        }

        __device__ vec3 Get_color(float u, float v, uint polygon_id){
            uint x = (uint)(u*(tex_w-1));
            uint y = (uint)(v*(tex_h-1));
            vec3 res;
            if (polygon_id == 0){
                uchar4 p = tex2D<uchar4>(texObj, x, y);

                res = vec3{p.x/255.0f, p.y/255.0f, p.z/255.0f};
            }
            else if (polygon_id == 1){
                uchar4 p = tex2D<uchar4>(texObj, (tex_w-1)-x, (tex_h-1)-y);
                res = vec3{p.x/255.0f, p.y/255.0f, p.z/255.0f};
            }

            return res;
        }
};

Object* scene_objects;
uint scene_objects_number;
PointLight* scene_lights;
uint scene_lights_number;

Textured_floor* scene_floor;

class Ray{
    public:
        vec3 p, d;
        uint parent;
        uint bounses;
        float intencity;
        vec3 color;

        __host__ __device__ Ray(vec3 init_p, vec3 init_d, uint init_bounces, float init_intencity){
            p = init_p;
            d = init_d;
            bounses = init_bounces;
            intencity = init_intencity;
            color = vec3{-1.0, -1.0, -1.0};
        }
};

__device__ float is_ray_hits_polygon(Ray ray, Polygon polygon){
    vec3 e1 = diff(polygon.b, polygon.a);
    vec3 e2 = diff(polygon.c, polygon.a);

    vec3 p = prod(ray.d, e2);
    double div = dot(p, e1);
    if (fabs(div) < 1e-10)
        return INF;
    vec3 t = diff(ray.p, polygon.a);
    double u = dot(p, t)/div;
    if (u<0.0 || u>1.0)
        return INF;
    vec3 q = prod(t, e1);
    double v = dot(q, ray.d)/div;
    if (v<0.0 || v+u>1.0)
        return INF;
    double ts = dot(q, e2)/div; 
    if (ts < 0.0)
        return INF;

    return ts;
}

__device__ float is_floor_ray_hits_polygon(float& x, float& y, Ray ray, Polygon polygon){
    vec3 e1 = diff(polygon.b, polygon.a);
    vec3 e2 = diff(polygon.c, polygon.a);

    vec3 p = prod(ray.d, e2);
    double div = dot(p, e1);
    if (fabs(div) < 1e-10)
        return INF;
    vec3 t = diff(ray.p, polygon.a);
    double u = dot(p, t)/div;
    if (u<0.0 || u>1.0)
        return INF;
    vec3 q = prod(t, e1);
    double v = dot(q, ray.d)/div;
    if (v<0.0 || v+u>1.0)
        return INF;
    double ts = dot(q, e2)/div; 
    if (ts < 0.0)
        return INF;

    x = u;
    y = v;

    return ts;
}

__global__ void trace_rays_kernel(Ray* rays, bool* rays_bitmap, uint rays_traced_number, uint rays_to_trace_number, Object* objects, uint objects_number, PointLight* lights, uint lights_number, Textured_floor* floor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    while (idx < rays_to_trace_number){
        float min_t = INF;
        Material hit_material;
        vec3 hit_normal;
        bool hit_bevel = false;
        bool no_shadow = false;
        uint hit_object_id;

        for (uint object_id = 0; object_id < objects_number; ++object_id){
            // грани
            for (uint polygon_id = 0; polygon_id < objects[object_id].faces_number; ++polygon_id){
                float t = is_ray_hits_polygon(rays[rays_traced_number+idx], objects[object_id].faces[polygon_id]);
                if (min_t > t){
                    min_t = t;
                    hit_material = objects[object_id].face_material;
                    hit_normal = objects[object_id].faces[polygon_id].Normal();
                    hit_bevel = false;
                }
            }
            // ребра
            for (uint polygon_id = 0; polygon_id < objects[object_id].bevels_number; ++polygon_id){
                float t = is_ray_hits_polygon(rays[rays_traced_number+idx], objects[object_id].bevels[polygon_id]);
                if (min_t > t){

                    min_t = t;
                    hit_material = objects[object_id].bevel_material;
                    hit_normal = objects[object_id].bevels[polygon_id].Normal();
                    hit_bevel = true;
                    hit_object_id = object_id;
                }
            }
            // крышки
            for (uint polygon_id = 0; polygon_id < objects[object_id].caps_number; ++polygon_id){
                float t = is_ray_hits_polygon(rays[rays_traced_number+idx], objects[object_id].caps[polygon_id]);
                if (min_t > t){
                    min_t = t;
                    hit_material = objects[object_id].bevel_material;
                    hit_normal = objects[object_id].caps[polygon_id].Normal();
                    hit_bevel = false;
                }
            }
        }

        vec3 hit_color;
        vec3 hit_p = add(rays[rays_traced_number+idx].p, mult(min_t, rays[rays_traced_number+idx].d));

        // bool hit_floor = false;
        if (min_t == INF){
            // столкновение с полом
            uint hit_polygon_id;
            float u, v;
            for (uint polygon_id = 0; polygon_id < 2; ++polygon_id){
                float t = is_floor_ray_hits_polygon(u, v, rays[rays_traced_number+idx], floor[0].dev_polygons[polygon_id]);

                if (min_t > t){
                    min_t = t;
                    hit_material = floor[0].floor_material;
                    hit_normal = floor[0].dev_polygons[polygon_id].Normal();
                    hit_polygon_id = polygon_id;
                }
            }

            // если попал
            if (min_t != INF){
                // hit_floor = true;
                hit_color = floor[0].Get_color(u, v, hit_polygon_id);
            }
        }
        else if (hit_bevel){
            hit_color = hit_material.color;
            // printf("%d\n", hit_object_id);
            for (uint light_id = 0; light_id < objects[hit_object_id].lights_number; ++light_id){
                if (dot(hit_normal, rays[rays_traced_number+idx].d) > 0.0)
                    hit_normal = mult(-1.0, hit_normal);
                if (dot(hit_normal, diff(hit_p, objects[hit_object_id].p)) < 0.0
                    && objects[hit_object_id].lights_radius > length(diff(hit_p, objects[hit_object_id].dev_lights[light_id]))){
                    hit_color = objects[hit_object_id].lights_material.color;    
                    hit_material = objects[hit_object_id].lights_material;
                    no_shadow = true;
                }
            }
        }
        else {
            hit_color = hit_material.color;
        }

        // двусторонние полигоны
        if (dot(hit_normal, rays[rays_traced_number+idx].d) > 0.0)
            hit_normal = mult(-1.0, hit_normal);
        if (min_t == INF){
            rays[rays_traced_number+idx].color = BACKGOUND_COLOR;
        }
        else {
            vec3 p = add(rays[rays_traced_number+idx].p, mult(min_t, rays[rays_traced_number+idx].d));
            for (uint light_id = 0; light_id < lights_number; ++light_id){
                    vec3 l = diff(lights[light_id].p, p);
                    
                    // теневой луч
                        vec3 shadow_d = norm(l);
                        Ray shadow_ray = Ray(add(p, mult(-RAY_OFFSET, rays[rays_traced_number+idx].d)), shadow_d, 0, 1.0);
                        vec3 shadow_color{1.0, 1.0, 1.0};
                        bool full_shadow = false;

                        if (!no_shadow){
                            for (uint object_id = 0; object_id < objects_number; ++object_id){
                                // грани
                                for (uint polygon_id = 0; polygon_id < objects[object_id].faces_number; ++polygon_id){
                                    float t = is_ray_hits_polygon(shadow_ray, objects[object_id].faces[polygon_id]);
                                    if (t != INF){
                                        Material shadow_hit_material = objects[object_id].face_material;

                                        if (shadow_hit_material.transparency > 0.0){
                                            vec3 subtractive_color{1.0f - shadow_hit_material.color.x,
                                                            1.0f - shadow_hit_material.color.y,
                                                            1.0f - shadow_hit_material.color.z};
                                            subtractive_color = mult((1.0f-shadow_hit_material.transparency), subtractive_color);
                                            shadow_color = diff(shadow_color, subtractive_color);
                                        }
                                        else{
                                            full_shadow = true;
                                            break;
                                        }
                                    }
                                }
                                // ребра
                                for (uint polygon_id = 0; polygon_id < objects[object_id].bevels_number; ++polygon_id){
                                    float t = is_ray_hits_polygon(shadow_ray, objects[object_id].bevels[polygon_id]);
                                    if (t != INF){
                                        Material shadow_hit_material = objects[object_id].bevel_material;

                                        if (shadow_hit_material.transparency > 0.0){
                                            vec3 subtractive_color{1.0f - shadow_hit_material.color.x,
                                                                1.0f - shadow_hit_material.color.y,
                                                                1.0f - shadow_hit_material.color.z};
                                            subtractive_color = mult((1.0f-shadow_hit_material.transparency), subtractive_color);
                                            shadow_color = diff(shadow_color, subtractive_color);
                                        }
                                        else{
                                            full_shadow = true;
                                            break;
                                        }
                                    }
                                }
                                // крышки
                                for (uint polygon_id = 0; polygon_id < objects[object_id].caps_number; ++polygon_id){
                                    float t = is_ray_hits_polygon(shadow_ray, objects[object_id].caps[polygon_id]);
                                    if (t != INF){
                                        Material shadow_hit_material = objects[object_id].bevel_material;

                                        if (shadow_hit_material.transparency > 0.0){
                                            vec3 subtractive_color{1.0f - shadow_hit_material.color.x,
                                                            1.0f - shadow_hit_material.color.y,
                                                            1.0f - shadow_hit_material.color.z};
                                            subtractive_color = mult((1.0f-shadow_hit_material.transparency), subtractive_color);
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
                    rays[rays_traced_number+idx].color = vec3{0.0, 0.0, 0.0};
                    if (!full_shadow){
                        float n_l_cos = dot(hit_normal, norm(l));
                        if (n_l_cos < 0.0)
                            n_l_cos = 0.0;
                        rays[rays_traced_number+idx].color = mult(n_l_cos, hit_color);

                        vec3 r = diff(mult(2*dot(hit_normal, norm(l)), hit_normal), norm(l));
                        float v_r_cos = dot(mult(-1.0, rays[rays_traced_number+idx].d), norm(r));
                        if (v_r_cos < 0){
                            v_r_cos = 0.0;
                        }
                        float specular_sharpness = 8.0;
                        rays[rays_traced_number+idx].color = add(rays[rays_traced_number+idx].color, mult(pow(v_r_cos, specular_sharpness), lights[light_id].color));

                        rays[rays_traced_number+idx].color.x *= shadow_color.x;
                        rays[rays_traced_number+idx].color.y *= shadow_color.y;
                        rays[rays_traced_number+idx].color.z *= shadow_color.z;
                    }
                    rays[rays_traced_number+idx].color = add(rays[rays_traced_number+idx].color, mult(0.1, hit_color));
                    rays[rays_traced_number+idx].color = add(rays[rays_traced_number+idx].color, BACKGOUND_LIGHT_COLOR);
                }
            if (rays[rays_traced_number+idx].bounses != 0 and rays[rays_traced_number+idx].intencity > INTENCITY_THRESHOLD){
                // уменьшить интесивность Фонга
                    rays[rays_traced_number+idx].color = mult(hit_material.phong, rays[rays_traced_number+idx].color);
                // отраженный
                if (rays[rays_traced_number+idx].intencity*hit_material.reflective > INTENCITY_THRESHOLD){
                    vec3 r_dir = mult(-2.0*dot(rays[rays_traced_number+idx].d, hit_normal), hit_normal);
                    r_dir = add(r_dir, rays[rays_traced_number+idx].d);
                    rays[rays_traced_number+rays_to_trace_number+idx] = Ray(add(p, mult(-RAY_OFFSET, rays[rays_traced_number+idx].d)), r_dir, rays[rays_traced_number+idx].bounses-1, rays[rays_traced_number+idx].intencity*hit_material.reflective);
                    rays[rays_traced_number+rays_to_trace_number+idx].parent = rays_traced_number+idx;
                    rays_bitmap[rays_traced_number+rays_to_trace_number+idx] = false;
                }
                // сквозной
                if (rays[rays_traced_number+idx].intencity*hit_material.transparency > INTENCITY_THRESHOLD){
                    rays[rays_traced_number+2*rays_to_trace_number+idx] = Ray(add(p, mult(RAY_OFFSET, rays[rays_traced_number+idx].d)), rays[rays_traced_number+idx].d, rays[rays_traced_number+idx].bounses-1, rays[rays_traced_number+idx].intencity*hit_material.transparency);
                    rays[rays_traced_number+2*rays_to_trace_number+idx].parent = rays_traced_number+idx;
                    rays_bitmap[rays_traced_number+2*rays_to_trace_number+idx] = false;
                }
            }
        }

        idx += gridDim.x * blockDim.x;
    }
}

// сортировка
    __global__ void map_rays_bitmap_kernel(bool* rays_bitmap, uint rays_number, uint bitmap_size){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        while(idx < bitmap_size){
            if (idx < rays_number){
                rays_bitmap[idx] = false;
            }
            else {
                rays_bitmap[idx] = true;
            }

            idx += gridDim.x * blockDim.x;
        }
    }

    __global__ void scan_blocks_kernel(uint* data, uint* sums, uint sums_size) {
        int blockId = blockIdx.x;
        while (blockId < sums_size) {
            extern __shared__ uint temp[];
            temp[_index(threadIdx.x)] = data[blockDim.x * blockId + threadIdx.x];
            __syncthreads();

            uint stride = 1;
            for (uint d = 0; d < LOG2_BLOCK_SIZE; ++d) {
                if ((threadIdx.x + 1) % (stride<<1) == 0) {
                    temp[_index(threadIdx.x)] += temp[_index(threadIdx.x-stride)];
                }
                stride <<= 1;
                __syncthreads();
            }

            if (threadIdx.x == 0) {
                sums[blockId] = temp[_index(BLOCK_SIZE - 1)];
                temp[_index(BLOCK_SIZE - 1)] = 0;
            }
            __syncthreads();

            stride = 1 << LOG2_BLOCK_SIZE-1;
            while (stride != 0) {
                if ((threadIdx.x + 1) % (stride<<1) == 0) {
                    uint tmp = temp[_index(threadIdx.x)];
                    temp[_index(threadIdx.x)] += temp[_index(threadIdx.x - stride)];
                    temp[_index(threadIdx.x - stride)] = tmp;
                }
                stride >>= 1;
                __syncthreads();
            }

            data[blockDim.x * blockId + threadIdx.x] = temp[_index(threadIdx.x)];

            blockId += gridDim.x;
        }
    }

    __global__ void add_kernel(uint* data, uint* sums, uint sums_size) {
        uint blockId = blockIdx.x+1; 
        while (blockId < sums_size) {
            if (blockId != 0) {
                data[blockDim.x * blockId + threadIdx.x] += sums[blockId];
            }

            blockId += gridDim.x;
        }
    }

    __global__ void s_gen_kernel(bool* b, uint n, uint* s){
        uint idx = blockIdx.x * blockDim.x + threadIdx.x;
        while (idx < n){
            s[idx] = b[idx];

            idx += gridDim.x * blockDim.x;
        }
    }

    __global__ void binary_digit_sort_kernel(Ray* a, bool* b, uint* s, uint size, Ray* res){
        uint idx = blockIdx.x * blockDim.x + threadIdx.x;

        while(idx < size){
            if (b[idx]){
                res[s[idx]+(size-s[size])] = Ray(a[idx]);
                // print(res[s[idx]+(size-s[size])].d);
            }
            else{
                res[idx-s[idx]] = Ray(a[idx]);
                // print(res[idx-s[idx]].d);
            }
                
            
            idx += gridDim.x * blockDim.x;
        }
    }

    void scan(uint* dev_data, uint size) {
        if (size % BLOCK_SIZE != 0)
            size += BLOCK_SIZE - (size % BLOCK_SIZE);
        uint sums_size = size/BLOCK_SIZE;

        uint* dev_sums;
        CSC(cudaMalloc(&dev_sums, (sums_size * sizeof(uint))));
        scan_blocks_kernel <<< GRID_SIZE, BLOCK_SIZE, _index(BLOCK_SIZE) * sizeof(uint) >>> (dev_data, dev_sums, sums_size);
        CSC(cudaGetLastError());

        if (size <= BLOCK_SIZE)
            return;
        scan(dev_sums, sums_size);

        add_kernel <<< GRID_SIZE, BLOCK_SIZE >>> (dev_data, dev_sums, sums_size);
        CSC(cudaGetLastError());

        cudaFree(dev_sums);
    }

    uint compact(Ray* &dev_rays, bool* dev_rays_bitmap, uint size){

        Ray* dev_compact_rays;
        CSC(cudaMalloc(&dev_compact_rays, (size*sizeof(Ray))));

        uint* dev_s;
        CSC(cudaMalloc(&dev_s, ((size+1)*sizeof(uint))));
        
        s_gen_kernel <<< GRID_SIZE, BLOCK_SIZE >>> (dev_rays_bitmap, size, dev_s);
        CSC(cudaGetLastError());

        scan(dev_s, size+1);

        uint new_rays_number;
        cudaMemcpy(&new_rays_number, dev_s+size, sizeof(uint), cudaMemcpyDeviceToHost);
        new_rays_number = size-new_rays_number;

        binary_digit_sort_kernel <<< GRID_SIZE, BLOCK_SIZE >>> (dev_rays, dev_rays_bitmap, dev_s, size, dev_compact_rays);
        CSC(cudaGetLastError());

        std::swap(dev_rays, dev_compact_rays);

        cudaFree(dev_s);
        cudaFree(dev_compact_rays);

        return new_rays_number;
    }

// обратный проход
    // left - индекс первого элемента на отрезке, right - индекс первого элемента за отрезком
    __global__ void back_collection_step_kernel(Ray* rays, uint rays_number, uint left, uint right){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        while(idx < right-left){

            atomicAdd(&(rays[rays[left+idx].parent].color.x), rays[left+idx].color.x*rays[left+idx].intencity);
            atomicAdd(&(rays[rays[left+idx].parent].color.y), rays[left+idx].color.y*rays[left+idx].intencity);
            atomicAdd(&(rays[rays[left+idx].parent].color.z), rays[left+idx].color.z*rays[left+idx].intencity);
            
            idx += gridDim.x * blockDim.x;
        }
    }

class Camera{
    public:
        vec3 p, d;
        uint w, h;
        float fov;
        uchar4* data;
        Ray* dev_rays;

        Object* dev_scene_objects;
        PointLight* dev_scene_lights;
        Textured_floor* dev_scene_floor;

        Camera(vec3 init_p, vec3 init_d, uint init_w, uint init_h, float init_fov){
            p = init_p;
            d = init_d;
            w = init_w;
            h = init_h;
            fov = init_fov;

            // записать сцену в видеопамять
                CSC(cudaMalloc(&dev_scene_objects, scene_objects_number*sizeof(Object)));
                CSC(cudaMalloc(&dev_scene_lights, scene_lights_number*sizeof(PointLight)));
                CSC(cudaMemcpy(dev_scene_objects, scene_objects, scene_objects_number*sizeof(Object), cudaMemcpyHostToDevice));
                CSC(cudaMemcpy(dev_scene_lights, scene_lights, scene_lights_number*sizeof(PointLight), cudaMemcpyHostToDevice));

            // пол в видеопамять
                CSC(cudaMalloc(&dev_scene_floor, sizeof(Textured_floor)));
                CSC(cudaMemcpy(dev_scene_floor, scene_floor, sizeof(Textured_floor), cudaMemcpyHostToDevice));
        }

        void Render(uint bounces){
            Ray* rays;
            rays = (Ray*)malloc(w*h*sizeof(Ray));

            float dw = 2.0/(w-1.0);
            float dh = 2.0/(h-1.0);
            float z_side = 1.0/tan(fov*(M_PI/360.0));

            vec3 basis_z = norm(diff(d, p));
            vec3 basis_x = norm(prod(basis_z, vec3{0.0, 0.0, 1.0}));
            vec3 basis_y = norm(prod(basis_x, basis_z));
            for (int y = 0; y < h; ++y){
                for (int x = 0; x < w; ++x){
                    vec3 v = vec3{-1.0f+dw*x, -1.0f+dh*y* h / w, z_side};
                    matr3 matr(basis_x, basis_y, basis_z);
                    vec3 dir = norm(mult(matr, v));

                    rays[x*h+y] = Ray(p, dir, bounces, 1.0);
                }
            }

            bool* dev_rays_bitmap;
            // копирование лучей в память gpu
                CSC(cudaMalloc(&dev_rays, 3*w*h*sizeof(Ray)));
                CSC(cudaMalloc(&dev_rays_bitmap, 3*w*h*sizeof(bool)));
                CSC(cudaMemcpy(dev_rays, rays, w*h*sizeof(Ray), cudaMemcpyHostToDevice));
                free(rays);
                map_rays_bitmap_kernel <<< GRID_SIZE, BLOCK_SIZE >>> (dev_rays_bitmap, w*h, 3*w*h);

            uint rays_number = w*h;
            uint traced_rays_number = 0;
            uint rays_to_trace_number = w*h;

            std::vector<uint> rays_traced_chunks_shifts(1, 0);

            for (int i = 0; i <= bounces; ++i){
                rays_traced_chunks_shifts.push_back(rays_to_trace_number+rays_traced_chunks_shifts.back());
                trace_rays_kernel <<< GRID_SIZE, BLOCK_SIZE >>> (dev_rays, dev_rays_bitmap, 
                                                                traced_rays_number, 
                                                                rays_to_trace_number,
                                                                dev_scene_objects, 
                                                                scene_objects_number, 
                                                                dev_scene_lights,
                                                                scene_lights_number,
                                                                dev_scene_floor);
                CSC(cudaGetLastError());

                uint old_traced_rays_number = traced_rays_number;
                traced_rays_number = rays_number;
                rays_number = compact(dev_rays, dev_rays_bitmap, old_traced_rays_number+(3*rays_to_trace_number));

                rays_to_trace_number = rays_number-traced_rays_number;

                if (rays_to_trace_number == 0){
                    break;
                }

                // перевыделение памяти
                    Ray* new_dev_rays;
                    CSC(cudaMalloc(&new_dev_rays, (traced_rays_number+(3*rays_to_trace_number))*sizeof(Ray)));
                    CSC(cudaMemcpy(new_dev_rays, dev_rays, rays_number*sizeof(Ray), cudaMemcpyDeviceToDevice));
                    cudaFree(dev_rays);
                    dev_rays = new_dev_rays;

                    cudaFree(dev_rays_bitmap);
                    CSC(cudaMalloc(&dev_rays_bitmap, (traced_rays_number+(3*rays_to_trace_number))*sizeof(bool)));
                    map_rays_bitmap_kernel <<< GRID_SIZE, BLOCK_SIZE >>> (dev_rays_bitmap, rays_number, (traced_rays_number+(3*rays_to_trace_number)));
                    CSC(cudaGetLastError());
            }
            cudaFree(dev_rays_bitmap);

            // сжать результат
                Ray* new_dev_rays;
                CSC(cudaMalloc(&new_dev_rays, rays_number*sizeof(Ray)));
                CSC(cudaMemcpy(new_dev_rays, dev_rays, rays_number*sizeof(Ray), cudaMemcpyDeviceToDevice));
                cudaFree(dev_rays);
                dev_rays = new_dev_rays;

                rays_count = rays_number;

            // обратный обход деревьев
                for (uint shift_index = rays_traced_chunks_shifts.size()-2; shift_index >= 1; --shift_index){
                    back_collection_step_kernel <<< GRID_SIZE, BLOCK_SIZE >>> (dev_rays, rays_number, rays_traced_chunks_shifts[shift_index], rays_traced_chunks_shifts[shift_index+1]);
                    CSC(cudaGetLastError());
                }


            Ray* screen_rays = (Ray*)malloc(w*h*sizeof(Ray));
            CSC(cudaMemcpy(screen_rays, dev_rays, w*h*sizeof(Ray), cudaMemcpyDeviceToHost));

            data = (uchar4*)malloc(w*h*sizeof(uchar4));
            for (int y = 0; y < h; ++y){
                for (int x = 0; x < w; ++x){
                    data[(h-1-y)*w+x].w = 255;
                    data[(h-1-y)*w+x].x = (uint)(clamp(screen_rays[x*h+y].color.x, 0.0f, 1.0f)*255);
                    data[(h-1-y)*w+x].y = (uint)(clamp(screen_rays[x*h+y].color.y, 0.0f, 1.0f)*255);
                    data[(h-1-y)*w+x].z = (uint)(clamp(screen_rays[x*h+y].color.z, 0.0f, 1.0f)*255);
                }
            }

            free(screen_rays);
            cudaFree(dev_rays);
        }
};

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

    // bool hit_floor = false;
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
            // hit_floor = true;
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

                    cpu_vec3 traced_color = trace(cpu_Ray(p, dir, bounces, 1.0), cpu_scene_objects, cpu_scene_objects_number, cpu_scene_lights, cpu_scene_lights_number, cpu_scene_floor);

                    data[(h-1-y)*w+x].w = 255;
                    data[(h-1-y)*w+x].x = (uint)(clamp(traced_color.x, 0.0f, 1.0f)*255);
                    data[(h-1-y)*w+x].y = (uint)(clamp(traced_color.y, 0.0f, 1.0f)*255);
                    data[(h-1-y)*w+x].z = (uint)(clamp(traced_color.z, 0.0f, 1.0f)*255);
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
    default_mode = true;

    if (gpu){
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

        vec3 icosahedron_center;
        vec3 icosahedron_color;
        float icosahedron_size;
        float icosahedron_reflective, icosahedron_transparency;
        uint icosahedron_bevel_lights_number;

        vec3 octahedron_center;
        vec3 octahedron_color;
        float octahedron_size;
        float octahedron_reflective, octahedron_transparency;
        uint octahedron_bevel_lights_number;

        vec3 tetrahedron_center;
        vec3 tetrahedron_color;
        float tetrahedron_size;
        float tetrahedron_reflective, tetrahedron_transparency;
        uint tetrahedron_bevel_lights_number;

        vec3 floor_a, floor_b, floor_c, floor_d;

        std::string floor_texture_path;
            
        vec3 floor_color;
        float floor_reflective;

        uint max_bounces;    
        uint ssaa_scale;

        if (default_mode){
            frame_number = 126;
            std::cout << frame_number << '\n';
            save_path = "render/%d.data";
            std::cout << save_path << '\n';

            pref_save_path = "render/";
            suff_save_path = ".data";

            w = 600;
            h = 600;
            fow = 30.0;
            std::cout << w << ' ' << h << ' ' << fow << '\n';

            r_0_c = 7.0; z_0_c = 3.0; f_0_c = 0.0;
            a_r_c = 2.0; a_z_c = 1.0;
            w_r_c = 2.0; w_z_c = 6.0; w_f_c = 1.0;
            p_r_c = 0.0; p_z_c = 0.0;
            std::cout << std::fixed << std::setprecision(1);
            std::cout << r_0_c << ' ' << z_0_c << ' ' << f_0_c << '\t';
            std::cout << a_r_c << ' ' << a_z_c << "\t\t";
            std::cout << w_r_c << ' ' << w_z_c << ' ' << w_f_c << '\t';
            std::cout << p_r_c << ' ' << p_z_c << '\n';

            r_0_n = 2.0; z_0_n = 0.0; f_0_n = 0.0;
            a_r_n = 0.5; a_z_n = 0.1;
            w_r_n = 1.0; w_z_n = 4.0; w_f_n = 1.0;
            p_r_n = 0.0; p_z_n = 0.0;
            std::cout << r_0_n << ' ' << z_0_n << ' ' << f_0_n << '\t';
            std::cout << a_r_n << ' ' << a_z_n << "\t\t";
            std::cout << w_r_n << ' ' << w_z_n << ' ' << w_f_n << '\t';
            std::cout << p_r_n << ' ' << p_z_n << '\n';

            // объекты
                icosahedron_center = vec3{0.0, 0.0, 0.0};
                icosahedron_color = vec3{1.0, 0.0, 0.0};
                icosahedron_size = 1.0;
                icosahedron_reflective = 0.3;
                icosahedron_transparency = 0.6;
                icosahedron_bevel_lights_number = 3;

                std::cout << icosahedron_center.x << ' ' << icosahedron_center.y << ' ' << icosahedron_center.z << '\t';
                std::cout << icosahedron_color.x << ' ' << icosahedron_color.y << ' ' << icosahedron_color.z << '\t';
                std::cout << icosahedron_size << '\t' << icosahedron_reflective << '\t' << icosahedron_transparency << '\t' << icosahedron_bevel_lights_number << '\n';

                octahedron_center = vec3{-2.0, 0.0, 0.0};
                octahedron_color = vec3{0.0, 1.0, 0.0};
                octahedron_size = 1.0;
                octahedron_reflective = 0.3;
                octahedron_transparency = 0.6;
                octahedron_bevel_lights_number = 4;

                std::cout << octahedron_center.x << ' ' << octahedron_center.y << ' ' << octahedron_center.z << '\t';
                std::cout << octahedron_color.x << ' ' << octahedron_color.y << ' ' << octahedron_color.z << '\t';
                std::cout << octahedron_size << '\t' << octahedron_reflective << '\t' << octahedron_transparency << '\t' << octahedron_bevel_lights_number << '\n';

                tetrahedron_center = vec3{2.0, 0.0, 0.0};
                tetrahedron_color = vec3{0.0, 0.0, 1.0};
                tetrahedron_size = 1.0;
                tetrahedron_reflective = 0.3;
                tetrahedron_transparency = 0.6;
                tetrahedron_bevel_lights_number = 5;

                std::cout << tetrahedron_center.x << ' ' << tetrahedron_center.y << ' ' << tetrahedron_center.z << '\t';
                std::cout << tetrahedron_color.x << ' ' << tetrahedron_color.y << ' ' << tetrahedron_color.z << '\t';
                std::cout << tetrahedron_size << '\t' << tetrahedron_reflective << '\t' << tetrahedron_transparency << '\t' << tetrahedron_bevel_lights_number << '\n';

            // пол
                floor_a = vec3{-8.0, -8.0, -2.0};
                floor_b = vec3{-8.0, 8.0, -2.0};
                floor_c = vec3{8.0, -8.0, -2.0};
                floor_d = vec3{8.0, 8.0, -2.0};
                floor_texture_path = "images/shrek.data";
                floor_color = vec3{1.0, 1.0, 1.0};
                floor_reflective = 0.3;

                std::cout << floor_a.x << ' ' << floor_a.y << ' ' << floor_a.z << '\t';
                std::cout << floor_b.x << ' ' << floor_b.y << ' ' << floor_b.z << '\t';
                std::cout << floor_c.x << ' ' << floor_c.y << ' ' << floor_c.z << '\t';
                std::cout << floor_d.x << ' ' << floor_d.y << ' ' << floor_d.z << '\t';
                std::cout << floor_texture_path << '\t';
                std::cout << floor_color.x << ' ' << floor_color.y << ' ' << floor_color.z << '\t';
                std::cout << floor_reflective << '\n';
            
            // свет
                scene_lights_number = 2;
                scene_lights = (PointLight*)malloc(scene_lights_number*sizeof(PointLight));

                scene_lights[0] = PointLight{vec3{3.0, 3.0, 3.0}, vec3{1.0, 1.0, 1.0}, 3.0};
                scene_lights[1] = PointLight{vec3{3.0, -3.0, 3.0}, vec3{0.3, 1.0, 0.3}, 0.4};

                std::cout << scene_lights_number << '\n';
                std::cout << scene_lights[0].p.x << ' ' << scene_lights[0].p.y << ' ' << scene_lights[0].p.z << '\t';
                std::cout << scene_lights[0].color.x << ' ' << scene_lights[0].color.y << ' ' << scene_lights[0].color.z << '\n';
                std::cout << scene_lights[1].p.x << ' ' << scene_lights[1].p.y << ' ' << scene_lights[1].p.z << '\t';
                std::cout << scene_lights[1].color.x << ' ' << scene_lights[1].color.y << ' ' << scene_lights[1].color.z << '\n';

            max_bounces = 5;
            ssaa_scale = 1;

            std::cout << max_bounces << ' ' << ssaa_scale << '\n';
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
            std::cin >> a_r_c >> a_z_c;
            std::cin >> w_r_c >> w_z_c >> w_f_c;
            std::cin >> p_r_c >> p_z_c;

            std::cin >> r_0_n >> z_0_n >> f_0_n;
            std::cin >> a_r_n >> a_z_n;
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

            std::cin >> scene_lights_number;
            scene_lights = (PointLight*)malloc(scene_lights_number*sizeof(PointLight));
            for (uint i = 0; i < scene_lights_number; ++i){
                vec3 cur_p;
                std::cin >> cur_p.x, cur_p.y, cur_p.z;
                vec3 cur_color;
                std::cin >> cur_color.x, cur_color.y, cur_color.z;
                scene_lights[i] = PointLight{cur_p, cur_color, 3.0};
            }

            std::cin >> max_bounces;
            std::cin >> ssaa_scale;
        }

        // материалы
            Material grey_bevels_material{vec3{0.1, 0.1, 0.1}, 1.0, -1.0, -1.0};
            Material lights_material{vec3{1.0, 1.0, 1.0}, 3.0, 0.6, -1.0};

        // загрузка объектов
            // данные о полигонах уже в видеопамяти по указателю
            scene_objects_number = 3;
            scene_objects = (Object*)malloc(scene_objects_number*sizeof(Object));

            Material icosahedron_faces_material{icosahedron_color, 1.0f-icosahedron_reflective-icosahedron_transparency, icosahedron_reflective, icosahedron_transparency};
            scene_objects[0] = Object(icosahedron_center, icosahedron_size, "meshes/icosahedron", icosahedron_faces_material, grey_bevels_material, lights_material, icosahedron_bevel_lights_number, 0.02);

            Material octahedron_faces_material{octahedron_color, 1.0f-octahedron_reflective-octahedron_transparency, octahedron_reflective, octahedron_transparency};
            scene_objects[1] = Object(octahedron_center, octahedron_size, "meshes/octahedron", octahedron_faces_material, grey_bevels_material, lights_material, octahedron_bevel_lights_number, 0.02);
            Material tetrahedron_faces_material{tetrahedron_color, 1.0f-tetrahedron_reflective-tetrahedron_transparency, tetrahedron_reflective, tetrahedron_transparency};
            scene_objects[2] = Object(tetrahedron_center, tetrahedron_size, "meshes/tetrahedron", tetrahedron_faces_material, grey_bevels_material, lights_material, tetrahedron_bevel_lights_number, 0.02);
        // загрузка пола
            Material floor_faces_material{floor_color, 1.0f-floor_reflective, floor_reflective, -1.0};
            scene_floor = (Textured_floor*)malloc(sizeof(Textured_floor));
            scene_floor[0] = Textured_floor(floor_a, floor_b, floor_c, floor_d,
                                            floor_texture_path,
                                            floor_faces_material);

        // камера
            Camera cam(vec3{0.0, 0.0, 0.0}, vec3{0.0, 0.0, 0.0}, ssaa_scale*w, ssaa_scale*h, fow);

        // рендер
            for(uint k = 0; k < frame_number; k++) {
                float t = k*(2*M_PI/frame_number);

                vec3 cylindric_p{r_0_c+a_r_c*sin(w_r_c*t+p_r_c),
                                f_0_c+w_f_c*t,
                                z_0_c+a_z_c*sin(w_z_c*t+p_z_c)};

                vec3 cylindric_d{r_0_n+a_r_n*sin(w_r_n*t+p_r_n),
                                f_0_n+w_f_n*t,
                                z_0_n+a_z_n*sin(w_z_n*t+p_z_n)};

                vec3 cam_p{cylindric_p.x*cos(cylindric_p.y),
                        cylindric_p.x*sin(cylindric_p.y),
                        cylindric_p.z};
                vec3 cam_d{cylindric_d.x*cos(cylindric_d.y),
                        cylindric_d.x*sin(cylindric_d.y),
                        cylindric_d.z};

                cam.p = cam_p;
                cam.d = cam_d;

                cudaEvent_t start, stop;
                float time;
                CSC(cudaEventCreate(&start));
                CSC(cudaEventCreate(&stop));
                CSC(cudaEventRecord(start));

                cam.Render(max_bounces);

                CSC(cudaEventRecord(stop));
                CSC(cudaEventSynchronize(stop));
                CSC(cudaEventElapsedTime(&time, start, stop));
                CSC(cudaEventDestroy(start));
                CSC(cudaEventDestroy(stop));

                std::string cur_save_path = pref_save_path+std::to_string(k)+suff_save_path;
                std::cout << std::fixed << std::setprecision(3);
                std::cout << k << '\t' << time << '\t' << rays_count << '\n';
                
                FILE* fp = fopen(cur_save_path.c_str(), "wb");
                    fwrite(&w, sizeof(int), 1, fp);
                    fwrite(&h, sizeof(int), 1, fp);

                    uchar4* res_data = (uchar4*)malloc(w*h*sizeof(uchar4));

                    for (uint y = 0; y < h; ++y){
                        for (uint x = 0; x < w; ++x){

                            vec3 mean_color{0.0, 0.0, 0.0};
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

                            res_data[y*w+x].x = (uint)(clamp(mean_color.x, 0.0f, 255.0f));
                            res_data[y*w+x].y = (uint)(clamp(mean_color.y, 0.0f, 255.0f));
                            res_data[y*w+x].z = (uint)(clamp(mean_color.z, 0.0f, 255.0f));
                            res_data[y*w+x].w = 255;
                        }
                    }

                    fwrite(res_data, sizeof(uchar4), w*h, fp);
                    free(res_data);
                    free(cam.data);
                    fclose(fp);
            }

        // завершение работы
            cudaFree(cam.dev_scene_lights);
            cudaFree(cam.dev_scene_objects);

            free(scene_lights);
            for (int i = 0; i < scene_objects_number; ++i){
                cudaFree(scene_objects[i].faces);
                cudaFree(scene_objects[i].bevels);
                cudaFree(scene_objects[i].caps);
                cudaFree(scene_objects[i].dev_lights);
            }
            free(scene_objects);
    }
    else {
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

            // w = 200;
            // h = 300;
            
            w = 600;
            h = 600;

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
                std::cin >> a_r_c >> a_z_c;
                std::cin >> w_r_c >> w_z_c >> w_f_c;
                std::cin >> p_r_c >> p_z_c;

                std::cin >> r_0_n >> z_0_n >> f_0_n;
                std::cin >> a_r_n >> a_z_n;
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

                                res_data[y*w+x].x = (uint)(clamp(mean_color.x, 0.0f, 255.0f));
                                res_data[y*w+x].y = (uint)(clamp(mean_color.y, 0.0f, 255.0f));
                                res_data[y*w+x].z = (uint)(clamp(mean_color.z, 0.0f, 255.0f));
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
    }

	return 0;
}