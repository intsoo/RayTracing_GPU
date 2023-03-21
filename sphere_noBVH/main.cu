#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>

#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hittable_list.h"
#include "camera.h"
#include "material.h"
#include "mkCuda.h"
#include "mkPpm.h"

#define RND (curand_uniform(&local_rand_state))  // Choose random float in (0.0 ~ 1.0]

using namespace std;

// Generate Random Seed for createScene()
__global__ void randInit1(curandState *rand_state) 
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        curand_init(1984, 0, 0, rand_state);
    }
}

// Generate Random Seed for render()
__global__ void randInit2(int max_x, int max_y, curandState *rand_state) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    // Original: Each thread gets same seed, a different sequence number, no offset
    // curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
    // BUGFIX, see Issue#2: Each thread gets different seed, same sequence for
    // performance improvement of about 2x!
    curand_init(1984+pixel_index, 0, 0, &rand_state[pixel_index]);
}


// Create a scnene(3D world) including a camera
__global__ void createScene(
    hittable **d_list, 
    hittable **d_world, 
    camera **d_camera, 
    int n_obj,
    int nx, int ny, 
    curandState *rand_state) 
{
    if (!(threadIdx.x == 0 && blockIdx.x == 0)) return;

    curandState local_rand_state = *rand_state;

    material* ground_material = new lambertian(vec3(0.5f, 0.5f, 0.5f));
    d_list[0] = new sphere(vec3(0, -1000.0, -1), 1000, ground_material);
    
    int count = 1;  // number of objects
    for(int a = -n_obj; a < n_obj; a++) 
    {
        for(int b = -n_obj; b < n_obj; b++) 
        {
            float choose_mat = RND;  // Choose material randomly
            material* sphere_material;
            vec3 center(a+RND, 0.2, b+RND);  // Choose center of the sphere

            if ((center - vec3(4, 0.2f, 0)).length() <= 0.9f) 
            {
                b--;
                continue;
            }

            // Check the material type
            // 1) Lambertian
            if(choose_mat < 0.8f)
            {
                sphere_material = new lambertian(vec3(RND*RND, RND*RND, RND*RND));
                d_list[count++] = new sphere(center, 0.2f, sphere_material);
            }
            // 2) Metal
            else if(choose_mat < 0.95f) 
            {
                sphere_material = new metal(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.5f*RND);
                d_list[count++] = new sphere(center, 0.2f, sphere_material);
            }
            // 3) Dielectric
            else 
            {
                sphere_material = new dielectric(1.5f);
                d_list[count++] = new sphere(center, 0.2f, sphere_material);
            }
        }
    }
    material* material1 = new dielectric(1.5f);
    material* material2 = new lambertian(vec3(0.4f, 0.2f, 0.1f));
    material* material3 = new metal(vec3(0.7f, 0.6f, 0.5f), 0);

    d_list[count++] = new sphere(vec3(0, 1, 0), 1.0f, material1);
    d_list[count++] = new sphere(vec3(-4, 1, 0), 1.0f, material2);
    d_list[count++] = new sphere(vec3(4, 1, 0), 1.0f, material3);
    
    *rand_state = local_rand_state;
    *d_world  = new hittable_list(d_list, (2*n_obj)*(2*n_obj)+1+3);

    // Create a scnene camera
    vec3 lookfrom(13, 2, 3);
    vec3 lookat(0, 0, 0);
    float dist_to_focus = (lookfrom-lookat).length();  // 10.0f
    float aperture = 0.1f;
    *d_camera = new camera(lookfrom,lookat, vec3(0, 1, 0), 30.0f, float(nx)/float(ny), aperture, dist_to_focus);
}


// Compute the current pixel color via ray tracing
__device__ vec3 computeRayColor(
    int depth, 
    const ray& r, 
    hittable **world, 
    curandState *local_rand_state) 
{
   // Iterative Ray Tracing
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0f, 1.0f, 1.0f);
   
    for(int i = 0; i < depth; i++) 
    {
        hit_record rec;

        // If any object is hit
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) 
        {
            ray scattered;  // direction of secondary(child) ray 
            vec3 attenuation;  // attenuation of color
            
            // Generate a secondary(child) ray by the object's material
            if(rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) 
            {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else 
            {
                return vec3(0, 0, 0);
            }
        }
        // no hit
        else 
        {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);  // color gradation in y(vertical) direction
            vec3 c = (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
           
            return cur_attenuation * c;
        }
    }
    return vec3(0, 0, 0); // exceeded recursion
}


__global__ void render(
    unsigned char* d_image,
    int image_width, int image_height, 
    int samples_per_pixel, 
    int depth,
    camera **cam, 
    hittable **world, 
    curandState *rand_state) 
{
    // Compute pixel index
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= image_width) || (j >= image_height)) return;  // Check index out of range
    int pixel_index = j * image_width + i;
    int global_index = (j * image_width + i) * 3;

    // Get random seed for the current pixel
    curandState local_rand_state = rand_state[pixel_index];

    // Compute pixel color
    vec3 pixel_color(0,0,0);  // current pixel color
    for(int s = 0; s < samples_per_pixel; s++) 
    {
        float u = float(i + curand_uniform(&local_rand_state)) / float(image_width-1);
        float v = float((image_height-j-1) + curand_uniform(&local_rand_state)) / float(image_height-1);

        //float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        //float v = float(j + curand_uniform(&local_rand_state)) / float(max_y); 
     
        ray cur_ray = (*cam)->get_ray(u, v, &local_rand_state);
        pixel_color += computeRayColor(depth, cur_ray, world, &local_rand_state);
    }

    rand_state[pixel_index] = local_rand_state;

    // antialiasing
    pixel_color /= float(samples_per_pixel);
    pixel_color[0] = sqrt(pixel_color[0]);
    pixel_color[1] = sqrt(pixel_color[1]);
    pixel_color[2] = sqrt(pixel_color[2]);
   
    d_image[global_index] = (unsigned char)(255.99f * pixel_color[0]);
    d_image[global_index + 1] = (unsigned char)(255.99f * pixel_color[1]);
    d_image[global_index + 2] = (unsigned char)(255.99f * pixel_color[2]);

}

__global__ void eraseScene(
    int n_obj, 
    hittable **d_list, 
    hittable **d_world, 
    camera **d_camera) 
{
    int num_obj = (2*n_obj) * (2*n_obj) + 1 + 3;
    for(int i = 0; i < num_obj; i++) 
    {
        delete ((sphere *)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}


int main() {
    // Image Setting
	float aspect_ratio = 16.0f / 9.0f;
    int image_width = 400;
	int image_height = image_width / aspect_ratio;
    int samples_per_pixel = 1;
    int max_depth = 50;
   
    // Thread Block Setting
    int tb_width = 16;
    int tb_height = 16;
    dim3 tb(image_width/tb_width + 1, image_height/tb_height + 1);  // ceiling
    dim3 thd(tb_width, tb_height);

    // Print Rendering Information
    cout << "[Image] " << image_width << "x" << image_height << endl;
    cout << "  - # of samples per pixel: " << samples_per_pixel << endl;
    cout << "  - # of samples per pixel: " << samples_per_pixel << endl;
    cout << "[Thread Block] " << tb_width << "x" << tb_height << endl;


	// Output Image Arrays
    int num_pixels = image_width * image_height;
	size_t image_size = sizeof(unsigned char) * num_pixels * 3;
	unsigned char *h_image, *d_image;

	// 1) Host Array
	h_image = (unsigned char*)malloc(image_size);

	// 2) Device Array
	cudaError_t err = cudaMalloc((void **)&d_image, image_size);  // Allocate memory for output image array in GPU.
	checkCudaError(err);

    // Allocate random seeds
    curandState *d_rand_state1;
    curandState *d_rand_state2;
    err = cudaMalloc((void **)&d_rand_state1, 1 * sizeof(curandState));
    checkCudaErrors(err);
    err = cudaMalloc((void **)&d_rand_state2, num_pixels * sizeof(curandState));
    checkCudaErrors(err);
   
    // Initialize random seeds for createScene()
    randInit1<<<1,1>>>(d_rand_state1);
    err = cudaDeviceSynchronize();
    checkCudaErrors(err);
    
    // Create new scene(world)
    int n_obj = 0;
    int num_obj = (2*n_obj) * (2*n_obj) + 1 + 3;
    hittable **d_list;
    hittable **d_world;
    camera **d_camera;
    
    // #FIX
    err = cudaMalloc((void **)&d_list, num_obj*sizeof(hittable *));   
    checkCudaErrors(err);

    err = cudaMalloc((void **)&d_world, sizeof(hittable *));
    checkCudaErrors(err);

    err = cudaMalloc((void **)&d_camera, sizeof(camera *));
    checkCudaErrors(err);

    createScene<<<1,1>>>(d_list, d_world, d_camera, n_obj, image_width, image_height, d_rand_state1);
    err = cudaDeviceSynchronize();
    checkCudaErrors(err);


    // Initialize random seeds for render()
    randInit2<<<tb, thd>>>(image_width, image_height, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
   
    // Render
    render<<<tb, thd>>>(d_image, image_width, image_height, samples_per_pixel, max_depth, d_camera, d_world, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

	// Store the output image data in a PPM image file.
	err = cudaMemcpy(h_image, d_image, image_size, cudaMemcpyDeviceToHost);
	checkCudaError(err);
    ppmSave("img.ppm", h_image, image_width, image_height);


    // Free memory
    eraseScene<<<1,1>>>(n_obj, d_list, d_world, d_camera);
    err = cudaFree(d_camera);
    err = cudaFree(d_world);
    err = cudaFree(d_list);
    err = cudaFree(d_rand_state1);
    err = cudaFree(d_rand_state2);
    err = cudaFree(d_image);
    cudaDeviceReset();
}

