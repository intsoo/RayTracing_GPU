/* 
 * ===================================================
 *
 *     Project Name:  RT_GPU
 *        File Name:  main.cu
 *      Description:  
 *					  [UPDATES]
 *					    1) All classes are replaced with structures.
 *					    2) Sturctures are passed by pointers(*) instead of by references(&) in __global__ functions (not in __device__ functions).
 *          Created:  2022/11/08
 * 
 * ===================================================
 */

/*
 <GPU Code>
 1) GPU Kernel 1: A single thread creates the 3D world and camera. Then it constructs a BVH.
 2) GPU Kernel 2: Multiple threads render the output image. (Each thread handles one pixel.)
 3) CPU: Copy the output image data to CPU array and create a ppm image file.
 */


// Preprocessors

#include <curand_kernel.h>

#include "material.h"
#include "sphere.h"
#include "camera.h"
#include "bvh.h"

#include "mkPpm.h"
#include "mkCuda.h"
#include "mkClockMeasure.h"

#include <iostream>

#define RND (curand_uniform(&local_randState))


// Variables
//Stack *d_stack;
curandState *d_randState;
const double infinity = std::numeric_limits<float>::infinity();


// Functions
// 0. (NVIDIA Code) rand_init: initializes the CUDA Random Generator State.
__global__ void rand_init(curandState *randState) 
{
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		curand_init(1984, 0, 0, randState);
	}
}


// 0. computeNodeNum: Compute the number of nodes in BVH.
int computeNodeNum(int objectNum) 
{
	return objectNum * 2 - 1;  // n objects => 2n-1 nodes
}

__device__ float gpuClamp(float num, float min, float max) 
{
	float result = 0;

	if(num < min)  result = min;
	else if(num > max) result = max;
	else result = num;

	return result;
}

__device__ int get_tree_height(int obj_num, int branch_factor) 
{
	int height = 1;  // tree height
	int pow_of_two = 1;  // Starts from 2^0.

	while(pow_of_two < obj_num) {
		pow_of_two *= 2;
		height++;
	}
	return height;
}

int cpu_get_tree_height(int obj_num, int branch_factor) 
{
	int height = 1;  // tree height
	int pow_of_two = 1;  // Starts from 2^0.

	while(pow_of_two < obj_num) {
		pow_of_two *= 2;
		height++;;
		printf("%d < %d <= %d: height = %d\n", pow_of_two/2, obj_num, pow_of_two, height);
	}
	return height;
}

// 1. createWorld: Implement the 3D World(object list) and the BVH.
//__global__ void createWorld(bvh_node **d_bvh, int num, hittable **d_objects, hittable **d_world, camera **d_camera, bvh_node **d_nodes, int width, int height, curandState *randState) {
__global__ void createWorld(int width, int height, int n, Camera *d_camera, Sphere *d_obj_list, Material *d_materials, BvhNode *d_bvh, curandState *randState)
{
	// A single (first) thread creates the world.
	if((threadIdx.x || blockIdx.x) != 0)	return;

	// CUDA Random Number Generator State
    curandState local_randState = *randState;

	int obj_num = (n+n)*(n+n)+4;
	float t0 = 0.0001f, t1 = 1.0f;

	// 1. Create a camera.
	Vec3 lookfrom{13,2,3};
    Vec3 lookat{0,0,0};
	Vec3 vup{0,1,0};
	float vfov = 20.0f;
    //float focusDistance = (lookfrom-lookat).length();  // 10.0f
	float focusDistance = 10.0f;  // 10.0f
    float aperture = 0.1f;
	Camera cam;
	
	cam.init(lookfrom, lookat, vup, vfov, (float)width/height, aperture, focusDistance, t0, t1);
	*d_camera = cam;
	
	// 2. Fill the material array with materials and fill the object array with spheres.
	int count = 0;  // the # of objects
	
	// Ground Sphere
	d_materials[count] = Material{'l', 0, Vec3{0.5f, 0.5f, 0.5f}};  // type, c, albedo
	d_obj_list[count] = Sphere{count, Vec3{0.0f,-999.0f,0.0f}, 999, count};  
	count++;

	// Small Spheres
	for(int x = -n; x < n; x++) {  
		for(int z = -n; z < n; z++) {
			float matType = RND;  // Chooses the type of the current material.
			//Vec3 center{x+RND,0.2f,z+RND};  // center of the current material.
			Vec3 center{(float)x,0.2f,(float)z};  // center of the current material.
	    	
			if(matType < 0.8f) {  // diffuse material
				d_materials[count] = Material{'l', 0, Vec3{RND*RND, RND*RND, RND*RND}};  // type, c, albedo
                d_obj_list[count] = Sphere{count, center, 0.2f, count};  // idx, center, radius, matIdx
            }
			else if(matType < 0.95f) {  // metal
				d_materials[count] = Material{'m', 0.5f*RND, Vec3{0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)}};
                d_obj_list[count] = Sphere{count, center, 0.2f, count};
			}
			else {  // dielectric
				d_materials[count] = Material{'d', 1.5f, Vec3{0}};
                d_obj_list[count] = Sphere{count, center, 0.2f, count};				
			}
			count++;
		}
	}

	// Giant materials
	d_materials[count] = Material{'l', 0, Vec3{0.4f, 0.2f, 0.1f}};  // diffuse
	d_obj_list[count] = Sphere{count, Vec3{-4.0f,1.3f,0.0f}, 1.3f, count};   // back
	count++;

	d_materials[count] = Material{'d', 1.5f, Vec3{0, 0, 0}};  // dielectric
	d_obj_list[count] = Sphere{count, Vec3{0.0f,1.3f,0.0f}, 1.3f, count};  // middle
	count++;

	d_materials[count] = Material{'m', 0.0f, Vec3{0.7f, 0.6f, 0.5f}};  // metal
	d_obj_list[count] = Sphere{count, Vec3{4.0f, 1.3f, 0.0f}, 1.3f, count};  // front
	count++;


	//////////////////////////////////////////////////////////////////////////////////////
	// DEBUGGING
	// 1) Check the material list.
	for(int i=0; i<obj_num; i++) {
		d_materials[i].print_info(i);
	}
	printf("\n");

	// 2) Check the object list.
	for(int i=0; i<obj_num; i++) {
		d_obj_list[i].print_info();
	}
	printf("\n");
	//////////////////////////////////////////////////////////////////////////////////////

   	*randState = local_randState;

	// 3. Construct a BVH.
	generate_bvh(d_bvh, d_obj_list, obj_num, t0, t1);
	// DEBUGGING
	print_bvh(d_bvh, d_obj_list, obj_num);
}



// 2. ray_color: Calculate color of the current ray intersection point.
__device__ Vec3 ray_color(int global_idx, Ray &r, int obj_num, Sphere *obj_list, Material *mat_list, BvhNode *bvh, int depth, BvhNode **stack, curandState *local_randState)		
{
	HitRecord rec;
	Ray cur_ray = r;  // current ray
	Vec3 cur_attenuation = Vec3{1.0f, 1.0f, 1.0f};  // current attenuation
	const int STACK_SIZE = get_tree_height(obj_num, 2) - 1;  // binary tree: branch_factor = 2

	// Ray Tracing
	for(int i = 0; i < depth; i++) {  // Limit the number of child ray.
		if(search_bvh(bvh, obj_list, obj_num, cur_ray, 0.001f, infinity, rec, local_randState))  // Per-ray BVH traversal using stack
//		if(bvh_traversal(stack, bvh, STACK_SIZE, obj_list, cur_ray, 0.001f, infinity, rec))  // BVH Traversal using an array for all nodes
//		if(list_hit(obj_list, obj_num, cur_ray, 0.001f, infinity, rec))  // without BVH
		{ 	
			//printf("HIT\n");
			Ray scattered;
			Vec3 attenuation;
			Material *cur_obj_mat;

			// If hit at the nearest point, generate a child ray.
			int mat_idx = (obj_list[rec.sphereIdx]).matIdx;  // int mat_idx = rec.sphereIdx (구의 고유 인덱스 = 재질의 고유 인덱스)
			if(mat_idx >= 0 && mat_idx < obj_num)  // correct index
			{  
				cur_obj_mat = &mat_list[mat_idx];

				// Compute the intersection (ray attenuation & child ray direction)
				if(cur_obj_mat->scatter(cur_ray, rec, attenuation, scattered, local_randState))
				{
					cur_ray = scattered;  // child ray
					cur_attenuation *=  attenuation;  // reducing the light intensity = darkening the pixel color
				}
				else 
				{
					printf("SCATTER FUNCTION ERROR\n");
					return Vec3{1.0f, 1.0f, 1.0f};
				}
			}
			else  // index out of range
			{  
				printf("MATERIAL INDEX IS OUT OF RANGE\n");
			}
		}
		// If ray hits no object: background(sunlight)
		else 
		{
			Vec3 unit_direction = unit_vector(cur_ray.direction);

			float t = 0.5f * (unit_direction.y() + 1.0f);  // gradation based on y value
			//float t = 1;
			
			Vec3 c = (1.0f - t) * Vec3{1.0f, 1.0f, 1.0f} + t * Vec3{0.5f, 0.7f, 1.0f};

			return cur_attenuation * c;
		}
	}
	return Vec3{0.0f, 0.0f, 0.0f}; // too many hits => considered as no light is reaching
}


// 3. deleteWorld: Delete the 3D world.
__global__ void deleteWorld(int n, Camera *d_camera, Material *d_mat_list, Sphere *d_obj_list, BvhNode *d_bvh) 
{
	//int obj_num = (n+n)*(n+n)+4;
	//int node_num = 2*obj_num-1;

	// Delete the camera;
	delete d_camera;
    
	// Delete the material list.
    delete d_mat_list;

	// Delete the object list.
    delete d_obj_list;

	// Delete the BVH.
    delete d_bvh;
}


// 4. render_init: Generate random seeds for threads in kernel 'render'.
__global__ void render_init(int max_x, int max_y, curandState *randState) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if((i >= max_x) || (j >= max_y)) return;
    int pixel_idx = j*max_x + i;

    // Original: Each thread gets same seed, a different sequence number, no offset
    // curand_init(1984, pixelIdx, 0, &randState[pixelIdx]);
    // BUGFIX, see Issue#2: Each thread gets different seed, same sequence for
    // performance improvement of about 2x!
    curand_init(1984+pixel_idx, 0, 0, &randState[pixel_idx]);
}


// 5. render: Render an image via ray tracing.
__global__ void render(int width, int height, int samples_per_pix, int max_depth, int obj_num, Sphere *obj_list, Material *mat_list, BvhNode *bvh, Camera *cam, unsigned char *out_img_array, BvhNode **stack, curandState *randState) 
{
	// index of the current thread
	int x_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int y_idx = threadIdx.y + blockIdx.y * blockDim.y;
    	
	if((x_idx < width) && (y_idx < height)) {  // If the current thread is within pixel range
		int global_idx = (y_idx * width + x_idx) * 3;  // global index of the current thread

		float r, g, b;
		curandState local_randState = randState[global_idx/3];
		Vec3 pixel_color{0.0f, 0.0f, 0.0f};

		// Antialiasing
		for(int s = 0; s < samples_per_pix; s++) 
		{
			float u = float(x_idx + curand_uniform(&local_randState)) / float(width-1);	
			float v = float((height - y_idx - 1) + curand_uniform(&local_randState)) / float(height-1);
			//float v = float(y_idx + curand_uniform(&local_randState)) / float(width);

			Ray cur_ray = cam->get_ray(u, v, &local_randState);

			// DEBUGGING
			/*
			printf("RENDERING PIXEL(%d, %d): Ray Direction = (%lf, %lf, %lf)\n", 
			x_idx, y_idx, 
			(cur_ray.direction).e[0], (cur_ray.direction).e[1], (cur_ray.direction).e[2]);
			*/

			pixel_color += ray_color(global_idx, cur_ray, obj_num, obj_list, mat_list, bvh, max_depth, stack, &local_randState);
			
			// DEBUGGING: CHECK THE COLOR OF THE CURRENT RAY
			/*
			printf("THD%3d PIXEL(%3d,%3d) SAMPLE%3d: %.3lf %.3lf %.3lf\n", 
			global_idx/3, x_idx, y_idx, s, pixel_color.e[0], pixel_color.e[1], pixel_color.e[2]);
			*/
		}

		//error:	randState[globalIdx] = local_randState;

		r = pixel_color.e[0];
		g = pixel_color.e[1];
		b = pixel_color.e[2];

		float scale = 1.0f / samples_per_pix; 
		r = sqrt(scale * r);
		g = sqrt(scale * g);
		b = sqrt(scale * b);

		// Color of the Current Pixel
       	// Mapping: 0.0~1.0 (float) -> 0~256 (unsigned char)
		// 256 -> 255.999f (prevent overflow)
		out_img_array[global_idx] = (unsigned char)(gpuClamp(r, 0.0f, 0.999f) * 256);  // (unsigned char)r * 256 => always 0
		out_img_array[global_idx+1] = (unsigned char)(gpuClamp(g, 0.0f, 0.999f) * 256);  // RT18: 256 -> 255.999f
		out_img_array[global_idx+2] = (unsigned char)(gpuClamp(b, 0.0f, 0.999f) * 256);

		// DEBUGGING: PRINT THE OUTPUT IMAGE
		/*
		printf("PIXEL(%3d,%3d): %3d %3d %3d\n", 
		x_idx, y_idx, out_img_array[global_idx], out_img_array[global_idx+1], out_img_array[global_idx+2]);
		*/
	}
}

// 5. render: Render an image via ray tracing.
__global__ void render_iter(int width, int height, int samples_per_pix, int max_depth, int obj_num, Sphere *obj_list, Material *mat_list, BvhNode *bvh, Camera *cam, unsigned char *out_img_array, BvhNode **stack, curandState *randState) 
{    	
	for(int y = 0; y < height; y++) {  // If the current thread is within pixel range
		for(int x = 0; x < width; x++){
			int array_idx = (y * width + x) * 3;  // global index of the current thread

			float r, g, b;
			curandState local_randState = randState[array_idx/3];
			Vec3 pixel_color{0.0f, 0.0f, 0.0f};

			// Antialiasing
			for(int s = 0; s < samples_per_pix; s++) 
			{
				float u = float(x + curand_uniform(&local_randState)) / float(width-1);	
				float v = float((height - y - 1) + curand_uniform(&local_randState)) / float(height-1);
				//float v = float(y_idx + curand_uniform(&local_randState)) / float(width);

				Ray cur_ray = cam->get_ray(u, v, &local_randState);

				// DEBUGGING
				/*
				printf("RENDERING PIXEL(%d, %d): Ray Direction = (%lf, %lf, %lf)\n", 
				x, y, 
				(cur_ray.direction).e[0], (cur_ray.direction).e[1], (cur_ray.direction).e[2]);
				*/

				pixel_color += ray_color(array_idx, cur_ray, obj_num, obj_list, mat_list, bvh, max_depth, stack, &local_randState);
				
				// DEBUGGING: CHECK THE COLOR OF THE CURRENT RAY
				/*
				printf("THD%3d PIXEL(%3d,%3d) SAMPLE%3d: %.3lf %.3lf %.3lf\n", 
				array_idx/3, x, y, s, pixel_color.e[0], pixel_color.e[1], pixel_color.e[2]);
				*/
			}

			//error:	randState[array_idx] = local_randState;

			r = pixel_color.e[0];
			g = pixel_color.e[1];
			b = pixel_color.e[2];

			float scale = 1.0f / samples_per_pix; 
			r = sqrt(scale * r);
			g = sqrt(scale * g);
			b = sqrt(scale * b);

			// Color of the Current Pixel
			// Mapping: 0.0~1.0 (float) -> 0~256 (unsigned char)
			// 256 -> 255.999f (prevent overflow)
			out_img_array[array_idx] = (unsigned char)(gpuClamp(r, 0.0f, 0.999f) * 256);  // (unsigned char)r * 256 => always 0
			out_img_array[array_idx+1] = (unsigned char)(gpuClamp(g, 0.0f, 0.999f) * 256);  // RT18: 256 -> 255.999f
			out_img_array[array_idx+2] = (unsigned char)(gpuClamp(b, 0.0f, 0.999f) * 256);

			// DEBUGGING: PRINT THE OUTPUT IMAGE
			/*
			printf("PIXEL(%3d,%3d): %3d %3d %3d\n", 
			x, y, out_img_array[array_idx], out_img_array[array_idx+1], out_img_array[array_idx+2]);
			*/
		}
	}
}

// 6. main
int main() {
	cudaError_t err;

	// Execution Time
	mkClockMeasure *ckGpu = new mkClockMeasure("GPU CODE");
	ckGpu->clockReset();


    // Image
	auto aspect_ratio = 16.0 / 9.0;
	int img_width = 400;  // 400
	int img_height = img_width / aspect_ratio;
    int samples_per_pixel = 1;  // 100  
	const int max_depth = 50;  // 50
	int pixel_num = img_width * img_height;


	// Grid Settings
	const int t_size = 16;  // 16  RT18: 왜 32 이상으로 키우면 잘 동작 X? => 최대 TB 사이즈가 256인 듯!
	dim3 tb_size(t_size, t_size, 1);  // 2D TB
	dim3 grid_size((int)ceil((float)img_width/t_size), (int)ceil((float)img_height/t_size), 1);


	// Host & Device Arrays
	// 1. Size
	int n = 0;   // decides the number of objects(spheres) in the world.
	size_t obj_num = (n+n)*(n+n)+1+3;  // the # of objects (= small + ground + giant materials)
	size_t node_num = computeNodeNum(obj_num);  // the # of nodes (= 2n-1, n: the # of objects)
	size_t stack_size = cpu_get_tree_height(obj_num, 2) - 1;
	size_t img_size = sizeof(unsigned char) * img_width * img_height * 3;  // size of image arrays

	// 2. Pointers
	unsigned char *h_img, *d_img;  // output image arrays
	Camera *d_camera;  // camera
	Material *d_mat_list;  
	Sphere *d_obj_list;  // object list (= world)
	BvhNode *d_bvh;  // BVH
	BvhNode **d_stack;  // stack

	// 3. Memory Allocation
	// 3.1. Image Arrays
	h_img = (unsigned char*)malloc(img_size);  // host array
	err = cudaMalloc((void **)&d_img, img_size);  // device array
	checkCudaError(err); 

	// 3.2. Camera
	err = cudaMalloc((void **)&d_camera, sizeof(Camera));
	checkCudaError(err); 

	// 3.3. Material List
	err = cudaMalloc((void **)&d_mat_list, obj_num * sizeof(Material));  // three types of material -> unique material for each object
	checkCudaError(err); 

	// 3.4. Object List
	err = cudaMalloc((void **)&d_obj_list, obj_num * sizeof(Sphere));
	checkCudaError(err); 

	// 3.5. BVH
	err = cudaMalloc((void **)&d_bvh, node_num * sizeof(BvhNode));
	checkCudaError(err); 

	// 3.6. Random Seeds
	// Allocate random state(seed) for each thread.    	
	curandState *d_randState;
	err = cudaMalloc((void **)&d_randState, pixel_num*sizeof(curandState));  // for render() 
	checkCudaErrors(err);

	curandState *d_randState2;
    err = cudaMalloc((void **)&d_randState2, 1*sizeof(curandState));  // for createWorld()
	checkCudaErrors(err);

	// 3.7. Stack
	err = cudaMalloc((void **)&d_stack, stack_size * sizeof(BvhNode*));
	checkCudaError(err); 

	//////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////
	// Kernel Executions

    // 1. rand_init: Initialize the second random-state for world creation.
    rand_init<<<1,1>>>(d_randState2);
    err = cudaGetLastError();
	checkCudaErrors(err);

	err = cudaDeviceSynchronize();
	checkCudaErrors(err);

	// 2. createWorld: Create world(object list) and BVH.
	ckGpu->clockReset();  // Measure the kernel execution time.	
	ckGpu->clockResume();

    createWorld<<<1,1>>>(img_width, img_height, n, d_camera, d_obj_list, d_mat_list, d_bvh, d_randState2);
	err=cudaDeviceSynchronize();
	checkCudaErrors(err);

	ckGpu->clockPause();
	ckGpu->clockPrint();

    // 3. render_init: Initialize the first random-states for rendering.
	render_init<<<tb_size, t_size>>>(img_width, img_height, d_randState);
	err=cudaDeviceSynchronize();
	checkCudaErrors(err);

	// 4. render: Render an output image.
	ckGpu->clockReset();
	ckGpu->clockResume();

//	printf("Object Num: %d, Tree Height: %d\n", 7, cpu_get_tree_height(7, 2));


// 	render<<<grid_size, tb_size>>>(img_width, img_height, samples_per_pixel, max_depth, obj_num, d_obj_list, d_mat_list, d_bvh, d_camera, d_img, d_stack, d_randState);
   render_iter<<<1, 1>>>(img_width, img_height, samples_per_pixel, max_depth, obj_num, d_obj_list, d_mat_list, d_bvh, d_camera, d_img, d_stack, d_randState);
	err=cudaDeviceSynchronize();
	checkCudaErrors(err);

	ckGpu->clockPause();
	ckGpu->clockPrint();
	
	//////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////

	// Store the output image data in a PPM image file.
	err = cudaMemcpy(h_img, d_img, img_size, cudaMemcpyDeviceToHost);
	checkCudaError(err);
    ppmSave("img.ppm", h_img, img_width, img_height);

/*
	// DEBUGGING: PRINT THe OUTPUT IMAGE
	printf("----------------------------------------- FINAL IMAGE -----------------------------------------\n");
	for(int i = 0; i < img_height; i++) {
		for(int j = 0; j < img_width; j++) {
			int idx = (i * img_width + j) * 3;
			printf("\t(%d,%d) %3d %3d %3d", i, j, h_img[idx], h_img[idx+1], h_img[idx+2]);
		}
		printf("\n");
	}
	printf("-----------------------------------------------------------------------------------------------\n");
*/
    	
	return 0;
}
