/* FINISHED
 * ===================================================
 *
 *     Project Name:  RT_GPU
 *        File Name:  camera.h
 *      Description:  
 *					  [UPDATES]
 *					    1) All classes are replaced with structures.
 *					    2) Sturctures are passed by pointers(*) instead of by references(&) in __global__ functions (not in __device__ functions).
 *          Created:  2022/11/08
 * 
 * ===================================================
 */


// Preprocessors
#ifndef CAMERA_H
#define CAMERA_H


// Variables
const float pi = 3.1415926535897932385;


// Functions
// 1. degrees_to_radians: Convert angle in degree to radian.
__device__ inline float degrees_to_radians(float degrees) 
{
       	return degrees * pi / 180.0f;
}


// Structures
typedef struct _Camera {
	Vec3 origin;
	Vec3 lower_left_corner;
	Vec3 horizontal;
	Vec3 vertical;
	Vec3 u;
	Vec3 v;
	Vec3 w;
	float lens_radius;
	float time0;
	float time1;

	// 2. init: Initialize a camera
	__device__ void init(const Vec3 &lookfrom, const Vec3 &lookat, const Vec3 &vup, 
		float vfov, float aspect_ratio, float aperture, float focus_dist, float t0, float t1 )
	{
		float theta = degrees_to_radians(vfov);
		float h = tan(theta/2.0f);
		float viewport_height = 2.0f * h;
		float viewport_width = aspect_ratio * viewport_height;		

		w = unit_vector(lookfrom - lookat);
		u = unit_vector(cross(vup, w));
		v = cross(w, u);

		origin = lookfrom;
		horizontal = focus_dist * viewport_width * u;
		vertical = focus_dist * viewport_height * v;
		lower_left_corner = lookfrom - horizontal/2 - vertical/2 - focus_dist*w;
		lens_radius = aperture / 2;
		time0 = t0;
		time1 = t1;
	}

	// 3. get_ray: Generate a ray from the camera.
	__device__ Ray get_ray(float s, float t, curandState *local_randState)
	{
		Vec3 rd = lens_radius * random_in_unit_disk(local_randState);
		Vec3 offset = u * rd.e[0] + v * rd.e[1];
		
		return Ray{
			origin + offset,
			lower_left_corner + s*horizontal + t*vertical - origin - offset,
			random_float(time0, time1, local_randState)};
	}

} Camera;

/*
// 2. init_camera: Initialize a camera
__device__ void init_camera(Camera &cam, const Vec3 &lookfrom, const Vec3 &lookat, const Vec3 &vup, 
	float vfov, float aspect_ratio, float aperture, float focus_dist, float t0, float t1 )
{
	float theta = degrees_to_radians(vfov);
	float h = tan(theta/2.0f);
    float viewport_height = 2.0f * h;
    float viewport_width = aspect_ratio * viewport_height;		

   	cam.w = unit_vector(lookfrom - lookat);
    cam.u = unit_vector(cross(vup, cam.w));
    cam.v = cross(cam.w, cam.u);

    cam.origin = lookfrom;
	cam.horizontal = focus_dist * viewport_width * cam.u;
    cam.vertical = focus_dist * viewport_height * cam.v;
	cam.lower_left_corner = lookfrom - cam.horizontal/2 - cam.vertical/2 - focus_dist*cam.w;
	cam.lens_radius = aperture / 2;
	cam.time0 = t0;
    cam.time1 = t1;
}
*/

/*
// 3. get_ray: Generate a ray from the camera.
 __device__ Ray get_ray(const Camera &cam, float s, float t, curandState *local_randState)
{
	Vec3 rd = cam.lens_radius * random_in_unit_disk(local_randState);
	Vec3 offset = cam.u * rd.e[0] + cam.v * rd.e[1];
    
	return Ray{
		cam.origin + offset,
		lower_left_corner + s*cam.horizontal + t*cam.vertical - cam.origin - offset,
		random_float(cam.time0, cam.time1, local_randState)};
}
*/


#endif