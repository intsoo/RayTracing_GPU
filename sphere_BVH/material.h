/* FINISHED
 * ===================================================
 *
 *     Project Name:  RT_GPU
 *        File Name:  material.h
 *      Description:  
 *					  [UPDATES]
 *					    1) All classes are replaced with structures.
 *					    2) Sturctures are passed by pointers(*) instead of by references(&) in __global__ functions (not in __device__ functions).
 *          Created:  2022/11/08
 * 
 * ===================================================
 */


// Preprocessors
#ifndef MATERIAL_H
#define MATERIAL_H

#include "sphere.h"


// Structures
typedef struct _Material {
	char type;  // lambertian: 'l', metal: 'm', dielectric: 'd'
	float c;  // material constant (diffuse: none(0), metal: fuzziness, dielectric: refract index)
	Vec3 albedo;  // reflectance ratio

	// 1. reflectance: Compute the reflectance (dielectrics)
	__device__ float reflectance(float cosine, float ref_idx) 
	{
		// Schlick's approximation for reflectance & refractance
		float r0 = (1-ref_idx) / (1+ref_idx);
		r0 = r0*r0;
		return r0 + (1-r0)*pow((1 - cosine), 5);
	}

	// 2. scatter: Compute how the ray is scattered at the intersection (Compute the child ray).
	__device__ bool scatter(Ray& r_in, HitRecord& rec, 
	Vec3& attenuation, Ray& scattered, curandState *local_randState)
	{
		char mat_type = type;
		bool result;

		switch(mat_type) {
			case 'l':  // lambertian (diffuse)
				result =  scatter_lambertian(r_in, rec, attenuation, scattered, local_randState);
				break;
			case 'm':  // metal
				result =  scatter_metal(r_in, rec, attenuation, scattered, local_randState);
				break;
			case 'd':  // dielectrics
				result =  scatter_dielectric(r_in, rec, attenuation, scattered, local_randState);
				break;	
		}
		return result;
	}

	// 2.1. scatter_lambertian: scatter function of diffuse(lambertian) materials
	__device__ bool scatter_lambertian(Ray& r_in, HitRecord& rec, 
	Vec3& attenuation, Ray& scattered, curandState *local_randState)
	{
		Vec3 scatter_direction = rec.normal + random_in_unit_sphere(local_randState);
	    
		// If child ray's direction is a degenerate scatter direction
		if (scatter_direction.near_zero()) scatter_direction = rec.normal;

		scattered = Ray{rec.point, scatter_direction, r_in.time};
		attenuation = albedo;

		return true;
	}

	// 2.2. scatter_metal: scatter function of metals
	__device__ bool scatter_metal(Ray& r_in, HitRecord& rec, 
	Vec3& attenuation, Ray& scattered, curandState *local_randState)
	{
		Vec3 reflected = reflect(unit_vector(r_in.direction), rec.normal);
		
		scattered = Ray{rec.point, reflected + c*random_in_unit_sphere(local_randState), r_in.time};  // c: fuzziness
		attenuation = albedo;

		//DEBUGGING
		//printf("Color: %.1lf %.1lf %.1lf \n", albedo.e[0], albedo.e[1], albedo.e[2]);

		return (dot(scattered.direction, rec.normal) > 0.0f);
	}

	// 2.3. scatter_dielectric: scatter function of dielectrics
	__device__ bool scatter_dielectric(Ray& r_in, HitRecord& rec, 
	Vec3& attenuation, Ray& scattered, curandState *local_randState)
	{
		attenuation = Vec3{1.0, 1.0, 1.0};
		float refraction_ratio = rec.front_face ? (1.0/c) : c;  // c: Refraction Index
		Vec3 unit_direction = unit_vector(r_in.direction);
		Vec3 minus_unit_direction = {-unit_direction.e[0], -unit_direction.e[1], -unit_direction.e[2]};   		
		float cos_theta = dot(minus_unit_direction, rec.normal);
		if(cos_theta > 1.0f) cos_theta = 1.0f;
		float sin_theta = sqrt(1.0f - cos_theta*cos_theta);	
		bool cannot_refract = refraction_ratio * sin_theta > 1.0;
		Vec3 direction;

		if (cannot_refract || reflectance(cos_theta, refraction_ratio) > curand_uniform(local_randState))
			direction = reflect(unit_direction, rec.normal);
		else
			direction = refract(unit_direction, rec.normal, refraction_ratio);
		scattered = Ray{rec.point, direction, r_in.time};
		
		return true;
	}

	__device__ bool print_info(int idx) 
	{
		// DEBUGGING
		switch(type) {
			case 'l':  // lambertian (diffuse)
				printf("------------------------------ MATERIAL %d INFO ------------------------------\n  -%-8s: %-10s\n  -%-8s: (%.1lf, %.1lf, %.1lf)\n----------------------------------------------------------------------------\n",
				idx,
				"Type", "Diffuse",
				"Albedo", albedo.e[0], albedo.e[1], albedo.e[2]);				
				break;
			case 'm':  // metal
				printf("------------------------------ MATERIAL %d INFO ------------------------------\n  -%-8s: %-10s\n  -%-8s: %lf\n  -%-8s: (%.1lf, %.1lf, %.1lf)\n----------------------------------------------------------------------------\n",
				idx,
				"Type", "Metal",
				"Fuzziness", c,
				"Albedo", albedo.e[0], albedo.e[1], albedo.e[2]);	
				break;
			case 'd':  // dielectrics
				printf("------------------------------ MATERIAL %d INFO ------------------------------\n  -%-8s: %-10s\n  -%-8s: %lf\n----------------------------------------------------------------------------\n",
				idx,
				"Type", "Dielectric",
				"Refract Index", c);	
				break;	
		}
	}

} Material;


// functions
/*
// 1. reflectance: Compute the reflectance (dielectrics)
__device__ float reflectance(float cosine, float ref_idx) 
{
	// Schlick's approximation for reflectance & refractance
	float r0 = (1-ref_idx) / (1+ref_idx);
	r0 = r0*r0;
	return r0 + (1-r0)*pow((1 - cosine), 5);
}

// 2. scatter: Compute how the ray is scattered at the intersection (Compute the child ray).
__device__ bool scatter(const Material &mat, const Ray& r_in, const HitRecord& rec, 
Vec3& attenuation, Ray& scattered, curandState *local_randState)
{
	char mat_type = mat.type;
	bool result;

	switch(mat_type) {
		case 'l':  // lambertian (diffuse)
			result =  scatter_lambertian(r_in, rec, attenuation, scattered, local_randState);
			break;
		case 'm':  // metal
			result =  scatter_metal(r_in, rec, attenuation, scattered, local_randState);
			break;
		case 'd':  // dielectrics
			result =  scatter_dielectric(r_in, rec, attenuation, scattered, local_randState);
			break;	
	}
	return result;
}

// 2.1. scatter_lambertian: scatter function of diffuse(lambertian) materials
__device__ bool scatter_lambertian(const Material &mat, const Ray& r_in, const HitRecord& rec, 
Vec3& attenuation, Ray& scattered, curandState *local_randState)
{
	Vec3 scatter_direction = rec.normal + random_in_unit_sphere(local_randState);
	
	// If child ray's direction is a degenerate scatter direction
	if (scatter_direction.near_zero()) scatter_direction = rec.normal;

	scattered = Ray{rec.p, scatter_direction, r_in.time()};
	attenuation = mat.albedo;

	return true;
}

// 2.2. scatter_metal: scatter function of metals
__device__ bool scatter_metal(const Material &mat, const Ray& r_in, const HitRecord& rec, 
Vec3& attenuation, Ray& scattered, curandState *local_randState)
{
	Vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
	
	scattered = Ray{rec.p, reflected + mat.c*random_in_unit_sphere(local_randState), r_in.time()};  // c: fuzziness
	attenuation = mat.albedo;

	//DEBUGGING
	//printf("Color: %.1lf %.1lf %.1lf \n", mat.albedo.e[0], mat.albedo.e[1], mat.albedo.e[2]);

	return (dot(scattered.direction(), rec.normal) > 0.0f);
}

// 2.3. scatter_dielectric: scatter function of dielectrics
__device__ bool scatter_dielectric(const Material &mat, const Ray& r_in, const HitRecord& rec, 
Vec3& attenuation, Ray& scattered, curandState *local_randState)
{
	attenuation = Vec3{1.0, 1.0, 1.0};
	float refraction_ratio = rec.front_face ? (1.0/mat.c) : mat.c;  // c: Refraction Index
	Vec3 unit_direction = unit_vector(r_in.direction());
	Vec3 minus_unit_direction = {-unit_direction.e[0], -unit_direction.e[1], -unit_direction.e[2]};   		
	float cos_theta = dot(minus_unit_direction, rec.normal);
	if(cos_theta > 1.0f) cos_theta = 1.0f;
	float sin_theta = sqrt(1.0f - cos_theta*cos_theta);	
	bool cannot_refract = refraction_ratio * sin_theta > 1.0;
	Vec3 direction;

	if (cannot_refract || reflectance(cos_theta, refraction_ratio) > curand_uniform(local_randState))
		direction = reflect(unit_direction, rec.normal);
	else
		direction = refract(unit_direction, rec.normal, refraction_ratio);
	scattered = Ray{rec.p, direction, r_in.time()};
	
	return true;
}
*/


#endif
