/* FINISHED
 * ===================================================
 *
 *     Project Name:  RT_GPU
 *        File Name:  ray.h
 *      Description:  
 *					  [UPDATES]
 *					    1) All classes are replaced with structures.
 *					    2) Sturctures are passed by pointers(*) instead of by references(&) in __global__ functions (not in __device__ functions).
 *          Created:  2022/11/08
 * 
 * ===================================================
 */


// Preprocessors
#ifndef RAY_H
#define RAY_H

#include "vec3.h"


// Structures
typedef struct _Ray {
//	int thdIdx;
	Vec3 origin;
	Vec3 direction;
	float time;

	// 1. at: Compute position of the ray at a given time.
	__device__ Vec3 at(float t) 
	{
		return origin + t * direction;
    }

} Ray;


// Functions
// 1. ray_at: Compute position of the ray at a given time.
__device__ Vec3 ray_at(const Ray &r, float t) 
{
	return r.origin + t * r.direction;
}


#endif