/*
 * ===================================================
 *
 *     Project Name:  RT_GPU
 *        File Name:  aabb.h
 *      Description:  
 *					  [UPDATES]
 *					    1) All classes are replaced with structures.
 *					    2) Sturctures are passed by pointers(*) instead of by references(&) in __global__ functions (not in __device__ functions).
 *          Created:  2022/11/08
 * 
 * ===================================================
 */


// Preprocessors
#ifndef AABB_H
#define AABB_H

#include "ray.h"
#include "vec3.h"


// Functions
// 1. Compute the minimum/maximum value among two floats.
__device__ inline float fminGPU(float a, float b) { return a < b ? a : b; }
__device__ inline float fmaxGPU(float a, float b) { return a > b ? a : b; }


// Structures
typedef struct _Aabb {
	Vec3 minimum;
	Vec3 maximum;	

	// Q. member functions of struct; Is it possible in CUDA?
	__device__ Vec3 min() { return minimum; }
	__device__ Vec3 max() { return maximum; }

	// Q. Check if Aabb.hit() is possible.
	// 2. Test whether ray hits the AABB.
	__device__ bool hit( Ray &r, float t_min, float t_max)  // Aabb.hit()
	{
		for (int axis = 0; axis < 3; axis++) {  // check all x, y, z axis
			float t0 = fminGPU((minimum[axis] - r.origin[axis]) / r.direction[axis], (maximum[axis] - r.origin[axis]) / r.direction[axis]);
			float t1 = fmaxGPU((minimum[axis] - r.origin[axis]) / r.direction[axis], (maximum[axis] - r.origin[axis]) / r.direction[axis]);
			
			t_min = fmaxGPU(t0, t_min);
			t_max = fminGPU(t1, t_max);
			
			if (t_max <= t_min)	return false;  // AABB not hit
		}
		return true;  // AABB hit
	}

} Aabb;

/*
// 2. Test whether ray hits the AABB.
__device__ bool hitAabb(Aabb &box, Ray &r, float t_min, float t_max) 
{
	for (int axis = 0; axis < 3; axis++) {  // check all x, y, z axis
		float t0 = fminGPU((box.minimum[axis] - r.origin[axis]) / r.direction[axis], (box.maximum[axis] - r.origin[axis]) / r.direction[axis]);
		float t1 = fmaxGPU((box.minimum[axis] - r.origin[axis]) / r.direction[axis], (box.maximum[axis] - r.origin[axis]) / r.direction[axis]);
		
		t_min = fmaxGPU(t0, t_min);
		t_max = fminGPU(t1, t_max);
		
		if (t_max <= t_min)	return false;  // AABB not hit
	}
	return true;  // AABB hit
}
*/

// 3. Compute an AABB(bounding box) surrounding the two given boxes.
__device__ inline Aabb surrounding_box(Aabb &box0, Aabb &box1) 
{

	Vec3 small{fminGPU(box0.min().x(), box1.min().x()),
		fminGPU(box0.min().y(), box1.min().y()),
		fminGPU(box0.min().z(), box1.min().z())};

	Vec3 big{fmaxGPU(box0.max().x(), box1.max().x()),
		fmaxGPU(box0.max().y(), box1.max().y()),
		fmaxGPU(box0.max().z(), box1.max().z())};

	/*	Q. 구조체 멤버에 바로 접근하는 방식은 왜 안되는지 찾아보기!
	Vec3 small{(fminGPU(box0.minimum.e[0], box1.minimum.e[0]),  // x
		fminGPU(box0.minimum.e[1], box1.minimum.e[1]),  // y
		fminGPU(box0.minimum.e[2], box1.minimum.e[2]))};  // z

    Vec3 big{(fmaxGPU(box0.maximum.e[0], box1.maximum.e[0]), 
		fmaxGPU(box0.maximum.e[1], box1.maximum.e[1]),  
		fmaxGPU(box0.maximum.e[2], box1.maximum.e[2]))};
	*/
    return Aabb{small,big}; 
}


#endif