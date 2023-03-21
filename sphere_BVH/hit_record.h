/* FINISHED
 * ===================================================
 *
 *     Project Name:  RT_GPU
 *        File Name:  hit_record.h
 *      Description:  
 *					  [UPDATES]
 *					    1) All classes are replaced with structures.
 *					    2) Sturctures are passed by pointers(*) instead of by references(&) in __global__ functions (not in __device__ functions).
 *          Created:  2022/11/08
 * 
 * ===================================================
 */


// Preprocessors
#ifndef HITTABLE_H
#define HITTABLE_H

#include "aabb.h"
#include "ray.h"


// Structures
typedef struct _HitRecord {
	Vec3 point;
  	Vec3 normal;
	int sphereIdx;
 	float time;
 	bool front_face;

	// 1. set_face_normal: Set normal vector of the surface hit by the ray.
  	__device__ inline void set_face_normal(Ray &r, Vec3 &outward_normal) {
		front_face = dot(r.direction, outward_normal) < 0;
		normal = front_face ? outward_normal : -outward_normal;
  	}
  
} HitRecord;


// Functions 
// 1. set_face_normal: Set normal vector of the surface hit by the ray.
/*
__device__ inline void set_face_normal(HitRecord &rec, const Ray &r, const Vec3 &outward_normal) {
	rec.front_face = dot(r.direction(), outward_normal) < 0;
	rec.normal = rec.front_face ? outward_normal : -outward_normal;
}
*/


#endif
