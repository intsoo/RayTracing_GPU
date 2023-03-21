/*
 * ===================================================
 *
 *     Project Name:  RT_GPU
 *        File Name:  sphere_list.h
 *      Description:  
 *					  [UPDATES]
 *					    1) All classes are replaced with structures.
 *					    2) Sturctures are passed by pointers(*) instead of by references(&) in __global__ functions (not in __device__ functions).
 *          Created:  2022/11/08
 * 
 * ===================================================
 */


// Preprocessors

#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "aabb.h"
#include "sphere.h"
#include <memory>
#include <vector>


//using std::shared_ptr;
//using std::make_shared;


// Structures
typedef struct _SphereList {
	Sphere *obj;
	int objName;

	
} SphereList;



class hittable_list : public hittable {
public:
       	__device__ hittable_list() {}
    	__device__ hittable_list(hittable **object_list, int n) { objects = object_list, objects_num = n; }

    	//__device__ void clear() { objects.clear(); }
    	
	__device__ virtual bool hit(
			const ray& r, float t_min, float t_max, hit_record& rec) const override;

 	__device__ virtual bool bounding_box(
	    		float time0, float time1, aabb& output_box) const override;

 	__device__ virtual bool list_bounding_box(
	    		int start, int end, float time0, float time1, aabb& output_box) const;

public:
    	hittable **objects;
	int objects_num;
};


__device__ bool hittable_list::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    	hit_record temp_rec;
    	bool hit_anything = false;
    	auto closest_so_far = t_max;

    	for (int i = 0; i < objects_num; i++) {
		if (objects[i]->hit(r, t_min, closest_so_far, temp_rec)) {
	    		hit_anything = true;
	    		closest_so_far = temp_rec.t;
	    		rec = temp_rec;
		}
    	}
    	return hit_anything;
}


// Generate an AABB of object list.
__device__ bool hittable_list::bounding_box(float time0, float time1, aabb& output_box) const {
     	
	// If the object list is empty
	if (objects_num == 0) return false;
    	
	aabb temp_box;
    	bool first_box = true;

    	for (int i = 0; i < objects_num; i++) {
		if (objects[i]->bounding_box(time0, time1, temp_box)) {
			printf("FALSE\n");
			return false;
		}
		output_box = first_box ? temp_box : surrounding_box(output_box, temp_box);
		first_box = false;
    	}
    	return true;
}


// Generate an AABB of certain objects in object list.
__device__ bool hittable_list::list_bounding_box(int start, int end, float time0, float time1, aabb& output_box) const {
     	
	// If the object list is empty or the index is out of range
	if (objects_num == 0 || start < 0 || objects_num <= end) return false;
    	
	aabb temp_box;
    	bool first_box = true;

    	for (int i = start; i <= end; i++) {
		//printf("called\n");
		if (!(objects[i]->bounding_box(time0, time1, temp_box))) {  // 수정사항: ! -를 붙여준다
			printf("AABB BOX FALSE\n");
			return false;
		}
//		printf("AABB including %d~%d\n", start, end);
		output_box = first_box ? temp_box : surrounding_box(output_box, temp_box);
		first_box = false;
    	}
    	return true;
}


#endif
