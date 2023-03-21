/* FINISHED
 * ===================================================
 *
 *     Project Name:  RT_GPU
 *        File Name:  sphere.h
 *      Description:  
 *					  [UPDATES]
 *					    1) All classes are replaced with structures.
 *					    2) Sturctures are passed by pointers(*) instead of by references(&) in __global__ functions (not in __device__ functions).
 *          Created:  2022/11/08
 * 
 * ===================================================
 */


// Preprocessors
#ifndef SPHERE_H
#define SPHERE_H

#define IMAGE_WIDTH 8
#include "vec3.h"
#include "hit_record.h"


// Structures
typedef struct _Sphere {
	int idx;
	Vec3 center;
	float radius;
	int matIdx;

	// 1. hit: Test whether the ray hits the sphere and record its info.
	__device__ bool hit(Ray &r, float t_min, float t_max, HitRecord &rec) 
	{
		// DEBUGGING
		int xIdx = threadIdx.x + blockIdx.x * blockDim.x;
		int yIdx = threadIdx.y + blockIdx.y * blockDim.y;
		int pixelIdx = yIdx * IMAGE_WIDTH + xIdx; 	
		Vec3 pm = center - Vec3{radius, radius, radius};
		Vec3 pM = center + Vec3{radius, radius, radius};


		Vec3 oc = r.origin - center;
		float a = (r.direction).length_squared();
		float half_b = dot(oc, r.direction);
		float c = oc.length_squared() - radius * radius;
		float discriminant = half_b * half_b - a * c;
		float sqrtd = sqrt(discriminant);  // sqrt(b^2-ac)
		float root = (-half_b - sqrtd) / a;

		// If the ray does not hit the sphere, 
		if (discriminant < 0) {
			// DEBUGGING
			/*
			printf("------------------------------ THD %d: SPHERE NOT HIT! D/4 = %.1lf!---------------\n  -%-8s: (%.1lf,%.1lf,%.1lf)\n  -%-8s: (%.1lf, %.1lf, %.1lf) ~ (%.1lf, %.1lf, %.1lf)\n--------------------------------------------------------------------------\n",		     
				pixelIdx, discriminant,
				"Ray Position", (r.at(root)).e[0], (r.at(root)).e[1], (r.at(root)).e[2],
				"AABB", pm.e[0], pm.e[1], pm.e[2], pM.e[0], pM.e[1], pM.e[2]);
			*/

			return false;
		}
		// DEBUGGING
		// If the ray hits the sphere,
		else {
			// FOR DEBUGGING
			printf("------------------------------ THD %d: SPHERE HIT!------------------------------\n  -%-8s: (%.1lf,%.1lf,%.1lf)\n  -%-8s: (%.1lf, %.1lf, %.1lf) ~ (%.1lf, %.1lf, %.1lf)\n--------------------------------------------------------------------------\n",
			pixelIdx,
			"Ray Position", (r.at(root)).e[0], (r.at(root)).e[1], (r.at(root)).e[2],
			"AABB", pm.e[0], pm.e[1], pm.e[2], pM.e[0], pM.e[1], pM.e[2]);

		}
		
		// Find the nearest root that lies in the acceptable range.
		if (root < t_min || t_max < root) {
			root = (-half_b + sqrtd) / a;
			if (root < t_min || t_max < root)
				return false;
		}
		// Record the hit info.
		rec.sphereIdx = idx;
		rec.time = root;
    	rec.point = r.at(rec.time);
    	Vec3 outward_normal = (rec.point - center) / radius;
    	rec.set_face_normal(r, outward_normal);  // rec.mat_ptr = mat_ptr;
		
		// If the current hit distance is the closest by far,
		// DEBUGGING	
		/*
		printf("------------------------------ THD %d: HIT RECORD UPDATED ----------------------\n  -%-8s: %.1lf\n  -%-8s: (%.1lf,%.1lf,%.1lf)\n  -%-8s: (%.1lf, %.1lf, %.1lf) ~ (%.1lf, %.1lf, %.1lf)\n--------------------------------------------------------------------------\n",
		pixelIdx, "Root", rec.time,
		"Point", (rec.point).e[0], (rec.point).e[1], (rec.point).e[2],
		"AABB", pm.e[0], pm.e[1], pm.e[2], pM.e[0], pM.e[1], pM.e[2]);
		*/
		
		return true;
	}

	// 2. bounding_box: Compute AABB of the sphere.
	__device__ bool bounding_box(float time0, float time1, Aabb& output_box) 
	{
		Vec3 min = {center.e[0] - radius, center.e[1] - radius, center.e[2] - radius};
		Vec3 max = {center.e[0] + radius, center.e[1] + radius, center.e[2] + radius};

		output_box = Aabb{min, max};

		// DEBUGGING	
		/*
		Vec3 pm = output_box.minimum, pM = output_box.maximum;
		printf("------------------------------ SPHERE %d INFO ------------------------------\n  -%-8s: (%.1lf,%.1lf,%.1lf)\n  -%-8s: %.1lf\n  -%-8s: (%.1lf, %.1lf, %.1lf) ~ (%.1lf, %.1lf, %.1lf)\n----------------------------------------------------------------------------\n",
		idx,
		"Center", center.e[0], center.e[1], center.e[2],
		"Radius", radius,
		"AABB", pm.e[0], pm.e[1], pm.e[2], pM.e[0], pM.e[1], pM.e[2]);
		*/
		
		return true;
	}

	__device__ bool print_info()
	{
		// DEBUGGING	
		printf("------------------------------ SPHERE %d INFO ------------------------------\n  -%-8s: (%.1lf,%.1lf,%.1lf)\n  -%-8s: %.1lf\n----------------------------------------------------------------------------\n",
		idx,
		"Center", center.e[0], center.e[1], center.e[2],
		"Radius", radius);	
		
	}

} Sphere;

/*
// Functions
// 1. hit_sphere: Test whether the ray hits the sphere and record its info.
__device__ bool hit_sphere(const Sphere &sphere, const Ray& r, float t_min, float t_max, HitRecord& rec) const 
{
	Vec3 center = sphere.center;
	Vec3 radius = sphere.radius;

	// DEBUGGING
	int xIdx = threadIdx.x + blockIdx.x * blockDim.x;
	int yIdx = threadIdx.y + blockIdx.y * blockDim.y;
	int pixelIdx = yIdx * IMAGE_WIDTH + xIdx; 	
	Vec3 pm = center - vec3(radius, radius, radius);
	Vec3 pM = center + vec3(radius, radius, radius);


	Vec3 oc = r.origin() - center;
	float a = r.direction().length_squared();
	float half_b = dot(oc, r.direction());
	float c = oc.length_squared() - radius * radius;
	float discriminant = half_b * half_b - a * c;
	float sqrtd = sqrt(discriminant);  // sqrt(b^2-ac)
	float root = (-half_b - sqrtd) / a;

	// If the ray does not hit the sphere, 
	if (discriminant < 0) {
		// DEBUGGING
		printf("------------------------------ THD %d: SPHERE NOT HIT! D/4 = %.1lf!---------------\n  -%-8s: (%.1lf,%.1lf,%.1lf)\n  -%-8s: (%.1lf, %.1lf, %.1lf) ~ (%.1lf, %.1lf, %.1lf)\n--------------------------------------------------------------------------\n",		     
			pixelIdx, discriminant,
			"Ray Position", (r.at(root)).e[0], (r.at(root)).e[1], (r.at(root)).e[2],
			"AABB", pm.e[0], pm.e[1], pm.e[2], pM.e[0], pM.e[1], pM.e[2]);

		return false;
	}
	// DEBUGGING
	// If the ray hits the sphere,
	else {
		// FOR DEBUGGING
		printf("------------------------------ THD %d: SPHERE HIT!------------------------------\n  -%-8s: (%.1lf,%.1lf,%.1lf)\n  -%-8s: (%.1lf, %.1lf, %.1lf) ~ (%.1lf, %.1lf, %.1lf)\n--------------------------------------------------------------------------\n",
		pixelIdx,
		"Ray Position", (r.at(root)).e[0], (r.at(root)).e[1], (r.at(root)).e[2],
		"AABB", pm.e[0], pm.e[1], pm.e[2], pM.e[0], pM.e[1], pM.e[2]);

	}

	// Find the nearest root that lies in the acceptable range.
	if (root < t_min || t_max < root) {
		root = (-half_b + sqrtd) / a;
		if (root < t_min || t_max < root)
			return false;
	}
	// Record the hit info.
	rec.sphereIdx = sphere.idx;
	rec.time = root;
	rec.point = r.at(rec.time);
	Vec3 outward_normal = (rec.point - center) / radius;
	rec.set_face_normal(r, outward_normal);  // rec.mat_ptr = mat_ptr;
	
	// If the current hit distance is the closest by far,
	// DEBUGGING	
	printf("------------------------------ THD %d: HIT RECORD UPDATED ----------------------\n  -%-8s: %.1lf\n  -%-8s: (%.1lf,%.1lf,%.1lf)\n  -%-8s: (%.1lf, %.1lf, %.1lf) ~ (%.1lf, %.1lf, %.1lf)\n--------------------------------------------------------------------------\n",
	pixelIdx, "Root", rec.t,
	"Point", (rec.p).e[0], (rec.p).e[1], (rec.p).e[2],
	"AABB", pm.e[0], pm.e[1], pm.e[2], pM.e[0], pM.e[1], pM.e[2]);
	
	return true;
}

// 2. bounding_box_sphere: Compute AABB of the sphere.
__device__ bool bounding_box_sphere(const Sphere &sphere, float time0, float time1, Aabb& output_box) 
{
	Vec3 center = sphere.center;
	Vec3 radius = sphere.radius;

	Vec3 min = {center.e[0] - radius, center.e[1] - radius, center.e[2] - radius};
	Vec3 max = {center.e[0] + radius, center.e[1] + radius, center.e[2] + radius};

	output_box = Aabb{min, max};

	// DEBUGGING
	
	Vec3 pm = output_box.minimum, Vec3 = output_box.maximum;
	printf("------------------------------ SPHERE %d INFO ------------------------------\n  -%-8s: (%.1lf,%.1lf,%.1lf)\n  -%-8s: %.1lf\n  -%-8s: (%.1lf, %.1lf, %.1lf) ~ (%.1lf, %.1lf, %.1lf)\n----------------------------------------------------------------------------\n",
	sphereIdx,
	"Center", center.e[0], center.e[1], center.e[2],
	"Radius", radius,
	"AABB", pm.e[0], pm.e[1], pm.e[2], pM.e[0], pM.e[1], pM.e[2]);
	

	return true;
}
*/

// 3. Generate an AABB of certain objects in object list.
__device__ inline bool list_bounding_box(Sphere *obj_list, int obj_num, int start, int end, float time0, float time1, Aabb& output_box) 
{	
	// If the object list is empty or the index is out of range
	if (obj_num == 0 || start < 0 || obj_num <= end) return false;
    	
	Aabb temp_box;
    bool first_box = true;

	for (int i = start; i <= end; i++) {
		printf("AABB including %d~%d\n", start, end);
		if (!(obj_list[i].bounding_box(time0, time1, temp_box))) {  // 수정사항: ! -를 붙여준다
			printf("COULD NOT GENERATE FUNCTION bounding_box().\n");
			return false;
		}
		output_box = first_box ? temp_box : surrounding_box(output_box, temp_box);
		first_box = false;
	}
	return true;
}

__device__ inline bool list_hit(Sphere *obj_list, int obj_num, Ray& r, float t_min, float t_max, HitRecord& rec) 
{
    	HitRecord temp_rec;
    	bool hit_anything = false;
    	float closest_so_far = t_max;
		Sphere *cur_obj;

    	for (int i = 0; i < obj_num; i++) {
			cur_obj = &obj_list[i];
		
			// Do the hit test of the current object.
			if (cur_obj->hit(r, t_min, closest_so_far, temp_rec)) {
				hit_anything = true;
	    		closest_so_far = temp_rec.time;
	    		rec = temp_rec;
			}
    	}
		printf("NO BVH\n");
    	return hit_anything;
}

#endif
