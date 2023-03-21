/* FINISHED
 * ===================================================
 *
 *     Project Name:  RT_GPU
 *        File Name:  vec3.h
 *      Description:  
 *					  [UPDATES]
 *					    1) All classes are replaced with structures.
 *					    2) Sturctures are passed by pointers(*) instead of by references(&) in __global__ functions (not in __device__ functions).
 *          Created:  2022/11/08
 * 
 * ===================================================
 */


// Preprocessors
#ifndef VEC3_H
#define VEC3_H
#include <cmath>
#include <iostream>

// Generate a random 3D vector.
//#define RND3DVEC Vec3(curand_uniform(local_randState),curand_uniform(local_randState),curand_uniform(local_randState))


// Usings
//using std::sqrt;


// Functions
// 1. fabs_float: Compute the absolute value of a real number.
__device__ inline float fabs_float(float num) {
	if(num > 0)	
		return num;
	else 
		return -num;
}


// Structures
typedef struct _Vec3 {
	//Vec3(float e1, float e2, float e3) : e{e0, e1, e2} {}
	float e[3];

	__device__ float x() { return e[0]; }
	__device__ float y() { return e[1]; }
 	__device__ float z() { return e[2]; }

	__device__ _Vec3 operator-() const { return _Vec3{-e[0], -e[1], -e[2]}; }
	__device__ float operator[](int i) const { return e[i]; }
	__device__ float& operator[](int i) { return e[i]; }

	__device__ _Vec3& operator+=(const _Vec3& v) {
		e[0] += v.e[0];
		e[1] += v.e[1];
		e[2] += v.e[2];

		return *this;
	}

	__device__ _Vec3& operator*=(const float t) {
		e[0] *= t;
		e[1] *= t;
		e[2] *= t;

		return *this;
	}

	__device__ _Vec3& operator*=(const _Vec3 &v) {
		e[0] *= v.e[0];
		e[1] *= v.e[1];
		e[2] *= v.e[2];

		return *this;
	}

	__device__ _Vec3& operator/=(const float t) {
		return *this *= 1 / t;
	}


	__device__ float length_squared()
	{
		return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
	}
	
	__device__ float length() 
	{
		return sqrt(length_squared());
	}

	__device__ bool near_zero() 
	{
		// Returns true if the vector is close enough to zero in all dimensions.
		const float s = 1e-8;
		return (fabs_float(e[0]) < s) && (fabs_float(e[1]) < s) && (fabs_float(e[2]) < s);
	}
	
} Vec3;


// Functions
/*
__device__ float length_squared(Vec3 &vec)
{
	return vec.e[0] * vec.e[0] + vec.e[1] * vec.e[1] + vec.e[2] * vec.e[2];
}

__device__ float length(Vec3 &vec) 
{
	return sqrt(length_squared(vec));  // Q. What is 'sqrt'???
}

__device__ bool near_zero(Vec3 &vec) 
{
	// Returns true if the vector is close enough to zero in all dimensions.
	const float s = 1e-8;
	return (fabs_float(vec.e[0]) < s) && (fabs_float(vec.e[1]) < s) && (fabs_float(vec.e[2]) < s);
}
*/


// Utility Functions
__device__ inline Vec3 operator+(const Vec3& u, const Vec3& v) {
    	return Vec3{u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]};
}

__device__ inline Vec3 operator-(const Vec3& u, const Vec3& v) {
    	return Vec3{u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]};
}

__device__ inline Vec3 operator*(const Vec3& u, const Vec3& v) {
    	return Vec3{u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]};
}

__device__ inline Vec3 operator*(float t, const Vec3& v) {
    	return Vec3{t * v.e[0], t * v.e[1], t * v.e[2]};
}

__device__ inline Vec3 operator*(const Vec3& v, float t) {
    	return t * v;
}

__device__ inline Vec3 operator/(Vec3 v, float t) {
    	return (1 / t) * v;
}

__device__ inline float dot(const Vec3& u, const Vec3& v) {
    	return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

__device__ inline Vec3 cross(const Vec3& u, const Vec3& v) {
    	return Vec3{u.e[1] * v.e[2] - u.e[2] * v.e[1],
			u.e[2] * v.e[0] - u.e[0] * v.e[2],
			u.e[0] * v.e[1] - u.e[1] * v.e[0]};
}

__device__ inline Vec3 unit_vector(Vec3 v) {
    	return v / v.length();
}

__device__ inline float random_float(curandState *local_randState) {  // range: (0, 1]
	return curand_uniform(local_randState);
}  

__device__ inline float random_float(float min, float max, curandState *local_randState) {
	float f = random_float(local_randState) * (max - min) + min;
	return f;
}

__device__ inline Vec3 randomVec3(curandState *local_randState) {
    	return Vec3{curand_uniform(local_randState),curand_uniform(local_randState),curand_uniform(local_randState)};
}

__device__ inline Vec3 randomVec3(float min, float max, curandState *local_randState) {
	Vec3 v3 = (max - min) * randomVec3(local_randState) + Vec3{min, min, min};
	return v3;
}


__device__ inline Vec3 random_in_unit_sphere(curandState *local_randState) {
    while (true) {
		//point3 p = vec3::random(-1, 1);
		Vec3 p = randomVec3(local_randState) * 2 - Vec3{1, 1, 1};  // Since RAND3DVEC generates a random 3D vector in range (0, 1].
		if (p.length_squared() >= 1.0f) continue;

		return p;
    }
}

__device__ inline Vec3 random_unit_vector(curandState *local_randState) {
	return unit_vector(random_in_unit_sphere(local_randState));
}

__device__ inline Vec3 random_in_hemisphere(const Vec3& normal, curandState *local_randState) {
    Vec3 in_unit_sphere = random_in_unit_sphere(local_randState);

	if (dot(in_unit_sphere, normal) > 0.0) // If the direction is in the same hemisphere as the normal
		return in_unit_sphere;
    else
	   	return -in_unit_sphere;
}

__device__ inline Vec3 reflect(const Vec3& v, const Vec3& n) {
     	return v - 2*dot(v,n)*n;
}

__device__ inline Vec3 refract(const Vec3& uv, const Vec3& n, float etai_over_etat) {
    float cos_theta = dot(-uv, n);
    if(cos_theta > 1.0f) cos_theta = 1.0f;
	
	Vec3 r_out_perp =  etai_over_etat * (uv + cos_theta*n);
    Vec3 r_out_parallel = -sqrt(fabs_float(1.0 - r_out_perp.length_squared())) * n;
    
	return r_out_perp + r_out_parallel;
}

__device__ inline Vec3 random_in_unit_disk(curandState *local_randState) {
	while (true) {
	//auto p = vec3(random_float(-1,1), random_float(-1,1), 0);
	Vec3 p = Vec3{curand_uniform(local_randState), curand_uniform(local_randState), 0} * 2 - Vec3{1,1,0};
	if (p.length_squared() >= 1.0f) continue;

	return p;
	}
}

/*
// Type aliases for Vec3
using point3 = vec3;   // 3D point
using color = vec3;    // RGB color
*/


#endif