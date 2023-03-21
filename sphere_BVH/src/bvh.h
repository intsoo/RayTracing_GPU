/*
 * ===================================================
 *
 *     Project Name:  RT_GPU
 *        File Name:  bvh.h
 *      Description:  
 *					  [UPDATES]
 *					    1) All classes are replaced with structures.
 *					    2) Sturctures are passed by pointers(*) instead of by references(&) in __global__ functions (not in __device__ functions).
 *          Created:  2022/11/08
 * 
 * ===================================================
 */


// Preprocessors
#ifndef BVH_H
#define BVH_H

#include <algorithm>
#include "sphere.h"

#define RND_ (curand_uniform(local_randState))
#define MAX_NODE_NUM 8
#define IMG_WIDTH 200


// Structures
typedef struct _BvhNode {
	int idx;
	int first;
	int last;
//	int leftIdx;   // 2n+1 (n: current index)
//	int rightIdx;  // 2n+2
	Aabb box;

	// 1. hit: Test whether ray hits the node.
	__device__ bool hit(Ray &r, float t_min, float t_max, HitRecord &rec) {
		// DEBUGGING
		int xIdx = threadIdx.x + blockIdx.x * blockDim.x;
		int yIdx = threadIdx.y + blockIdx.y * blockDim.y;
		int pixelIdx = yIdx * IMG_WIDTH + xIdx;

		// No hit
		if (!box.hit(r, t_min, t_max)) {
			printf("------------------------------ THD %d: NODE %d NOT HIT! ------------------------------\n", pixelIdx, idx);  // DEBUGGING
			return false;
		}
		// Hit
		// DEBUGGING
		Vec3 pm = box.minimum, pM = box.maximum;
		printf("------------------------------ THD %d: NODE %d HIT! ------------------------------\n  -%-8s: (%d ~ %d)\n  -%-8s: (%.1lf,%.1lf,%.1lf) ~ (%.1lf,%.1lf,%.1lf)\n--------------------------------------------------------------------------\n", 
		pixelIdx, idx, "Objects", first, last, 
		"AABB", pm.e[0], pm.e[1], pm.e[2], pM.e[0], pM.e[1], pM.e[2]);
		
		return true;	
	}

	// 2. boounding_box: Return the AABB of the node.
	__device__ bool bounding_box(float time0, float time1, Aabb *output_box) {
		*output_box = box;
		return true;
	}

	// 3. print_info: Print node info.
	__device__ bool print_info() {
		Vec3 pm = box.minimum, pM = box.maximum;
		printf("------------------------------ NODE %d INFO ------------------------------\n  -%-8s: [%d~%d]\n  -%-8s: (%.1lf, %.1lf, %.1lf) ~ (%.1lf, %.1lf, %.1lf)\n----------------------------------------------------------------------------\n",
		idx,
		"Object Range", first, last,
		"AABB", pm.e[0], pm.e[1], pm.e[2], pM.e[0], pM.e[1], pM.e[2]);
	
	}

} BvhNode;

/*
// Functions
// 1. hit_bvhNode: Test whether ray hits the node.
__device__ bool hit_bvhNode(const BvhNode &node, const Ray& r, float t_min, float t_max, HitRecord& rec) {
	// DEBUGGING
	int xIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int yIdx = threadIdx.y + blockIdx.y * blockDim.y;
	int pixelIdx = yIdx * width + xIdx;

	// No hit
	if (!node.box.hit(r, t_min, t_max)) {
		printf("BVH NODE *NOT* HIT\n");  // DEBUGGING
		return false;
	}
	// Hit
	// DEBUGGING
	Vec3 pm = node.box.minimum, pM = node.box.maximum;
	printf("------------------------------ THD %d: NODE %d HIT! ------------------------------\n  -%-8s: (%d ~ %d)\n  -%-8s: (%.1lf,%.1lf,%.1lf) ~ (%.1lf,%.1lf,%.1lf)\n--------------------------------------------------------------------------\n", 
	pixelIdx, node.idx, "Objects", node.first, node.last, 
	"AABB", pm.e[0], pm.e[1], pm.e[2], pM.e[0], pM.e[1], pM.e[2]);
	
	return true;	
}

// 2. bounding_box_bvhNode: Return the AABB of the node.
__device__ bool bounding_box_bvhNode(const BvhNode &node, float time0, float time1, Aabb& output_box) {
	output_box = node.box;
	return true;
}
*/

// 3. Generate random integers in range (min, max].
__device__ inline int random_integer(int min, int max, curandState *local_randState) {
    int n = (max - min) * RND_ + min;
	return n;
}

// 4. Print node info.
__device__ bool print_info_bvhNode(BvhNode &node) {
	Vec3 pm = node.box.minimum, pM = node.box.maximum;
	printf("Node %d [AABB] (%.1lf, %.1lf, %.1lf) ~ (%.1lf, %.1lf, %.1lf)\n", node.idx, pm.e[0], pm.e[1], pm.e[2], pM.e[0], pM.e[1], pM.e[2]);
}

// 5. box_compare: Compare which box is closer to the origin in respect to the chosen axis.
__device__ bool box_compare(Sphere &obj_a, Sphere &obj_b, int axis) {
	Aabb box_a;
    Aabb box_b;
    	
	if (!obj_a.bounding_box(0,0,box_a) || !obj_b.bounding_box(0,0,box_b))  // If the boxes exists,
		printf("No bounding box in BVH node constructor!\n");
    	
		return box_a.minimum.e[axis] < box_b.minimum.e[axis];
}

// 6. generate_bvh: Generate BVH.
__device__ inline void generate_bvh(
	BvhNode *bvh, 
	Sphere *obj_list,
	int obj_num,
	float time0, float time1)
{
	int node_num = 2 * obj_num - 1;

	// Root node
	BvhNode root{ .idx = 0, .first = 0, .last = obj_num-1 };
	bvh[0] = root;

	// BVH
	for(int idx = 0; idx < node_num; idx++) {
		BvhNode cur_node = bvh[idx];
		int start = cur_node.first;
		int end = cur_node.last;
		int obj_span = end - start + 1;  // number of objects
		int axis = 0;  // Fix to x-axis

		// Compute the AABB of the current node.
		list_bounding_box(obj_list, obj_num, start, end, time0, time1, bvh[idx].box);

		// Initialize first and last object index of the two child nodes.
		if(obj_span >= 2){  // If intermediate node,
			int mid = start + obj_span/2 - 1; 

			//////////////////////////////////////////////////////////
			// (*수정할 부분*) 인덱스 계산 사용 불가!!!!!!!!!!!!
			int l_idx = 2*idx+1;  // left child index (2n+1)
			int r_idx = l_idx+1;  // right child index (2n+2)
			//////////////////////////////////////////////////////////

			// Sort the objects(spheres) from closest to furthest in respect to the chosen axis.
			for(int i = start; i < end; i++) {
				for(int j = start; j < end-i; j++) {
					// If box B is closer to the origin O(0, 0, 0) than box A is, swap them.
					if(box_compare(obj_list[j+1], obj_list[j], axis)) { 
						Sphere temp = obj_list[j];
						obj_list[j] = obj_list[j+1];
						obj_list[j+1] = temp;
					}
				}
			}	

			if(l_idx && r_idx) {
				bvh[l_idx] = BvhNode{ .idx = l_idx, .first = start, .last = mid }; 
				bvh[r_idx] = BvhNode{ .idx = r_idx, .first = mid+1, .last = end };
			} else {
				printf("***** bvh[]: index out of range *****\n");
			}
		}
	}
	printf("\n\n================================== BVH CONSTURCTION COMPLETED ==================================\n\n\n");
}

// 7. print_bvh: Print the result of BVH.
__device__ inline void print_bvh(
	BvhNode *bvh, 
	Sphere *obj_list,
	int obj_num)
{
	int node_num = 2*obj_num-1;

	printf("================================== PRINT BVH ==================================\n\n");
	for(int idx = 0; idx < node_num; idx++) {
		BvhNode cur_node = bvh[idx];
		int obj_span = cur_node.last - cur_node.first + 1;
		Vec3 pm = cur_node.box.minimum, pM = cur_node.box.maximum;
		Sphere leaf_node = obj_list[cur_node.first];  // if cur_node is a leaf node

		printf("------------------------------ NODE %d INFO ------------------------------\n  -%-8s: [%d~%d]\n  -%-8s: (%.1lf, %.1lf, %.1lf) ~ (%.1lf, %.1lf, %.1lf)\n----------------------------------------------------------------------------\n",
			idx,
			"Object Range", cur_node.first, cur_node.last,
			"AABB", pm.e[0], pm.e[1], pm.e[2], pM.e[0], pM.e[1], pM.e[2]);

		if(obj_span < 2){
			printf("------------------------------ NODE %d: SPHERE %d INFO ------------------------------\n  -%-8s: (%.1lf,%.1lf,%.1lf)\n  -%-8s: %lf\n----------------------------------------------------------------------------\n",
				cur_node.idx, leaf_node.idx,
				"Left: Center", leaf_node.center.e[0], leaf_node.center.e[1], leaf_node.center.e[2],
				"Left: Radius", leaf_node.radius);
		}
	}
	printf("===============================================================================\n");
	//printf("%d\n", node_num);
}

// 8. search_bvh: Using BVH traversal, compute whether any object is hit and record the nearest hit intersection.
// 8.1. Write whether to visit each node in bool array 'visit' instead of using stack.
__device__ inline bool search_bvh(
	BvhNode *bvh,    // final bvh(nodes stored in depth-first order)
	Sphere *obj_list, int obj_num,
	Ray &r, 
	float t_min, float t_max, HitRecord &rec, 
	curandState *local_randState)
{
	int node_num = 2*obj_num-1;
	bool visit[MAX_NODE_NUM] = {0};  // whether to visit the node or not
	bool is_hit_first = false;  // whether any leaf node is hit

	// BVH Traversal
	// 1. Visit the root node.
	visit[0] = true;
	
	// 2. Search the tree in breath-first order.
	for(int idx = 0; idx < node_num; idx++)
	{ 
		if(visit[idx] == true)
		{
			BvhNode cur_node = bvh[idx];
			int obj_span = cur_node.last - cur_node.first + 1;
			//bool is_first_leaf = false;  // whether the first leaf node is reached

			// Leaf Node
			if(obj_span == 1)  // if(is_first_leaf) : 첫 leaf node를 만나면 그 이후부터는 쭉 leaf node이다!
			{
				Sphere *cur_obj = &obj_list[cur_node.first];  // current object(leaf node)
				
				// DEBUGGING
				/*
				printf("------------------------------ LEAF NODE %d: SPHERE INFO ------------------------------\n  -%-8s: (%.1lf,%.1lf,%.1lf)\n  -%-8s: %lf\n----------------------------------------------------------------------------\n",
				cur_node.idx,
				"Center", cur_obj.center.e[0], cur_obj.center.e[1], cur_obj.center.e[2],
				"Radius", cur_obj.radius);
				*/
				
				// Hit Test of Sphere(current object)
				if(cur_obj->hit(r, t_min, (is_hit_first)?rec.time:t_max, rec)) {  // if the current object is hit
					if(is_hit_first != true) is_hit_first = true; // if the current one is the first intersected object
				} 
			}
			// Intermediate Node
			else if(obj_span > 1) 
			{
				// Hit Test of BvhNode(current node)
				if(cur_node.hit(r, t_min, (is_hit_first)?rec.time:t_max, rec))  // If hit (at the point closer than any current hit point)
				{  
					/*
					// DEBUGGING
					Vec3 pm = cur_node.box.minimum, pM = cur_node.box.maximum;
					printf("------------------------------ NODE %d HIT! ------------------------------\n  -%-8s: (%d ~ %d)\n  -%-8s: (%.1lf,%.1lf,%.1lf) ~ (%.1lf,%.1lf,%.1lf)\n--------------------------------------------------------------------------\n", 
					cur_node.idx, "Objects", cur_node.first, cur_node.last, 
					"AABB", pm.e[0], pm.e[1], pm.e[2], pM.e[0], pM.e[1], pM.e[2]);
					*/
					
					// later visit its left&right child nodes.
					visit[2*idx+1] = true;  // left child
					visit[2*idx+2] = true;  // right child
				}
			}
		}
	}
	return is_hit_first;
}


// 8.2. Each thread(ray) uses a stack.
__device__ inline bool bvh_traversal( 
	BvhNode **stack,
	BvhNode *bvh,
	const int STACK_SIZE, 
	Sphere* obj_list,
	Ray &r,
	float t_min, float t_max,
	HitRecord &rec)
{
	int top = 0;  // top of the stack
	bool is_hit_first = false;

	// Visit the root node first.
	BvhNode *cur_node = &bvh[0];  
	if(!cur_node->hit(r, t_min, t_max, rec))
		return false;

	while(1) 
	{
		int obj_span = cur_node->last - cur_node->first + 1;  // # of objects in the current node
		
		// leaf node
		if (obj_span == 1) {
			
			Sphere *cur_obj = &obj_list[cur_node->first];  // current object(leaf node)
				
			// DEBUGGING
			/*
			printf("------------------------------ LEAF NODE %d: SPHERE INFO ------------------------------\n  -%-8s: (%.1lf,%.1lf,%.1lf)\n  -%-8s: %lf\n----------------------------------------------------------------------------\n",
			cur_node.idx,
			"Center", cur_obj.center.e[0], cur_obj.center.e[1], cur_obj.center.e[2],
			"Radius", cur_obj.radius);
			*/
			
			// Do the hit test of the current object(sphere).
			if(cur_obj->hit(r, t_min, (is_hit_first)?rec.time:t_max, rec)) {  // if the current object is hit
				if(is_hit_first != true) is_hit_first = true; // if the current one is the first intersected object
			} 

			// Decide which node to visit in the next iteration.
			if (top == 0)  // If the stack is empty,
				break;  // End the traversal.
			else 
				cur_node = stack[--top];  // Pop one node.	
		}
		
		// intermediate node
		else if (obj_span > 1) {  
			
			BvhNode* child_l = &bvh[cur_node->idx * 2 + 1];  // left child
			BvhNode* child_r = &bvh[cur_node->idx * 2 + 2];  // right child

			// hit test results
			int hit_l = child_l->hit(r, t_min, (is_hit_first)?rec.time:t_max, rec);
			int hit_r = child_r->hit(r, t_min, (is_hit_first)?rec.time:t_max, rec); 

			if(hit_l) {  // If the left child is hit,
				cur_node = child_l;  // Visit the left child in the next iteration.
				if(hit_r) {
					stack[top++] = child_r;  // Push the right child node if it is hit.
				}
			}
			else if(hit_r) {  // If only the right child is hit,
				cur_node = child_r;  // Visit the right child in the next iteration.
			}
			else {  // If both are not hit,
				if(top == 0) break;  // If stack is empty, end the traversal.
				else cur_node = stack[--top];  // Otherwise, pop one node.
			}
		}
	}	
	return is_hit_first;
}


/*
class stack {
	public: 
		int top;
		bvh_node **nodes;
	public:
		__device__ stack() { 
			top = 0;
	       	}

		__device__ stack(bvh_node **node_list) { 
			nodes = node_list;
		       	top = 0; 
		}
		
		__device__ bool isEmpty() { return top == 0; }
		__device__ bool isFull() { return top == MAX_SIZE; }
		__device__ int size() { return top; }
		__device__ void show() {
			printf("---------------------------------------------------\nSHOW %d NODES\n", top);

			printf("\n================STACK================\n");
			for(int i = 0; i < top; i++) { 
				printf("[%d] ", nodes[i]->idx);
			//	printf("-> NODE %d\n", nodes[i]->idx);
			}	
			printf("\n======================================\n\n");
		}
		__device__ void push(bvh_node *node, curandState *local_randState) {
			if(isFull()) {
				printf("[MESSAGE] Stack is already full\n");
				return;
			}

			//printf("***** NODE %d PUSHED *****\n", node->idx);
			nodes[top++] = node;
//			printf("[TOP = %d]\n", top);

			//show();
		}
		__device__ bvh_node* pop() {
			if(isEmpty()) {
				printf("[ERROR] Stack is empty\n");
				return NULL;
			}

			bvh_node *node = nodes[--top];

			//printf("***** NODE %d POPPED *****\n", node->idx);
//			printf("[TOP = %d]\n", top);
			
			//show();  // Show the stack.

			return node;
		}
};
*/



/*
__device__ bvh_node::bvh_node(
		hittable_list *l, hittable *pre,
		int start, int end, float time0, float time1,
		curandState *local_randState) {	
	
		first = start;
		last = end;

		//predecessor = pre;
		l->list_bounding_box(start, end, time0, time1, box);
		
		// DEBUGGING
		point3 pm = box.minimum, pM = box.maximum;
		printf("Node %d [AABB] (%.1lf, %.1lf, %.1lf) ~ (%.1lf, %.1lf, %.1lf)\n", idx, pm.e[0], pm.e[1], pm.e[2], pM.e[0], pM.e[1], pM.e[2]);
}
*/


/* 
// overloaded constructor
__device__ bvh_node::bvh_node(
		int idx,  // Assign an index to every node (for debugging).
		hittable_list *l, hittable *pre,
		int start, int end, float time0, float time1,
		curandState *local_randState) {	
		
		// Assign an index to each node.
		nodeIdx = idx;
	
		first = start;
		last = end;

		//predecessor = pre;
		l->list_bounding_box(start, end, time0, time1, box);


		// DEBUGGING
		int object_span = last - first + 1; 
		point3 pm = box.minimum, pM = box.maximum;
		printf("------------------------------ NODE %d CREATED (obj# = %d) ------------------------------\n", nodeIdx, object_span);
		printf("  -%-8s: (%d ~ %d)\n", "Objects", first, last);
		printf("  -%-8s: (%.1lf,%.1lf,%.1lf) ~ (%.1lf,%.1lf,%.1lf)\n", "AABB", pm.e[0], pm.e[1], pm.e[2], pM.e[0], pM.e[1], pM.e[2]);

		if ( object_span <= 2 ) { 
			sphere *leftN = (sphere *)left, *rightN = (sphere *)right;
			printf("------------------------------ Node %d: Sphere Info -------------------------------------\n  -%-8s: (%.1lf,%.1lf,%.1lf)\n  -%-8s: %lf\n  -%-8s: (%.1lf,%.1lf,%.1lf)\n  -%-8s: %lf\n-----------------------------------------------------------------------------------------\n",
			nodeIdx, 
			"Left Center", leftN->center.e[0], leftN->center.e[1], leftN->center.e[2],
			"Left Radius", leftN->radius,
			"Right Center", rightN->center.e[0], rightN->center.e[1], rightN->center.e[2],
			"Right Radius", rightN->radius);
		}			
}
*/


/*
__device__ inline bool testBvhHit(
		// 일단은 global memory에 저장하게끔 했음!
		bvh_node **d_bvh,    // final bvh(nodes stored in depth-first order)
		int objectNum,
	   	int pixelIdx, const ray&r, float t_min, float t_max, hit_record& rec, curandState *local_randState) {


	printf("CHECK BVH\n");
	for(int i=0; i<3; i++) {
		printf("[%d] Node %d -> \n", i, d_bvh[i]->nodeIdx);
	}
	printf("\n");


	// RT16: SEARCH THE SHARED MEMORY ADDRESS (BVH TREE) SEQUENTIALLY
	bvh_node *cur_node;
	//int nodeNum = 2 * objectNum - 1;  // # of nodes = 2n-1 (n: # of objects)
	int nodeNum = 2 * objectNum - 1 - objectNum;  // RT16: leaf node는 저장하지 않았기 때문! 

	bool isLeftHit = false;
	bool isRightHit = false;
	bool isHit = false;  // If at least one leaf node is hit, isHit == true

	//if(pixelIdx == 0) printf("\n\n========================================== BVH TRAVERSAL =======================================\n\n\n");

	for(int iter = 0; iter < nodeNum; iter+=1) {
//		printf("ITER: %d\n", iter);


		if ( cur_node->hit(pixelIdx, r, t_min, t_max, rec) ) { 

			// DEBUGGING
			// Node Information
			point3 pm = cur_node->box.minimum, pM = cur_node->box.maximum;
			printf("------------------------------ THD %d: NODE %d HIT! ------------------------------\n  -%-8s: (%d ~ %d)\n  -%-8s: (%.1lf,%.1lf,%.1lf) ~ (%.1lf,%.1lf,%.1lf)\n--------------------------------------------------------------------------\n", 
			pixelIdx, cur_node->nodeIdx, "Objects", cur_node->first, cur_node->last, 
			"AABB", pm.e[0], pm.e[1], pm.e[2], pM.e[0], pM.e[1], pM.e[2]);


			bool hasLeft = (cur_node->left != NULL);
			bool hasRight = (cur_node->right != NULL);	

			if ( object_span <= 2 ) {  // parents of leaf nodes (!= leaf nodes)


				// DEBUGGING
				sphere *leftN = (sphere *)cur_node->left, *rightN = (sphere *)cur_node->right;
				printf("------------------------------ Node %d: SPHERE INFO (obj # = %d) ------------------------------\n  -%-8s: (%.1lf,%.1lf,%.1lf)\n  -%-8s: %lf\n  -%-8s: (%.1lf,%.1lf,%.1lf)\n  -%-8s: %lf\n----------------------------------------------------------------------------\n",
				cur_node->nodeIdx, object_span,
				"Left: Center", leftN->center.e[0], leftN->center.e[1], leftN->center.e[2],
				"Left: Radius", leftN->radius,
				"Right: Center", rightN->center.e[0], rightN->center.e[1], rightN->center.e[2],
				"Right: Radius", rightN->radius);
			

				if(hasLeft) isLeftHit = (cur_node->left->hit(r, t_min, (isLeftHit||isRightHit)?rec.t:t_max, rec));
				if(hasRight) isRightHit = (cur_node->right->hit(r, t_min, (isLeftHit||isRightHit)?rec.t:t_max, rec));

				// RT18
				if(isLeftHit || isRightHit)	isHit = true;  // 한 번이라도 leaf node와 충돌했다면 무조건 true를 반환한다.

				//RT17: Hit test should end until checking all objects(leaf nodes).
				//RT16: if(isLeftHit || isRightHit)	return true;
			}
		}
	}
	
	//RT16: return false;
	//RT17: return isLeftHit || isRightHit;
	return isHit;  // RT18

}
*/

/*
__device__ inline bool testBvhHit(int pixelIdx, const ray& r, float t_min, float t_max, hit_record& rec, 
		bvh_node *root, bvh_node **d_nodes, curandState *local_randState) {	

	// Stack
	stack *bvhStack = new stack(d_nodes);

	// Root Node
	bvhStack->push(root, local_randState);
	bvhStack->show();

	// Initialization
	rec.t  = t_max; 

	bool isLeftHit = false;
	bool isRightHit = false;

	// Iterative Traversal
	while ( !(bvhStack->isEmpty()) ) {
		bvh_node *cur_node = bvhStack->pop();  // Take out the current node.
//		bvhStack->show();

		int object_span = cur_node->last - cur_node->first + 1;

//		printf("[HIT TEST] Thd: %d, Node: %d\n", pixelIdx, cur_node->nodeIdx);

		if ( cur_node->hit(pixelIdx, r, t_min, t_max, rec) ) {  // 수정사항: t_max -> rec.t (시간 단축)


		
			// FOR DEBUGGING: Node Information
			point3 pm = cur_node->box.minimum, pM = cur_node->box.maximum;
			printf("------------------------------ NODE %d HIT! ------------------------------\n", cur_node->nodeIdx);	
			printf("  -%-8s: (%d ~ %d)\n", "Objects", cur_node->first, cur_node->last);
			printf("  -%-8s: (%.1lf,%.1lf,%.1lf) ~ (%.1lf,%.1lf,%.1lf)\n", "AABB", pm.e[0], pm.e[1], pm.e[2], pM.e[0], pM.e[1], pM.e[2]);
			printf("--------------------------------------------------------------------------\n");

			printf("------------------------------ THD %d: NODE %d HIT! ------------------------------\n  -%-8s: (%d ~ %d)\n  -%-8s: (%.1lf,%.1lf,%.1lf) ~ (%.1lf,%.1lf,%.1lf)\n--------------------------------------------------------------------------\n", 
			pixelIdx, cur_node->nodeIdx, "Objects", cur_node->first, cur_node->last, 
			"AABB", pm.e[0], pm.e[1], pm.e[2], pM.e[0], pM.e[1], pM.e[2]);



			bool hasLeft = (cur_node->left != NULL);
			bool hasRight = (cur_node->right != NULL);	

			if ( object_span <= 2 ) {  // parents of leaf nodes (!= leaf nodes)
				if(hasLeft) isLeftHit = (cur_node->left->hit(r, t_min, rec.t, rec));
				if(hasRight) isRightHit = (cur_node->right->hit(r, t_min, rec.t, rec));

			}
			else {  // If the current node is an intermediate node,
				// Push left/right child node(s) of the current node. 	
//				printf("** INTERMEDIATE NODE **\n");

				if(hasLeft) bvhStack->push((bvh_node *)cur_node->right, local_randState);
				if(hasRight) bvhStack->push((bvh_node *)cur_node->left, local_randState);
			}
		}
	}

	printf("\n\n================================== BVH TRAVERSAL COMPLETED ==================================\n\n\n");

	// RT16: return false;
	return isLeftHit || isRightHit;  // RT17

}
*/


#endif