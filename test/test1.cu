#include "Broadphase.cuh"
#include <assert.h>

// Simple test a single bin in broadphase

int main(int argc, char** argv)
{
	// Test
	int numObjects = 10;
	if(argc==2) numObjects = atoi(argv[1]);

	custom_vector<real3> aabb_data;
	custom_vector<long long> potentialCollisions;

	int numObjectsFactorial = 0;
	real3 point;
	for(int i=0;i<numObjects;i++)
	{
		point = make_real3(0,0,0);
		aabb_data.push_back(point);
		numObjectsFactorial+=i;
	}
	for(int i=0;i<numObjects;i++)
	{
		point = make_real3(1,1,1);
		aabb_data.push_back(point);
	}

	Broadphase broadphaseManager;
	broadphaseManager.setBinsPerAxis(make_real3(1,1,1));
	broadphaseManager.detectPossibleCollisions(aabb_data, potentialCollisions);

	assert(potentialCollisions.size()==numObjectsFactorial);

	cout << "Test Passed!" << endl;

	return 0;
}
