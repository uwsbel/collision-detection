#include "Broadphase.cuh"

// Test to see how many bins can be created. There will only be two objects in each bin, so the number of potential collisions will equal the number of bins.

struct Sphere {
		double r;
		real3 pos;
		uint index;
};

int main(int argc, char** argv) {
	// Step 1: Generate random spheres
	uint numSpheresPerSide = 10;     // number of spheres per side
	uint bpa = 10;
	if (argc == 2){
		numSpheresPerSide = atoi(argv[1]);
	}
	if (argc == 3){
			numSpheresPerSide = atoi(argv[1]);
			bpa = atoi(argv[2]);
	}
	double sphereRad = 0.25;
	double sphereSpacing = 1;

	vector<Sphere> spheres;

	Sphere sphere;
	int index = 0;
	for (int k = 0; k < numSpheresPerSide; k++) {
		for (int j = 0; j < numSpheresPerSide; j++) {
			for (int i = 0; i < numSpheresPerSide; i++) {
				sphere.pos = R3(i * sphereSpacing, j * sphereSpacing, k * sphereSpacing);
				sphere.r = sphereRad;
				sphere.index = index;

				// add two of these spheres so they must collide with eachother
				spheres.push_back(sphere);
				index++;
				spheres.push_back(sphere);
				index++;
			}
		}
	}
	// End Step 1

	// Step 2: Generate aabb_data from spheres
	custom_vector<real3> aabb_data;

	for (int i = 0; i < spheres.size(); i++) {
		real3 temp = spheres[i].pos;
		temp.x -= spheres[i].r;
		temp.y -= spheres[i].r;
		temp.z -= spheres[i].r;
		aabb_data.push_back(temp);
	}
	for (int i = 0; i < spheres.size(); i++) {
		real3 temp = spheres[i].pos;
		temp.x += spheres[i].r;
		temp.y += spheres[i].r;
		temp.z += spheres[i].r;
		aabb_data.push_back(temp);
	}
	// End Step 2

	// Step 3: Run broadphase algorithm to find potential collisions
	custom_vector<long long> potentialCollisions;

	Broadphase broadphaseManager;
	broadphaseManager.setBinsPerAxis(I3(8,8,bpa));

	cout << "Begin parallel broadphase" << endl;
	double startTime = omp_get_wtime();
	broadphaseManager.detectPossibleCollisions(aabb_data, potentialCollisions);
	double endTime = omp_get_wtime();
	printf("Time to detect: %lf seconds (%d possible collisions)\n", (endTime - startTime), broadphaseManager.getNumPossibleContacts());
	cout << "End parallel broadphase\n" << endl;
	// End Step 3
//int count = 0;
//	for(int i=0; i<potentialCollisions.size(); i++){
//		int2 pair = I2(int(potentialCollisions[i] >> 32), int(potentialCollisions[i] & 0xffffffff));
//		count+=2;
//		cout<<pair.x<<" "<<pair.y<<" "<<count-1<<endl;
//	}

	if (broadphaseManager.getNumPossibleContacts() == pow(numSpheresPerSide, 3)) {
		cout << "TEST PASSED" << endl;
	} else {
		cout << "TEST FAILED" << endl;
	}

	return 0;
}
