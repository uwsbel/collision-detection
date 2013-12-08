#include "Broadphase.cuh"

// Test to see if the list of potential collisions from broadphase includes all of the ACTUAL collisions.

double getRandomNumber(double min, double max) {
	// x is in [0,1[
	double x = rand() / static_cast<double>(RAND_MAX);

	// [0,1[ * (max - min) + min is in [min,max[
	double that = min + (x * (max - min));

	return that;
}

struct Sphere {
		double r;
		real3 pos;
		uint index;
};

bool isColliding(Sphere sA, Sphere sB) {
	double dist = pow(sA.pos.x - sB.pos.x, 2.0) + pow(sA.pos.y - sB.pos.y, 2.0) + pow(sA.pos.z - sB.pos.z, 2.0);
	if (dist < pow(sA.r + sB.r, 2.0))
		return true;
	return false;
}

int main(int argc, char** argv) {
	double total_time = 0;
	for (int i = 0; i < 10; i++) {
		// Step 1: Generate random spheres
		uint numSpheres = 300000;
		double xMin = -30;
		double xMax = 30;
		double yMin = -30;
		double yMax = 30;
		double zMin = -30;
		double zMax = 30;
		double rMin = .1;
		double rMax = 1;

		vector<Sphere> spheres;

		Sphere sphere;
		for (int i = 0; i < numSpheres; i++) {
			sphere.pos = R3(getRandomNumber(xMin, xMax), getRandomNumber(yMin, yMax), getRandomNumber(zMin, zMax));
			sphere.r = getRandomNumber(rMin, rMax);
			sphere.index = i;

			//printf("Sphere %d pos = (%.3f, %.3f, %.3f), r = %.3f\n",i,sphere.pos.x,sphere.pos.y,sphere.pos.z,sphere.r);

			spheres.push_back(sphere);
			//positions.push_back(pos);
			//radii.push_back(r);
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
		broadphaseManager.setBinsPerAxis(I3(50, 50, 50));

		if (argc == 12) {
			broadphaseManager.setParallelConfiguration(
					atoi(argv[1]),
					atoi(argv[2]),
					atoi(argv[3]),
					atoi(argv[4]),
					atoi(argv[5]),
					atoi(argv[6]),
					atoi(argv[7]),
					atoi(argv[8]),
					atoi(argv[9]),
					atoi(argv[10]));
			omp_set_num_threads(atoi(argv[11]));
		} else {
			broadphaseManager.setParallelConfiguration(0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
		}

		//cout << "Begin parallel broadphase" << endl;
		double startTime = omp_get_wtime();
		broadphaseManager.detectPossibleCollisions(aabb_data, potentialCollisions);
		double endTime = omp_get_wtime();
		//printf("Time to detect: %lf seconds (%d possible collisions)\n", (endTime - startTime), broadphaseManager.getNumPossibleContacts());
		//cout << "End parallel broadphase\n" << endl;
		total_time += (endTime - startTime);
		if (i == 9) {
			cout << omp_get_max_threads() << " " << total_time / 10.0 << " " << broadphaseManager.getNumPossibleContacts() << " ";
			broadphaseManager.getParallelConfiguration();
			cout << endl;
		}

	}

	return 0;
}
