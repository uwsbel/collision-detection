#include "Broadphase.cuh"
#include <assert.h>

// Test to see if the list of potential collisions from broadphase includes all of the ACTUAL collisions.

double getRandomNumber(double min, double max)
{
   // x is in [0,1[
   double x = rand()/static_cast<double>(RAND_MAX);

   // [0,1[ * (max - min) + min is in [min,max[
   double that = min + ( x * (max - min) );

   return that;
}

struct Sphere {
    double r;
    real3 pos;
    uint index;
};

bool isColliding(Sphere sA, Sphere sB)
{
	double dist = pow(sA.pos.x-sB.pos.x,2.0)+pow(sA.pos.y-sB.pos.y,2.0)+pow(sA.pos.z-sB.pos.z,2.0);
	if(dist<pow(sA.r+sB.r,2.0)) return true;
	return false;
}

int main(int argc, char** argv)
{
	// Step 1: Generate random spheres
	uint numSpheres = 300;
	double xMin = -30;
	double xMax = 30;
	double yMin = -30;
	double yMax = 30;
	double zMin = -30;
	double zMax = 30;
	double rMin = 1;
	double rMax = 20;

	vector<Sphere> spheres;

	Sphere sphere;
	for(int i=0;i<numSpheres;i++)
	{
		sphere.pos = make_real3(getRandomNumber(xMin,xMax),getRandomNumber(yMin,yMax),getRandomNumber(zMin,zMax));
		sphere.r = getRandomNumber(rMin,rMax);
		sphere.index = i;

		//printf("Sphere %d pos = (%.3f, %.3f, %.3f), r = %.3f\n",i,sphere.pos.x,sphere.pos.y,sphere.pos.z,sphere.r);

		spheres.push_back(sphere);
		//positions.push_back(pos);
		//radii.push_back(r);
	}
	// End Step 1

	// Step 2: Generate aabb_data from spheres
	custom_vector<real3> aabb_data;

	for(int i=0;i<spheres.size();i++)
	{
		real3 temp = spheres[i].pos;
		temp.x-=spheres[i].r;
		temp.y-=spheres[i].r;
		temp.z-=spheres[i].r;
		aabb_data.push_back(temp);
	}
	for(int i=0;i<spheres.size();i++)
	{
		real3 temp = spheres[i].pos;
		temp.x+=spheres[i].r;
		temp.y+=spheres[i].r;
		temp.z+=spheres[i].r;
		aabb_data.push_back(temp);
	}
	// End Step 2

	// Step 3: Run broadphase algorithm to find potential collisions

	custom_vector<long long> potentialCollisions;

	Broadphase broadphaseManager;
	broadphaseManager.setBinsPerAxis(make_real3(50,50,50));

	cout << "Begin parallel broadphase" << endl;
	double startTime = omp_get_wtime();
	broadphaseManager.detectPossibleCollisions(aabb_data, potentialCollisions);
	double endTime = omp_get_wtime();
	printf("Time to detect: %lf seconds (%d possible collisions)\n", (endTime - startTime),broadphaseManager.getNumPossibleContacts());
	cout << "End parallel broadphase\n" << endl;

	thrust::host_vector<long long> potentialCollisions_h = potentialCollisions;
	thrust::host_vector<int2> gpuCollisions;
	for (int i = 0; i < broadphaseManager.getNumPossibleContacts(); i++)
	{
		int2 collisionPair;
		collisionPair.x = int(potentialCollisions_h[i] >> 32);
		collisionPair.y = int(potentialCollisions_h[i] & 0xffffffff);
		gpuCollisions.push_back(collisionPair);
	}
	// End Step 3

	// Step 4: Perform collision detection using N-Squared approach (Detects ACTUAL collisions)
	cout << "Begin n-squared validation" << endl;
	double startTimeN2 = omp_get_wtime();
	thrust::host_vector<int2> nSquaredCollisions;
	for(int i=0;i<spheres.size();i++)
	{
		for(int j=i+1;j<spheres.size();j++)
		{
			if(isColliding(spheres[i],spheres[j]))
			{
				nSquaredCollisions.push_back(make_int2(i,j));
			}
		}
	}
	double endTimeN2 = omp_get_wtime();
	printf("Time to detect: %lf seconds (%d actual collisions)\n", (endTimeN2 - startTimeN2),nSquaredCollisions.size());
	cout << "End n-squared validation\n" << endl;
	// End Step 4

	// Step 5: Check to see if all the collisions are detected
	bool testFail = false;
	thrust::host_vector<int2> collisionListA = gpuCollisions; // what you want to check
	thrust::host_vector<int2> collisionListB = nSquaredCollisions; // what you think is correct
	thrust::host_vector<int2> badCollisions;
	for(int i=0;i<collisionListB.size();i++)
	{
		bool collisionFound = false;
		for(int j=0;j<collisionListA.size()&&collisionFound==false;j++)
		{
			if((collisionListB[i].x==collisionListA[j].x&&collisionListB[i].y==collisionListA[j].y)||(collisionListB[i].x==collisionListA[j].y&&collisionListB[i].y==collisionListA[j].x))
			{
				collisionFound = true;
			}
		}
		if(collisionFound==false)
		{
			testFail = true;
			badCollisions.push_back(collisionListB[i]);
		}

	}

	if(testFail)
	{
		printf("TEST FAILED (%d missed collisions)\n",badCollisions.size());
		printf("\tx \ty \tz \tr \tx \ty \tz \tr \tdelta\n");
		for(int i=0;i<badCollisions.size();i++)
		{
			real3 posA = spheres[badCollisions[i].x].pos;
			real3 posB = spheres[badCollisions[i].y].pos;
			real rA = spheres[badCollisions[i].x].r;
			real rB = spheres[badCollisions[i].y].r;

			real delta = sqrt(pow(posA.x-posB.x,2)+pow(posA.y-posB.y,2)+pow(posA.z-posB.z,2))-(rA+rB);

			printf("\t%.2f \t%.2f \t%.2f \t%.2f \t%.2f \t%.2f \t%.2f \t%.2f \t%.2f\n",posA.x,posA.y,posA.z,rA,posB.x,posB.y,posB.z,rB,delta);
		}
	}
	else
	{
		cout << "TEST PASSED" << endl;
	}
	// End Step 5

	return 0;
}
