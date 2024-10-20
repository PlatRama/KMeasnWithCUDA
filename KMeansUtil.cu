#include <cstdlib>
#include <cfloat>
#include <chrono>

using namespace std;
using namespace chrono;


// Struct representing a set of points in a 2D space.
// Uses an Array of Structures (AoS) approach.
// Each array stores a component of the point (x, y) and the index of the closest centroid.
struct Points {
    float *x;
    float *y;
    int *closestCentroid;     // Array storing the index of the closest centroid for each point
};

// Struct representing a set of centroids used in the k-means algorithm.
// Uses an Array of Structures (AoS) approach.
// Also includes fields to calculate the new centroid positions by summing the coordinates of the assigned points.
struct Centroids {
    float *x;
    float *y;
    int *pointCount;         // Array storing the number of points assigned to each centroid
    float *sumX;             // Sum of x coordinates of points assigned to each centroid (for recalculating positions)
    float *sumY;             // Sum of y coordinates of points assigned to each centroid (for recalculating positions)
};

void kMeansSequential(Points &points, Centroids &centroids, int numPoints, int numCentroids, int iterationTolerance);

void kMeansCUDA(Points points, Centroids centroids, int numPoints, int numCentroids, int threadsPerBlock, int iterationTolerance);

void allocateAndInitializeCuda(Points &points, Centroids &centroids, int numPoints, int numCentroids);

void allocateAndInitializeSeq(Points &points, Centroids &centroids, int numPoints, int numCentroids);

void freeCudaMemory(Points &points, Centroids &centroids);

void freeSeqMemory(Points &points, Centroids &centroids);

/**
 * @brief Executes the k-means algorithm sequentially.
 *
 * This method implements the k-means algorithm to cluster a set of points into a specified number of centroids.
 * The algorithm iterates until the centroids converge, meaning the shift in centroids is less than a given tolerance.
 * In each iteration, every point is assigned to the nearest centroid, and the centroids are updated based on the average
 * coordinates of the assigned points. The process continues until the centroids stabilize.
 *
 * @param points Reference to a structure containing the coordinates of the points.
 * @param centroids Reference to a structure containing the coordinates of the centroids.
 * @param numPoints The total number of points in the dataset.
 * @param numCentroids The number of centroids used for clustering.
 * @param iterationTolerance The iteration tolerance threshold for determining the end of the k-means.
 */
void kMeansSequential(Points &points, Centroids &centroids, int numPoints, int numCentroids, int iterationTolerance) {
    int iteration = 0;

    // Main loop continues until centroids converge
    while (iteration < iterationTolerance) {
        // Reset centroids accumulators for new iteration
        for (int i = 0; i < numCentroids; ++i) {
            centroids.sumX[i] = 0.0f;
            centroids.sumY[i] = 0.0f;
            centroids.pointCount[i] = 0;
        }

        // Assign points to the nearest centroid
        for (int i = 0; i < numPoints; ++i) {
            float minDist = FLT_MAX; // Start with maximum possible distance
            int closest = -1;

            // Find the closest centroid to the current point
            for (int j = 0; j < numCentroids; ++j) {
                // Calculate squared Euclidean distance between point and centroid
                float dx = points.x[i] - centroids.x[j];
                float dy = points.y[i] - centroids.y[j];
                float dist = (dx * dx) + (dy * dy);

                //Checks whether the centroid is closer than the last one
                if (dist < minDist) {
                    minDist = dist;
                    closest = j;
                }
            }

            // Update the closest centroid's accumulators
            points.closestCentroid[i] = closest;
            centroids.sumX[closest] += points.x[i];
            centroids.sumY[closest] += points.y[i];
            centroids.pointCount[closest] += 1;
        }

        // Update centroids based on the mean of assigned points
        for (int i = 0; i < numCentroids; ++i) {
            if (centroids.pointCount[i] > 0) {
                // Calculate the new centroid position
                float newX = centroids.sumX[i] / centroids.pointCount[i];
                float newY = centroids.sumY[i] / centroids.pointCount[i];

                // Update centroid position
                centroids.x[i] = newX;
                centroids.y[i] = newY;
            }
        }
        iteration++;
    }

    //cout << "K-means sequential finished in " << iteration << " iterations.\n";
}


/**
 * Assign each point to the closest centroid and update centroid accumulators.
 *
 * @param points Struct containing point coordinates and the assigned closest centroid.
 * @param centroids Struct containing centroid coordinates, accumulators for new centroids, and the count of points.
 * @param numPoints Total number of points.
 * @param numCentroids Total number of centroids.
 */
__global__ void assignCentroids(Points points, Centroids centroids, int numPoints, int numCentroids) {
    // Compute global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure thread index is within bounds
    if (idx < numPoints) {
        float minDist = FLT_MAX;  // Start with maximum possible distance
        int closest = -1;

        // Loop over all centroids to find the closest one
        for (int i = 0; i < numCentroids; ++i) {
            // Calculate squared Euclidean distance between point and centroid
            float dx = points.x[idx] - centroids.x[i];
            float dy = points.y[idx] - centroids.y[i];
            float dist = (dx * dx) + (dy * dy);

            //Checks whether the centroid is closer than the last one
            if (dist < minDist) {
                minDist = dist;
                closest = i;
            }
        }

        // Assign the closest centroid to the current point
        points.closestCentroid[idx] = closest;

        // Atomically update centroid accumulators to avoid race conditions
        atomicAdd(&centroids.sumX[closest], points.x[idx]);
        atomicAdd(&centroids.sumY[closest], points.y[idx]);
        atomicAdd(&centroids.pointCount[closest], 1);
    }
}

/**
 * Update the centroids based on accumulated sums and reset accumulators.
 *
 * @param centroids Struct containing centroid coordinates, accumulators for new centroids, and point counts.
 * @param numCentroids Total number of centroids.
 */
__global__ void updateCentroids(Centroids centroids, int numCentroids) {
    // Compute global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure thread index is within bounds
    if (idx < numCentroids) {
        // Only update centroids with at least one assigned point
        if (centroids.pointCount[idx] > 0) {
            // Calculate new centroid position
            float newX = centroids.sumX[idx] / centroids.pointCount[idx];
            float newY = centroids.sumY[idx] / centroids.pointCount[idx];

            // Update centroid coordinates
            centroids.x[idx] = newX;
            centroids.y[idx] = newY;
        }

        // Reset the accumulators for the next iteration
        centroids.sumX[idx] = 0.0f;
        centroids.sumY[idx] = 0.0f;
        centroids.pointCount[idx] = 0;
    }
}

void kMeansCUDA(Points points, Centroids centroids, int numPoints, int numCentroids, int threadsPerBlock, int iterationTolerance) {
    int numBlocksPoints = (numPoints + threadsPerBlock - 1) / threadsPerBlock;
    int numBlocksCentroids = (numCentroids + threadsPerBlock - 1) / threadsPerBlock;

    int iteration = 0;
    while (iteration < iterationTolerance) {
        assignCentroids<<<numBlocksPoints, threadsPerBlock>>>(points, centroids, numPoints, numCentroids);
        cudaDeviceSynchronize();

        updateCentroids<<<numBlocksCentroids, threadsPerBlock>>>(centroids, numCentroids);
        cudaDeviceSynchronize();

        iteration++;
    }
    //cout << "K-means parallel finished in " << iteration << " iterations.\n";
}


void allocateCudaMemory(Points &points, Centroids &centroids, int numPoints, int numCentroids) {
    //Allocation memory for CUDA
    cudaMallocManaged(&points.x, numPoints * sizeof(float));
    cudaMallocManaged(&points.y, numPoints * sizeof(float));
    cudaMallocManaged(&points.closestCentroid, numPoints * sizeof(int));

    cudaMallocManaged(&centroids.x, numCentroids * sizeof(float));
    cudaMallocManaged(&centroids.y, numCentroids * sizeof(float));
    cudaMallocManaged(&centroids.pointCount, numCentroids * sizeof(int));
    cudaMallocManaged(&centroids.sumX, numCentroids * sizeof(float));
    cudaMallocManaged(&centroids.sumY, numCentroids * sizeof(float));
}

void allocateSeqMemory(Points &points, Centroids &centroids, int numPoints, int numCentroids) {
    //Allocation memory for CPU
    points.x = (float*)malloc(numPoints * sizeof(float));
    points.y = (float*)malloc(numPoints * sizeof(float));
    points.closestCentroid = (int*)malloc(numPoints * sizeof(int));

    centroids.x = (float*)malloc(numCentroids * sizeof(float));
    centroids.y = (float*)malloc(numCentroids * sizeof(float));
    centroids.pointCount = (int*)malloc(numCentroids * sizeof(int));
    centroids.sumX = (float*)malloc(numCentroids * sizeof(float));
    centroids.sumY = (float*)malloc(numCentroids * sizeof(float));
}

void initializeData(Points &points, Centroids &centroids, int numPoints, int numCentroids) {
    // Generate random points
    for (int i = 0; i < numPoints; ++i) {
        points.x[i] = static_cast<float>(rand()) / RAND_MAX * 100.0f;
        points.y[i] = static_cast<float>(rand()) / RAND_MAX * 100.0f;
        points.closestCentroid[i] = -1;
    }

    // Generate random centroids
    for (int i = 0; i < numCentroids; ++i) {
        centroids.x[i] = static_cast<float>(rand()) / RAND_MAX * 100.0f;
        centroids.y[i] = static_cast<float>(rand()) / RAND_MAX * 100.0f;
        centroids.pointCount[i] = 0;
        centroids.sumX[i] = 0.0f;
        centroids.sumY[i] = 0.0f;
    }
}

void freeCudaMemory(Points &points, Centroids &centroids) {
    cudaFree(points.x);
    cudaFree(points.y);
    cudaFree(points.closestCentroid);

    cudaFree(centroids.x);
    cudaFree(centroids.y);
    cudaFree(centroids.pointCount);
    cudaFree(centroids.sumX);
    cudaFree(centroids.sumY);
}

void freeSeqMemory(Points &points, Centroids &centroids) {
    free(points.x);
    free(points.y);
    free(points.closestCentroid);

    free(centroids.x);
    free(centroids.y);
    free(centroids.pointCount);
    free(centroids.sumX);
    free(centroids.sumY);
}

void allocateAndInitializeCuda(Points &points, Centroids &centroids, int numPoints, int numCentroids) {
    allocateCudaMemory(points, centroids, numPoints, numCentroids);
    initializeData(points, centroids, numPoints, numCentroids);
}

void allocateAndInitializeSeq(Points &points, Centroids &centroids, int numPoints, int numCentroids) {
    allocateSeqMemory(points, centroids, numPoints, numCentroids);
    initializeData(points, centroids, numPoints, numCentroids);
}