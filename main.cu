#include <iostream>
#include "KMeansUtil.cu"
#include "StastManager.cu"

#define NUMBER_TRIALS 25

void kMeansSeqVersion(KMeansStatsManager& statsManager, int numPoints, int numCentroids, int iterationTolerance, int threadsPerBlock);

void kMeansCUDAVersion(KMeansStatsManager& statsManager, int numPoints, int numCentroids, int iterationTolerance, int threadsPerBlock);

int main()
{
    srand(NULL);
    int numPointsDefault = 1000000;
    int numCentroidsDefault = 10;
    int numIterationDefault = 100;
    int threadsPerBlockDefault = 256;

    //NUM POINTS TEST
    cout << "INIT NUMBER POINT TEST\n";
    vector numPoints = {100, 1000, 10000, 100000, 1000000, 10000000};
    KMeansStatsManager numPointTestStatsManager;
    for (int numPoint : numPoints)
    {
        kMeansSeqVersion(numPointTestStatsManager, numPoint, numCentroidsDefault, numIterationDefault, threadsPerBlockDefault);

        kMeansCUDAVersion(numPointTestStatsManager, numPoint, numCentroidsDefault, numIterationDefault, threadsPerBlockDefault);
    }
    cout << "END NUMBER POINT TEST\n";
    // Print stats
    numPointTestStatsManager.printStats();
    // Export stats in csv
    numPointTestStatsManager.exportToCSV("C:/Users/plato/Downloads/pointsTest.csv");


    //NUM CENTROIDS TEST
    cout << "\nINIT NUMBER CENTROIDS TEST\n";
    vector numCentroids = {1, 10, 20, 30, 40, 50, 100};
    KMeansStatsManager numCentroidTestStatsManager;
    for (int numCentroid : numCentroids)
    {
        kMeansSeqVersion(numCentroidTestStatsManager, numPointsDefault, numCentroid, numIterationDefault, threadsPerBlockDefault);

        kMeansCUDAVersion(numCentroidTestStatsManager, numPointsDefault, numCentroid, numIterationDefault, threadsPerBlockDefault);
    }
    cout << "END NUMBER CENTROIDS TEST\n";
    // Print stats
    numCentroidTestStatsManager.printStats();
    // Export stats in csv
    numCentroidTestStatsManager.exportToCSV("C:/Users/plato/Downloads/centroidsTest.csv");

    //ITERATION TEST
    cout << "\nINIT NUMBER ITERATION TEST\n";
    vector numIterations = {50, 100, 200, 300, 400, 500};
    KMeansStatsManager iteTestStatsManager;
    for (int numIteration : numIterations)
    {
        kMeansSeqVersion(iteTestStatsManager, numPointsDefault, numCentroidsDefault, numIteration, threadsPerBlockDefault);

        kMeansCUDAVersion(iteTestStatsManager, numPointsDefault, numCentroidsDefault, numIteration, threadsPerBlockDefault);
    }
    cout << "\nEND NUMBER ITERATION TEST\n";
    // Print stats
    iteTestStatsManager.printStats();
    // Export stats in csv
    iteTestStatsManager.exportToCSV("C:/Users/plato/Downloads/iterationTest.csv");


    //THREAD PER BLOCK TEST
    cout << "\nINIT THREAD PER BLOCK TEST\n";
    vector numThreadsPerBlock = {16, 32, 64, 128, 256, 512, 1024};
    KMeansStatsManager threadTestStatsManager;
    kMeansSeqVersion(threadTestStatsManager, numPointsDefault, numCentroidsDefault, numIterationDefault,
                     threadsPerBlockDefault);
    for (int threads_per_block : numThreadsPerBlock)
    {
        kMeansCUDAVersion(threadTestStatsManager, numPointsDefault, numCentroidsDefault, numIterationDefault, threads_per_block);
    }
    cout << "\nEND THREAD PER BLOCK TEST\n";
    // Print stats
    threadTestStatsManager.printStats();
    // Export stats in csv
    threadTestStatsManager.exportToCSV("C:/Users/plato/Downloads/threadTest.csv");

    return 0;
}

void kMeansSeqVersion(KMeansStatsManager& statsManager, int numPoints, int numCentroids, int iterationTolerance, int threadsPerBlock)
{
    cout << "START SEQ VERSION with " << numPoints << " points, " << numCentroids << " centroids, " <<
        iterationTolerance << " iteration tolerance\n";
    auto startGlobal = high_resolution_clock::now();
    for (int trial = 0; trial < NUMBER_TRIALS; ++trial)
    {
        Points points;
        Centroids centroids;
        allocateAndInitializeSeq(points, centroids, numPoints, numCentroids);

        auto start = high_resolution_clock::now();
        kMeansSequential(points, centroids, numPoints, numCentroids, iterationTolerance);
        auto end = high_resolution_clock::now();

        freeSeqMemory(points, centroids);

        duration<double> seqDuration = end - start;
        statsManager.addSeqStats(numPoints, numCentroids, threadsPerBlock, iterationTolerance, seqDuration.count());
    }
    auto endGlobal = high_resolution_clock::now();
    duration<double> globalDuration = endGlobal - startGlobal;
    cout << "END SEQ VERSION in " << globalDuration.count() << "s\n\n";
}

void kMeansCUDAVersion(KMeansStatsManager& statsManager, int numPoints, int numCentroids, int iterationTolerance, int threadsPerBlock)
{
    cout << "START PAR VERSION with " << numPoints << " points, " << numCentroids << " centroids, " <<
        iterationTolerance << " iteration tolerance, " << threadsPerBlock << " thread per block\n";
    auto startGlobal = high_resolution_clock::now();
    for (int trial = 0; trial < NUMBER_TRIALS; ++trial)
    {
        Points points;
        Centroids centroids;
        allocateAndInitializeCuda(points, centroids, numPoints, numCentroids);

        auto start = high_resolution_clock::now();
        kMeansCUDA(points, centroids, numPoints, numCentroids, threadsPerBlock, iterationTolerance);
        auto end = high_resolution_clock::now();

        freeCudaMemory(points, centroids);

        duration<double> parDuration = end - start;
        statsManager.addParStats(numPoints, numCentroids, threadsPerBlock, iterationTolerance, parDuration.count());
    }
    auto endGlobal = high_resolution_clock::now();
    duration<double> globalDuration = endGlobal - startGlobal;
    cout << "END PAR VERSION with " << globalDuration.count() << "s\n\n";
}
