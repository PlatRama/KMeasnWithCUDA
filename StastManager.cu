#include <iostream>
#include <vector>
#include <fstream>
#include <numeric>
#include <string>
#include <map>
#include <tuple>
#include <cmath>

using namespace std;

class KMeansStats
{
private:
    vector<double> seqTimes;
    vector<double> parTimes;

    double calculateMean(const vector<double>& times) const
    {
        return accumulate(times.begin(), times.end(), 0.0) / times.size();
    }

    double calculateStdDev(const vector<double>& times, double mean) const
    {
        double sum = 0.0;
        for (double time : times)
        {
            sum += (time - mean) * (time - mean);
        }
        return sqrt(sum / times.size());
    }

public:
    void addSeqTime(double time)
    {
        seqTimes.push_back(time);
    }

    void addParTime(double time)
    {
        parTimes.push_back(time);
    }

    double getSeqMeanTime() const
    {
        return calculateMean(seqTimes);
    }

    double getParMeanTime() const
    {
        return calculateMean(parTimes);
    }

    double getSeqStdDev() const
    {
        return calculateStdDev(seqTimes, getSeqMeanTime());
    }

    double getParStdDev() const
    {
        return calculateStdDev(parTimes, getParMeanTime());
    }

    double getSpeedup() const
    {
        return getSeqMeanTime() / getParMeanTime();
    }

    double getEfficiency(int numCudaThreads) const
    {
        return getSpeedup() / numCudaThreads;
    }
};


class KMeansStatsManager
{
private:
    // Map using a triple (numPoints, numCentroids, numCudaThreads) as a key
    map<tuple<int, int, int, int>, KMeansStats> statsMap;

public:
    void addStats(int numPoints, int numCentroids, int numCudaThreads, int iterationTolerance, double seqTime, double parTime)
    {
        auto key = make_tuple(numPoints, numCentroids, numCudaThreads, iterationTolerance);
        statsMap[key].addSeqTime(seqTime);
        statsMap[key].addParTime(parTime);
    }

    void addSeqStats(int numPoints, int numCentroids, int numCudaThreads, int iterationTolerance, double seqTime)
    {
        auto key = make_tuple(numPoints, numCentroids, numCudaThreads, iterationTolerance);
        statsMap[key].addSeqTime(seqTime);
    }

    void addParStats(int numPoints, int numCentroids, int numCudaThreads, int iterationTolerance, double parTime)
    {
        auto key = make_tuple(numPoints, numCentroids, numCudaThreads, iterationTolerance);
        statsMap[key].addParTime(parTime);
    }

    void printStats() const
    {
        for (const auto& entry : statsMap)
        {
            auto key = entry.first;
            const KMeansStats& stats = entry.second;
            int numPoints = get<0>(key);
            int numCentroids = get<1>(key);
            int numCudaThreads = get<2>(key);
            int iterationTolerance = get<3>(key);

            cout << "Number of Points: " << numPoints << endl;
            cout << "Number of Centroids: " << numCentroids << endl;
            cout << "Number of CUDA Threads: " << numCudaThreads << endl;
            cout << "Number of iteration tolerance: " << iterationTolerance << endl;
            cout << "Sequential Mean Time: " << stats.getSeqMeanTime() << "s" << endl;
            cout << "Sequential StdDev: " << stats.getSeqStdDev() << "s" << endl;
            cout << "Parallel Mean Time: " << stats.getParMeanTime() << "s" << endl;
            cout << "Parallel StdDev: " << stats.getParStdDev() << "s" << endl;
            cout << "Speedup: " << stats.getSpeedup() << endl;
            cout << "Efficiency: " << stats.getEfficiency(numCudaThreads) << endl;
            cout << "------------------------" << endl;
        }
    }

    void exportToCSV(const string& filename) const
    {
        ofstream file;
        file.open(filename);

        // Intestazione del CSV
        file <<
            "NumPoints,NumCentroids,NumCudaThreads,SeqMeanTime,SeqStdDev,ParMeanTime,ParStdDev,Speedup,Efficiency\n";

        for (const auto& entry : statsMap)
        {
            auto key = entry.first;
            const KMeansStats& stats = entry.second;
            int numPoints = get<0>(key);
            int numCentroids = get<1>(key);
            int numCudaThreads = get<2>(key);

            // Scrivi le statistiche nel file CSV
            file << numPoints << "," << numCentroids << "," << numCudaThreads << ","
                << stats.getSeqMeanTime() << "," << stats.getSeqStdDev() << ","
                << stats.getParMeanTime() << "," << stats.getParStdDev() << ","
                << stats.getSpeedup() << "," << stats.getEfficiency(numCudaThreads) << "\n";
        }

        file.close();
        cout << "Data exported to " << filename << endl;
    }
};
