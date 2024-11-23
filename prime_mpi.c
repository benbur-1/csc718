#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define GLOBAL_START 2
#define GLOBAL_END 1000000

// Function to perform the Sieve of Eratosthenes within a range
void sieveOfEratosthenes(int start, int end, int* primes) {
    for (int i = 0; i <= end - start; i++) primes[i] = 1;  // Mark all as potential primes
    int limit = sqrt(end);

    for (int i = 2; i <= limit; i++) {
        for (int j = fmax(i * i, (start + i - 1) / i * i); j <= end; j += i) {
            primes[j - start] = 0;  // Mark as non-prime
        }
    }
}

// Main function
int main(int argc, char** argv) {
    int rank, size, rangeStart, rangeEnd, localClusters = 0, totalClusters = 0;
    int smallestCluster[3] = {0, 0, 0}, globalSmallestCluster[3] = {0, 0, 0};
    double startTime, endTime;

    MPI_Init(&argc, &argv);  // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Get total number of processes

    int rangeSize = (GLOBAL_END - GLOBAL_START + 1) / size;  // Calculate range per process

    // Define the range for this process
    rangeStart = GLOBAL_START + rank * rangeSize;
    rangeEnd = (rank == size - 1) ? GLOBAL_END : rangeStart + rangeSize - 1;

    // Adjust boundaries to handle cross-boundary clusters
    if (rank != 0) rangeStart -= 2;

    printf("Process %d handling range [%d, %d]\n", rank, rangeStart, rangeEnd);

    int* primes = malloc((rangeEnd - rangeStart + 1) * sizeof(int));
    if (!primes) {
        fprintf(stderr, "Memory allocation failed for process %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    startTime = MPI_Wtime();  // Start timing

    // Perform the sieve
    sieveOfEratosthenes(rangeStart, rangeEnd, primes);

    // Find prime clusters
    for (int i = 2; i < rangeEnd - rangeStart - 4; i++) {
        if (primes[i] && primes[i + 2] && primes[i + 4]) {
            localClusters++;
            int clusterSum = (rangeStart + i) + (rangeStart + i + 2) + (rangeStart + i + 4);
            if (smallestCluster[0] == 0 || clusterSum < (smallestCluster[0] + smallestCluster[1] + smallestCluster[2])) {
                smallestCluster[0] = rangeStart + i;
                smallestCluster[1] = rangeStart + i + 2;
                smallestCluster[2] = rangeStart + i + 4;
            }
        }
    }

    // Reduce local results to the global results
    MPI_Reduce(&localClusters, &totalClusters, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(smallestCluster, globalSmallestCluster, 3, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    endTime = MPI_Wtime();  // End timing

    // Output results
    if (rank == 0) {
        printf("\nTotal prime clusters found: %d\n", totalClusters);
        printf("Smallest-sum cluster: {%d, %d, %d}\n", globalSmallestCluster[0], globalSmallestCluster[1], globalSmallestCluster[2]);
        printf("Execution time: %f seconds\n", endTime - startTime);
    }

    free(primes);
    MPI_Finalize();  // Finalize MPI
    return 0;
}
