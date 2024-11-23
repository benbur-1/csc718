#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <limits.h>

// Function to generate prime numbers using the Sieve of Eratosthenes
int* generatePrimes(int start, int end) {
    int* isPrime = (int*)malloc((end - start + 1) * sizeof(int));
    for (int i = 0; i <= end - start; i++) {
        isPrime[i] = 1;
    }

    for (int p = 2; p * p <= end; p++) {
        if (isPrime[p - start]) {
            for (int i = fmax(p * p, (start + p - 1) / p * p); i <= end; i += p) {
                isPrime[i - start] = 0;
            }
        }
    }

    int count = 0;
    for (int p = 2; p <= end; p++) {
        if (isPrime[p - start] && p >= start) {
            count++;
        }
    }

    int* primes = (int*)malloc(count * sizeof(int));
    int index = 0;
    for (int p = 2; p <= end; p++) {
        if (isPrime[p - start] && p >= start) {
            primes[index++] = p;
        }
    }

    free(isPrime);
    return primes;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rangeStart = 2;
    int rangeEnd = 1000000;
    int rangeSize = rangeEnd - rangeStart + 1;
    int chunkSize = rangeSize / size;

    int start = rangeStart + rank * chunkSize;
    int end = (rank == size - 1) ? rangeEnd : start + chunkSize - 1;

    if (rank == 0) {
        printf("Searching for prime clusters in range [%d, %d] using %d processes.\n", rangeStart, rangeEnd, size);
    }

    printf("Process %d handling range [%d, %d]\n", rank, start, end);

    clock_t startTime = clock();

    int* primes = generatePrimes(start, end);

    // Boundary exchange logic (if needed)
    // ...

    int totalClusters = 0;
    int minClusterSum = INT_MAX;
    int minCluster[3];

    int primeCount = 0;
    for (int i = 0; primes[i] != 0; i++) {
        primeCount++;
    }

    for (int i = 0; i < primeCount - 2; i++) {
        if (primes[i + 2] - primes[i] == 4) {
            totalClusters++;
            if (primes[i] + primes[i + 1] + primes[i + 2] < minClusterSum) {
                minClusterSum = primes[i] + primes[i + 1] + primes[i + 2];
                minCluster[0] = primes[i];
                minCluster[1] = primes[i + 1];
                minCluster[2] = primes[i + 2];
            }
        }
    }

    // Gather results from all processes (totalClusters and minCluster)
    // ...

    clock_t endTime = clock();
    double executionTime = double(endTime - startTime) / CLOCKS_PER_SEC;

    if (rank == 0) {
        printf("Total prime clusters found: %d\n", totalClusters);
        printf("Smallest-sum cluster: {%d, %d, %d}\n", minCluster[0], minCluster[1], minCluster[2]);
        printf("Execution time: %f seconds\n", executionTime);
    }

    free(primes);
    MPI_Finalize();
    return 0;
}
