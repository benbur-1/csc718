#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>

#define RANGE_START 2
#define RANGE_END 1000000

void sieveOfEratosthenes(int start, int end, int* primes) {
    int range_size = end - start + 1;
    int sqrt_end = sqrt(end);
    int* is_prime = malloc((sqrt_end + 1) * sizeof(int));

    for (int i = 0; i <= sqrt_end; i++) is_prime[i] = 1;
    for (int p = 2; p * p <= sqrt_end; p++) {
        if (is_prime[p]) {
            for (int j = p * p; j <= sqrt_end; j += p) {
                is_prime[j] = 0;
            }
        }
    }

    for (int i = 0; i < range_size; i++) primes[i] = 1;
    for (int p = 2; p <= sqrt_end; p++) {
        if (is_prime[p]) {
            int multiple = (start + p - 1) / p * p;
            if (multiple < p * p) multiple = p * p;
            for (int j = multiple; j <= end; j += p) {
                primes[j - start] = 0;
            }
        }
    }
    free(is_prime);
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int range_size = (RANGE_END - RANGE_START + 1) / size;
    int start = RANGE_START + rank * range_size;
    int end = (rank == size - 1) ? RANGE_END : start + range_size - 1;

    if (rank == 0) {
        printf("Searching for prime clusters in range [%d, %d] using %d processes.\n",
               RANGE_START, RANGE_END, size);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Process %d handling range [%d, %d]\n", rank, start, end);

    double start_time = MPI_Wtime();

    int* primes = malloc((end - start + 1) * sizeof(int));
    sieveOfEratosthenes(start, end, primes);

    // Further logic for detecting clusters and reducing results goes here.

    double end_time = MPI_Wtime();
    if (rank == 0) {
        printf("Execution time: %f seconds\n", end_time - start_time);
    }

    free(primes);
    MPI_Finalize();
    return 0;
}
