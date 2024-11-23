#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

// Utility macros for block decomposition
#define BLOCK_LOW(id, p, n) ((id) * (n) / (p))
#define BLOCK_HIGH(id, p, n) (BLOCK_LOW((id) + 1, p, n) - 1)
#define BLOCK_SIZE(id, p, n) (BLOCK_LOW((id) + 1, p, n) - BLOCK_LOW(id, p, n))

// Sieve function
void sieve_of_eratosthenes(bool *is_prime, int low, int high, int sqrt_n) {
    int i, j;
    for (i = 2; i <= sqrt_n; i++) {
        if (is_prime[i]) {
            int start = (low % i == 0) ? low : (low / i + 1) * i;
            if (start == i) start += i; // Skip marking the number itself
            #pragma omp parallel for
            for (j = start; j <= high; j += i) {
                is_prime[j - low] = false;
            }
        }
    }
}

// Detect prime clusters
int detect_clusters(int *primes, int prime_count, int **clusters) {
    int cluster_count = 0;
    for (int i = 0; i < prime_count - 2; i++) {
        if (primes[i + 1] - primes[i] == 2 && primes[i + 2] - primes[i + 1] == 2) {
            clusters[cluster_count] = (int *)malloc(3 * sizeof(int));
            clusters[cluster_count][0] = primes[i];
            clusters[cluster_count][1] = primes[i + 1];
            clusters[cluster_count][2] = primes[i + 2];
            cluster_count++;
        }
    }
    return cluster_count;
}

int main(int argc, char *argv[]) {
    int rank, size;
    int global_low = 2, global_high = 1000000;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int range = global_high - global_low + 1;
    int local_low = global_low + BLOCK_LOW(rank, size, range);
    int local_high = global_low + BLOCK_HIGH(rank, size, range);
    int local_range = local_high - local_low + 1;

    if (rank == 0) {
        printf("Searching for prime clusters in range [%d, %d] using %d processes.\n", global_low, global_high, size);
    }
    printf("Process %d handling range [%d, %d]\n", rank, local_low, local_high);

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // Initialize sieve array
    bool *is_prime = (bool *)malloc(local_range * sizeof(bool));
    for (int i = 0; i < local_range; i++) is_prime[i] = true;

    // Broadcast small primes from process 0
    int sqrt_n = (int)sqrt(global_high);
    bool *small_prime = NULL;
    if (rank == 0) {
        small_prime = (bool *)malloc((sqrt_n + 1) * sizeof(bool));
        for (int i = 0; i <= sqrt_n; i++) small_prime[i] = true;
        sieve_of_eratosthenes(small_prime, 2, sqrt_n, sqrt_n);
    }
    MPI_Bcast(small_prime, sqrt_n + 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    // Perform sieve on local range
    sieve_of_eratosthenes(is_prime, local_low, local_high, sqrt_n);

    // Gather primes
    int *local_primes = (int *)malloc(local_range * sizeof(int));
    int prime_count = 0;
    for (int i = 0; i < local_range; i++) {
        if (is_prime[i]) {
            local_primes[prime_count++] = local_low + i;
        }
    }

    // Detect clusters
    int **clusters = (int **)malloc(prime_count * sizeof(int *));
    int local_cluster_count = detect_clusters(local_primes, prime_count, clusters);

    // Reduce cluster counts
    int total_clusters = 0;
    MPI_Reduce(&local_cluster_count, &total_clusters, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Collect execution time
    end_time = MPI_Wtime();

    // Process 0 displays results
    if (rank == 0) {
        printf("Total prime clusters found: %d\n", total_clusters);
        printf("Execution time: %.4f seconds\n", end_time - start_time);
    }

    // Free memory
    for (int i = 0; i < local_cluster_count; i++) free(clusters[i]);
    free(clusters);
    free(local_primes);
    free(is_prime);
    if (rank == 0) free(small_prime);

    MPI_Finalize();
    return 0;
}
