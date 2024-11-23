#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>

#define RANGE_START 2
#define RANGE_END 1000000

// Function to generate primes using the Sieve of Eratosthenes
void sieveOfEratosthenes(int start, int end, int *primes) {
    int range_size = end - start + 1;
    int sqrt_end = sqrt(end);
    int *is_prime = malloc((sqrt_end + 1) * sizeof(int));

    // Step 1: Generate small primes up to sqrt(end)
    for (int i = 0; i <= sqrt_end; i++) is_prime[i] = 1;
    for (int p = 2; p * p <= sqrt_end; p++) {
        if (is_prime[p]) {
            for (int j = p * p; j <= sqrt_end; j += p) {
                is_prime[j] = 0;
            }
        }
    }

    // Step 2: Mark multiples of small primes in the sub-range
    for (int i = 0; i < range_size; i++) primes[i] = 1;
    for (int p = 2; p <= sqrt_end; p++) {
        if (is_prime[p]) {
            int multiple = (start + p - 1) / p * p;  // First multiple in range
            if (multiple < p * p) multiple = p * p;
            for (int j = multiple; j <= end; j += p) {
                primes[j - start] = 0;
            }
        }
    }

    free(is_prime);
}

// Function to find prime clusters
void find_prime_clusters(int start, int end, int *primes, int *total_clusters, int *smallest_sum) {
    int prev_prime = -1, prev_prev_prime = -1;
    *total_clusters = 0;
    *smallest_sum = INT_MAX;

    for (int i = 0; i <= end - start; i++) {
        if (primes[i]) {
            int current_prime = start + i;

            if (prev_prev_prime != -1 && current_prime - prev_prev_prime <= 6) {
                (*total_clusters)++;
                int cluster_sum = prev_prev_prime + prev_prime + current_prime;
                if (cluster_sum < *smallest_sum) {
                    *smallest_sum = cluster_sum;
                }
            }

            prev_prev_prime = prev_prime;
            prev_prime = current_prime;
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int range_size = (RANGE_END - RANGE_START + 1) / size;
    int start = RANGE_START + rank * range_size;
    int end = (rank == size - 1) ? RANGE_END : start + range_size - 1;

    // Adjust boundaries to handle cross-boundary clusters
    if (rank != 0) start -= 2;

    if (rank == 0) {
        printf("Searching for prime clusters in range [%d, %d] using %d processes.\n", 
               RANGE_START, RANGE_END, size);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Process %d handling range [%d, %d]\n", rank, start, end);

    double start_time = MPI_Wtime();

    int *primes = malloc((end - start + 1) * sizeof(int));
    sieveOfEratosthenes(start, end, primes);

    int local_clusters, local_smallest_sum;
    find_prime_clusters(start, end, primes, &local_clusters, &local_smallest_sum);

    // Reduce results to rank 0
    int total_clusters, smallest_sum_cluster;
    MPI_Reduce(&local_clusters, &total_clusters, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_smallest_sum, &smallest_sum_cluster, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("\nTotal prime clusters found: %d\n", total_clusters);
        printf("Smallest-sum cluster: %d\n", smallest_sum_cluster);
        printf("Execution time: %f seconds\n", end_time - start_time);
    }

    free(primes);
    MPI_Finalize();
    return 0;
}
