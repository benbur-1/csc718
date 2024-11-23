#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <limits.h>

// Function prototypes
void sieve_of_eratosthenes(bool *is_prime, int start, int end, int global_start);
int find_prime_clusters(int *primes, int prime_count, int **clusters, int *smallest_sum_cluster);

int main(int argc, char *argv[]) {
    int rank, size;
    int global_start = 2, global_end = 1000000;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Divide the range among processes
    int range = global_end - global_start + 1;
    int local_start = global_start + (rank * range / size);
    int local_end = global_start + ((rank + 1) * range / size) - 1;
    if (rank == size - 1) local_end = global_end;

    int local_range = local_end - local_start + 1;

    if (rank == 0) {
        printf("Searching for prime clusters in range [%d, %d] using %d processes.\n", global_start, global_end, size);
    }
    printf("Process %d handling range [%d, %d]\n", rank, local_start, local_end);

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // Allocate memory for sieve
    bool *is_prime = (bool *)malloc(local_range * sizeof(bool));
    for (int i = 0; i < local_range; i++) is_prime[i] = true;

    // Perform sieve
    sieve_of_eratosthenes(is_prime, local_start, local_end, global_start);

    // Collect primes
    int *local_primes = (int *)malloc(local_range * sizeof(int));
    int prime_count = 0;
    for (int i = 0; i < local_range; i++) {
        if (is_prime[i]) {
            local_primes[prime_count++] = local_start + i;
        }
    }

    // Handle boundary primes
    if (rank > 0) {
        int previous_prime;
        MPI_Recv(&previous_prime, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (previous_prime != -1 && local_primes[0] - previous_prime <= 10) {
            if (prime_count >= 2 && local_primes[1] - local_primes[0] <= 10) {
                prime_count++;
            }
        }
    }
    if (rank < size - 1) {
        int last_prime = (prime_count > 0) ? local_primes[prime_count - 1] : -1;
        MPI_Send(&last_prime, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
    }

    // Detect clusters
    int **clusters = (int **)malloc(prime_count * sizeof(int *));
    int *smallest_sum_cluster = (int *)malloc(3 * sizeof(int));
    int local_cluster_count = find_prime_clusters(local_primes, prime_count, clusters, smallest_sum_cluster);

    // Reduce cluster counts
    int total_clusters;
    MPI_Reduce(&local_cluster_count, &total_clusters, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Reduce smallest-sum cluster
    int global_smallest_cluster[3];
    MPI_Reduce(smallest_sum_cluster, global_smallest_cluster, 3, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    // Collect execution time
    end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Total prime clusters found: %d\n", total_clusters);
        printf("Smallest-sum cluster: {%d, %d, %d}\n", global_smallest_cluster[0], global_smallest_cluster[1], global_smallest_cluster[2]);
        printf("Execution time: %.4f seconds\n", end_time - start_time);
    }

    // Free memory
    for (int i = 0; i < local_cluster_count; i++) free(clusters[i]);
    free(clusters);
    free(local_primes);
    free(is_prime);
    free(smallest_sum_cluster);

    MPI_Finalize();
    return 0;
}

// Function to perform the Sieve of Eratosthenes
void sieve_of_eratosthenes(bool *is_prime, int start, int end, int global_start) {
    int sqrt_end = (int)sqrt(end) + 1;

    // Mark non-primes in the [2, sqrt(end)] range
    bool *small_primes = (bool *)malloc((sqrt_end + 1) * sizeof(bool));
    for (int i = 0; i <= sqrt_end; i++) small_primes[i] = true;
    for (int i = 2; i <= sqrt_end; i++) {
        if (small_primes[i]) {
            for (int j = i * i; j <= sqrt_end; j += i) {
                small_primes[j] = false;
            }
        }
    }

    // Mark non-primes in the [start, end] range using small primes
    for (int i = 2; i <= sqrt_end; i++) {
        if (small_primes[i]) {
            int first_multiple = (start / i) * i;
            if (first_multiple < start) first_multiple += i;
            if (first_multiple == i) first_multiple += i;
            for (int j = first_multiple; j <= end; j += i) {
                is_prime[j - start] = false;
            }
        }
    }

    free(small_primes);
}
