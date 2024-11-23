#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <limits.h>

// Function to mark non-prime numbers using the Sieve of Eratosthenes
void sieve_of_eratosthenes(bool *is_prime, int start, int end, int global_start) {
    int limit = (int)sqrt(end);
    for (int i = 2; i <= limit; i++) {
        if (is_prime[i]) {
            int first_multiple = (start / i) * i;
            if (first_multiple < start) first_multiple += i;
            if (first_multiple == i) first_multiple += i;

            for (int j = first_multiple; j <= end; j += i) {
                is_prime[j - global_start] = false;
            }
        }
    }
}

// Function to find prime clusters {p, p+2, p+4}
int find_prime_clusters(int *primes, int prime_count, int **clusters) {
    int count = 0;
    for (int i = 0; i < prime_count - 2; i++) {
        if (primes[i + 1] - primes[i] == 2 && primes[i + 2] - primes[i + 1] == 2) {
            clusters[count] = (int *)malloc(3 * sizeof(int));
            clusters[count][0] = primes[i];
            clusters[count][1] = primes[i + 1];
            clusters[count][2] = primes[i + 2];
            count++;
        }
    }
    return count;
}

int main(int argc, char *argv[]) {
    int rank, size;
    int global_start = 2, global_end = 1000000;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Divide the range among processes
    int range_size = (global_end - global_start + 1) / size;
    int local_start = global_start + rank * range_size;
    int local_end = (rank == size - 1) ? global_end : local_start + range_size - 1;

    if (rank == 0) {
        printf("Searching for prime clusters in range [%d, %d] using %d processes.\n", global_start, global_end, size);
    }
    printf("Process %d handling range [%d, %d]\n", rank, local_start, local_end);

    // Start timing
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // Allocate and initialize sieve array
    int local_range_size = local_end - local_start + 1;
    bool *is_prime = (bool *)malloc(local_range_size * sizeof(bool));
    for (int i = 0; i < local_range_size; i++) is_prime[i] = true;

    // Perform sieve of Eratosthenes
    sieve_of_eratosthenes(is_prime, local_start, local_end, local_start);

    // Collect primes into an array
    int *local_primes = (int *)malloc(local_range_size * sizeof(int));
    int prime_count = 0;
    for (int i = 0; i < local_range_size; i++) {
        if (is_prime[i]) local_primes[prime_count++] = local_start + i;
    }

    // Find prime clusters
    int **clusters = (int **)malloc(prime_count * sizeof(int *));
    int local_cluster_count = find_prime_clusters(local_primes, prime_count, clusters);

    // Handle boundary condition: exchange edge primes
    int edge_primes[2] = {0, 0};
    if (prime_count > 1) {
        edge_primes[0] = local_primes[prime_count - 2];
        edge_primes[1] = local_primes[prime_count - 1];
    }
    if (rank < size - 1) {
        MPI_Send(edge_primes, 2, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
    }
    if (rank > 0) {
        int neighbor_primes[2];
        MPI_Recv(neighbor_primes, 2, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (neighbor_primes[1] - neighbor_primes[0] == 2 && local_primes[0] - neighbor_primes[1] == 2) {
            clusters[local_cluster_count] = (int *)malloc(3 * sizeof(int));
            clusters[local_cluster_count][0] = neighbor_primes[0];
            clusters[local_cluster_count][1] = neighbor_primes[1];
            clusters[local_cluster_count][2] = local_primes[0];
            local_cluster_count++;
        }
    }

    // Gather total cluster count
    int total_clusters;
    MPI_Reduce(&local_cluster_count, &total_clusters, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Find smallest-sum cluster
    int local_smallest_sum = INT_MAX;
    int smallest_cluster[3] = {0, 0, 0};
    for (int i = 0; i < local_cluster_count; i++) {
        int sum = clusters[i][0] + clusters[i][1] + clusters[i][2];
        if (sum < local_smallest_sum) {
            local_smallest_sum = sum;
            smallest_cluster[0] = clusters[i][0];
            smallest_cluster[1] = clusters[i][1];
            smallest_cluster[2] = clusters[i][2];
        }
    }

    int global_smallest_cluster[3];
    MPI_Reduce(smallest_cluster, global_smallest_cluster, 3, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    // End timing
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    // Output results
    if (rank == 0) {
        printf("Total prime clusters found: %d\n", total_clusters);
        printf("Smallest-sum cluster: {%d, %d, %d}\n", global_smallest_cluster[0], global_smallest_cluster[1], global_smallest_cluster[2]);
        printf("Execution time: %.4f seconds\n", end_time - start_time);
    }

    // Free allocated memory
    for (int i = 0; i < local_cluster_count; i++) free(clusters[i]);
    free(clusters);
    free(local_primes);
    free(is_prime);

    MPI_Finalize();
    return 0;
}
