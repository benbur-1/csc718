#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>

#define RANGE_START 2
#define RANGE_END 1000000

// Function to check if a number is prime
int is_prime(int n) {
    if (n <= 1) return 0;
    if (n <= 3) return 1;
    if (n % 2 == 0 || n % 3 == 0) return 0;
    for (int i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return 0;
    }
    return 1;
}

// Function to find prime clusters
void find_prime_clusters(int start, int end, int *total_clusters, int *smallest_sum_cluster, int *last_two_primes) {
    int prev_prime = -1, prev_prev_prime = -1;
    *total_clusters = 0;
    *smallest_sum_cluster = INT_MAX;

    for (int i = start; i <= end; i++) {
        if (is_prime(i)) {
            if (prev_prev_prime != -1 && i - prev_prev_prime <= 6) {
                (*total_clusters)++;
                int cluster_sum = prev_prev_prime + prev_prime + i;
                if (cluster_sum < *smallest_sum_cluster) {
                    *smallest_sum_cluster = cluster_sum;
                }
            }
            prev_prev_prime = prev_prime;
            prev_prime = i;
        }
    }

    // Store the last two primes for boundary handling
    last_two_primes[0] = prev_prev_prime;
    last_two_primes[1] = prev_prime;
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int range_size = (RANGE_END - RANGE_START + 1) / size;
    int start = RANGE_START + rank * range_size;
    int end = (rank == size - 1) ? RANGE_END : start + range_size - 1;

    // Adjust for boundary overlap
    if (rank != 0) start -= 2;

    printf("Process %d handling range [%d, %d]\n", rank, start, end);

    double start_time = MPI_Wtime();

    int local_clusters, local_smallest_sum;
    int last_two_primes[2] = {-1, -1}, neighbor_primes[2] = {-1, -1};

    find_prime_clusters(start, end, &local_clusters, &local_smallest_sum, last_two_primes);

    // Communicate boundary primes with neighbors
    if (rank < size - 1) {
        MPI_Send(last_two_primes, 2, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
    }
    if (rank > 0) {
        MPI_Recv(neighbor_primes, 2, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // Check for clusters spanning ranges
        if (neighbor_primes[0] != -1 && neighbor_primes[1] != -1) {
            if (is_prime(start) && (start - neighbor_primes[1] <= 6)) {
                local_clusters++;
                int cluster_sum = neighbor_primes[0] + neighbor_primes[1] + start;
                if (cluster_sum < local_smallest_sum) {
                    local_smallest_sum = cluster_sum;
                }
            }
        }
    }

    int total_clusters, smallest_sum_cluster;
    MPI_Reduce(&local_clusters, &total_clusters, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_smallest_sum, &smallest_sum_cluster, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("\nTotal prime clusters found: %d\n", total_clusters);
        printf("Smallest-sum cluster: %d\n", smallest_sum_cluster);
        printf("Execution time: %f seconds\n", end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}
