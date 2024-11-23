/*
 * Author: Ben Burgess
 * Date: 2024-11-23
 * Class: CSC 718 - Dakota State University
 * Email: ben.burgess@trojans.dsu.edu
 *
 * Description:
 * This program searches for prime clusters of size three within the range [2, 1000000] using MPI to parallelize the search.
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include <limits.h> // Add this line

#define RANGE_START 2
#define RANGE_END 1000000

// Function to mark non-prime numbers
void sieve_of_eratosthenes(bool *prime, int start, int end) {
    int limit = (int)sqrt(end);
    for (int i = 2; i <= limit; i++) {
        if (prime[i]) {
            for (int j = i * i; j <= end; j += i) {
                prime[j] = false;
            }
        }
    }
}

// Function to check and count prime clusters
int find_prime_clusters(bool *prime, int start, int end, int **clusters) {
    int count = 0;
    for (int i = start; i <= end - 2; i++) {
        if (prime[i] && prime[i + 2] && prime[i + 4]) {
            clusters[count] = (int *)malloc(3 * sizeof(int));
            clusters[count][0] = i;
            clusters[count][1] = i + 2;
            clusters[count][2] = i + 4;
            count++;
        }
    }
    return count;
}

int main(int argc, char **argv) {
    int rank, size;
    double start_time, end_time;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Determine sub-range for each process
    int range_size = (RANGE_END - RANGE_START + 1) / size;
    int sub_start = RANGE_START + rank * range_size;
    int sub_end = (rank == size - 1) ? RANGE_END : sub_start + range_size - 1;

    if (rank == 0) {
        printf("Searching for prime clusters in range [%d, %d] using %d processes.\n", RANGE_START, RANGE_END, size);
    }
    printf("Process %d handling range [%d, %d]\n", rank, sub_start, sub_end);

    // Allocate memory for the sieve
    bool *prime = (bool *)malloc((sub_end + 1) * sizeof(bool));
    for (int i = 0; i <= sub_end; i++) prime[i] = true;

    // Start timing
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // Perform sieve of Eratosthenes
    sieve_of_eratosthenes(prime, sub_start, sub_end);

    // Find prime clusters in the assigned range
    int **clusters = (int **)malloc(10000 * sizeof(int *));
    int local_cluster_count = find_prime_clusters(prime, sub_start, sub_end, clusters);

    // Gather total cluster counts from all processes
    int total_cluster_count;
    MPI_Reduce(&local_cluster_count, &total_cluster_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Gather smallest-sum cluster from all processes
    int local_smallest_sum = INT_MAX;
    int smallest_sum_cluster[3] = {0, 0, 0};

    for (int i = 0; i < local_cluster_count; i++) {
        int sum = clusters[i][0] + clusters[i][1] + clusters[i][2];
        if (sum < local_smallest_sum) {
            local_smallest_sum = sum;
            smallest_sum_cluster[0] = clusters[i][0];
            smallest_sum_cluster[1] = clusters[i][1];
            smallest_sum_cluster[2] = clusters[i][2];
        }
    }

    int global_smallest_sum_cluster[3];
    MPI_Reduce(smallest_sum_cluster, global_smallest_sum_cluster, 3, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    // End timing
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    // Print results
    if (rank == 0) {
        printf("Total prime clusters found: %d\n", total_cluster_count);
        printf("Smallest-sum cluster: {%d, %d, %d}\n", global_smallest_sum_cluster[0], global_smallest_sum_cluster[1], global_smallest_sum_cluster[2]);
        printf("Execution time: %.4f seconds\n", end_time - start_time);
    }

    // Free allocated memory
    for (int i = 0; i < local_cluster_count; i++) {
        free(clusters[i]);
    }
    free(clusters);
    free(prime);

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
