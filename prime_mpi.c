#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
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
void find_prime_clusters(int start, int end, int *total_clusters, int *smallest_sum_cluster) {
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
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int range_size = (RANGE_END - RANGE_START + 1) / size;
    int start = RANGE_START + rank * range_size;
    int end = (rank == size - 1) ? RANGE_END : start + range_size - 1;

    printf("Process %d handling range [%d, %d]\n", rank, start, end);

    clock_t start_time = clock();

    int local_clusters, local_smallest_sum;
    find_prime_clusters(start, end, &local_clusters, &local_smallest_sum);

    int total_clusters, smallest_sum_cluster;
    MPI_Reduce(&local_clusters, &total_clusters, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_smallest_sum, &smallest_sum_cluster, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Total prime clusters found: %d\n", total_clusters);
        printf("Smallest-sum cluster: %d\n", smallest_sum_cluster);
        clock_t end_time = clock();
        double time_spent = (double)(end_time - start_time) / CLOCKS_PER_SEC;
        printf("Execution time: %f seconds\n", time_spent);
    }

    MPI_Finalize();
    return 0;
}
