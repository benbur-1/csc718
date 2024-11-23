/*
 * Author: Ben Burgess
 * Date: 2024-10-31
 * Class: CSC 718 - Dakota State University
 * Email: ben.burgess@trojans.dsu.edu
 *
 * Description:
 * This program searches for prime clusters of size three within the range [2, 1000000] using MPI to parallelize the search.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

#define START 2
#define END 1000000

// Function to check if a number is prime
bool is_prime(int n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (int i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
    }
    return true;
}

int main(int argc, char** argv) {
    int rank, size;
    int start_range, end_range;
    int total_prime_clusters = 0;
    clock_t start_time, end_time;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Calculate the sub-range for each process
    int range_size = (END - START + 1) / size;
    start_range = START + rank * range_size;
    end_range = (rank == size - 1) ? END : start_range + range_size - 1;

    if (rank == 0) {
        printf("Searching for prime clusters in range [%d, %d] using %d processes.\n", START, END, size);
    }
    printf("Process %d handling range [%d, %d]\n", rank, start_range, end_range);

    // Start timing
    start_time = clock();

    // Find prime clusters in the assigned range
    int local_prime_clusters = 0;
    int prev_prime = -1, second_prev_prime = -1;
    for (int i = start_range; i <= end_range; i++) {
        if (is_prime(i)) {
            if (prev_prime != -1 && second_prev_prime != -1) {
                if (i - prev_prime == 2 && prev_prime - second_prev_prime == 2) {
                    local_prime_clusters++;
                }
            }
            second_prev_prime = prev_prime;
            prev_prime = i;
        }
    }

    // Reduce the number of prime clusters from all processes
    MPI_Reduce(&local_prime_clusters, &total_prime_clusters, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // End timing
    end_time = clock();
    double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    // Print results
    if (rank == 0) {
        printf("Total prime clusters found: %d\n", total_prime_clusters);
        printf("Execution time: %.2f seconds\n", elapsed_time);
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
