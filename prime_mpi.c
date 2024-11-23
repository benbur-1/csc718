#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define MIN(a,b)  ((a)<(b)?(a):(b))
#define BLOCK_LOW(id,p,n)  ((id)*(n)/(p))
#define BLOCK_HIGH(id,p,n)   (BLOCK_LOW(((id)+1),p,n)-1)
#define BLOCK_SIZE(id,p,n)   ((BLOCK_LOW(((id)+1),p,n))-(BLOCK_LOW(id,p,n)))
#define BLOCK_OWNER(index,p,n)   (((p)*(index)+1)-1)/(n))

// Function to find prime clusters
int find_prime_clusters(int *primes, int prime_count, int **clusters) {
    int count = 0;
    for (int i = 0; i < prime_count - 2; i++) {
        if (primes[i+1] - primes[i] == 2 && primes[i+2] - primes[i+1] == 2) {
            clusters[count] = (int *)malloc(3 * sizeof(int));
            clusters[count][0] = primes[i];
            clusters[count][1] = primes[i+1];
            clusters[count][2] = primes[i+2];
            count++;
        }
    }
    return count;
}

int main(int argc, char *argv[]) {
    int count;          // Local prime count
    double elapsed_time; // Execution time
    int first;          // Index of the first sieve
    int global_count;   // Global count of prime clusters
    int high_value;     // Highest value assigned to this process
    int i;              // Loop counter
    int id;             // This process ID
    int index;          // Index of the current sieve
    int low_value;      // Lowest value assigned to this process
    int *marked;        // Array elements to be marked
    int n;              // Value of the largest number
    int p;              // Number of processes
    int proc0_size;     // Number of elements assigned to process zero
    int prime;          // Current prime or sieve
    int size;           // Elements in marked array

    MPI_Init(&argc, &argv);

    // Start timer
    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time = -MPI_Wtime();
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (argc != 2) {
        if (!id) {
            printf("Command line: %s <m>\n", argv[0]);
        }
        MPI_Finalize();
        exit(1);
    }

    n = atoi(argv[1]);

    // Find how many elements are assigned to this process
    low_value = 2 + BLOCK_LOW(id, p, n - 1);
    high_value = 2 + BLOCK_HIGH(id, p, n - 1);
    size = BLOCK_SIZE(id, p, n - 1);
    proc0_size = (n - 1) / p;

    if ((2 + proc0_size) < (int)sqrt((double)n)) {
        if (!id) printf("Too many processes\n");
        MPI_Finalize();
        exit(1);
    }

    marked = (int *)malloc(size * sizeof(int));
    if (marked == NULL) {
        printf("Cannot allocate enough memory\n");
        MPI_Finalize();
        exit(1);
    }

    for (i = 0; i < size; i++) marked[i] = 0;

    if (!id) index = 0;
    prime = 2;

    // Sieve of Eratosthenes
    do {
        if (prime * prime > low_value)
            first = prime * prime - low_value;
        else {
            if (!(low_value % prime)) first = 0;
            else first = prime - (low_value % prime);
        }
        for (i = first; i < size; i += prime) marked[i] = 1;

        if (!id) {
            while (marked[++index]);
            prime = index + 2;
        }
        MPI_Bcast(&prime, 1, MPI_INT, 0, MPI_COMM_WORLD);
    } while (prime * prime <= n);

    // Collect primes in this process
    int *primes = (int *)malloc(size * sizeof(int));
    int prime_count = 0;
    for (i = 0; i < size; i++) {
        if (!marked[i]) primes[prime_count++] = low_value + i;
    }

    // Detect prime clusters
    int **clusters = (int **)malloc(prime_count * sizeof(int *));
    int local_cluster_count = find_prime_clusters(primes, prime_count, clusters);

    // Share boundary primes for cluster detection
    int edge_primes[2] = {0, 0};
    if (prime_count > 1) {
        edge_primes[0] = primes[prime_count - 2];
        edge_primes[1] = primes[prime_count - 1];
    }
    if (id < p - 1) {
        MPI_Send(edge_primes, 2, MPI_INT, id + 1, 0, MPI_COMM_WORLD);
    }
    if (id > 0) {
        int neighbor_primes[2];
        MPI_Recv(neighbor_primes, 2, MPI_INT, id - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (neighbor_primes[1] - neighbor_primes[0] == 2 && primes[0] - neighbor_primes[1] == 2) {
            clusters[local_cluster_count] = (int *)malloc(3 * sizeof(int));
            clusters[local_cluster_count][0] = neighbor_primes[0];
            clusters[local_cluster_count][1] = neighbor_primes[1];
            clusters[local_cluster_count][2] = primes[0];
            local_cluster_count++;
        }
    }

    // Gather total cluster count
    MPI_Reduce(&local_cluster_count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Stop timer
    elapsed_time += MPI_Wtime();

    if (!id) {
        printf("Total prime clusters found: %d\n", global_count);
        printf("Total elapsed time: %10.6f\n", elapsed_time);
    }

    // Free memory
    for (i = 0; i < local_cluster_count; i++) {
        free(clusters[i]);
    }
    free(clusters);
    free(primes);
    free(marked);

    MPI_Finalize();
    return 0;
}
