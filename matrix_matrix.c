#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
    int rank, size, M, K, N;
    int *A = NULL, *B = NULL, *C = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <M> <K>\n", argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    M = atoi(argv[1]);
    K = atoi(argv[2]);
    N = K;  // Assuming square matrices for simplicity.

    if (rank == 0) {
        printf("Matrix A size: %dx%d\n", M, K);
        printf("Matrix B size: %dx%d\n", K, N);
        printf("Matrix C size: %dx%d\n", M, N);

        // Allocate memory for A, B, and C on the root process.
        A = (int*) malloc(M * K * sizeof(int));
        B = (int*) malloc(K * N * sizeof(int));
        C = (int*) calloc(M * N, sizeof(int));  // C initialized to zero.

        if (A == NULL || B == NULL || C == NULL) {
            fprintf(stderr, "Memory allocation failed on root process.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        // Initialize matrices A and B with random values.
        srand(time(NULL));
        for (int i = 0; i < M * K; i++) A[i] = rand() % 10 + 1;
        for (int i = 0; i < K * N; i++) B[i] = rand() % 10 + 1;
    }

    // Allocate space for local computations.
    int rows_per_process = M / size;
    int *local_A = (int*) malloc(rows_per_process * K * sizeof(int));
    int *local_C = (int*) calloc(rows_per_process * N, sizeof(int));

    if (local_A == NULL || local_C == NULL) {
        fprintf(stderr, "Memory allocation failed on rank %d.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Broadcast matrix B from root to all processes.
    if (rank == 0 && B == NULL) {
        fprintf(stderr, "Matrix B not allocated on root.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    MPI_Bcast(B, K * N, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter rows of A to each process.
    MPI_Scatter(A, rows_per_process * K, MPI_INT, local_A, rows_per_process * K, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process calculates its portion of the matrix C.
    for (int i = 0; i < rows_per_process; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                local_C[i * N + j] += local_A[i * K + k] * B[k * N + j];
            }
        }
    }

    // Gather results from all processes into matrix C on the root process.
    MPI_Gather(local_C, rows_per_process * N, MPI_INT, C, rows_per_process * N, MPI_INT, 0, MPI_COMM_WORLD);

    // Root process prints the result matrix C.
    if (rank == 0) {
        printf("Resulting matrix C:\n");
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                printf("%d ", C[i * N + j]);
            }
            printf("\n");
        }
    }

    // Free allocated memory.
    if (rank == 0) {
        free(A);
        free(B);
        free(C);
    }
    free(local_A);
    free(local_C);

    MPI_Finalize();
    return 0;
}
