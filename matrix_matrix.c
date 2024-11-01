// Filename: matrix_matrix.c
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_SIZE 16  // Define block size for memory efficiency

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

    M = atoi(argv[1]); // Rows in A and C
    K = atoi(argv[2]); // Columns in A and Rows in B
    N = K;             // Columns in B and C for simplicity

    if (rank == 0) {
        printf("Matrix A size: %dx%d\n", M, K);
        printf("Matrix B size: %dx%d\n", K, N);
        printf("Matrix C size: %dx%d\n", M, N);

        // Allocate and initialize matrices on the root process
        A = (int*) malloc(M * K * sizeof(int));
        B = (int*) malloc(K * N * sizeof(int));
        C = (int*) calloc(M * N, sizeof(int));

        if (A == NULL || B == NULL || C == NULL) {
            fprintf(stderr, "Memory allocation failed.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        srand(time(NULL));
        for (int i = 0; i < M * K; i++) A[i] = rand() % 10 + 1;
        for (int i = 0; i < K * N; i++) B[i] = rand() % 10 + 1;
    }

    // Local matrices for each process
    int rows_per_process = M / size;
    int *local_A = (int*) malloc(rows_per_process * K * sizeof(int));
    int *local_C = (int*) calloc(rows_per_process * N, sizeof(int));

    if (local_A == NULL || local_C == NULL) {
        fprintf(stderr, "Memory allocation failed on rank %d.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Broadcast matrix B to all processes
    MPI_Bcast(B, K * N, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter rows of matrix A among processes
    MPI_Scatter(A, rows_per_process * K, MPI_INT, local_A, rows_per_process * K, MPI_INT, 0, MPI_COMM_WORLD);

    // Local computation of matrix multiplication for assigned rows
    for (int i = 0; i < rows_per_process; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                local_C[i * N + j] += local_A[i * K + k] * B[k * N + j];
            }
        }
    }

    // Gather the computed parts of matrix C from all processes
    MPI_Gather(local_C, rows_per_process * N, MPI_INT, C, rows_per_process * N, MPI_INT, 0, MPI_COMM_WORLD);

    // Root process prints the resulting matrix C
    if (rank == 0) {
        printf("Resulting matrix C:\n");
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                printf("%d ", C[i * N + j]);
            }
            printf("\n");
        }
    }

    // Free allocated memory
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
