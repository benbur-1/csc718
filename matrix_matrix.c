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
    N = K;

    if (rank == 0) {
        printf("Matrix A size: %dx%d\n", M, K);
        printf("Matrix B size: %dx%d\n", K, N);
        printf("Matrix C size: %dx%d\n", M, N);

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

    int rows_per_process = M / size;
    int *local_A = (int*) malloc(rows_per_process * K * sizeof(int));
    int *local_C = (int*) calloc(rows_per_process * N, sizeof(int));

    if (local_A == NULL || local_C == NULL) {
        fprintf(stderr, "Memory allocation failed on rank %d.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    MPI_Bcast(B, K * N, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Scatter(A, rows_per_process * K, MPI_INT, local_A, rows_per_process * K, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < rows_per_process; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                local_C[i * N + j] += local_A[i * K + k] * B[k * N + j];
            }
        }
    }

    MPI_Gather(local_C, rows_per_process * N, MPI_INT, C, rows_per_process * N, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Resulting matrix C:\n");
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                printf("%d ", C[i * N + j]);
            }
            printf("\n");
        }
    }

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
