// Filename: matrix_matrix.c
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_SIZE 16

int main(int argc, char *argv[]) {
    int rank, size, M, K;
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
    int N = K;

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
        for (int i = 0; i < M * K; i++) A[i] = rand() % 100 + 1;
        for (int i = 0; i < K * N; i++) B[i] = rand() % 100 + 1;
    }

    int *local_A = (int*) malloc(BLOCK_SIZE * K * sizeof(int));
    int *local_C = (int*) calloc(BLOCK_SIZE * N, sizeof(int));

    if (local_A == NULL || local_C == NULL) {
        fprintf(stderr, "Memory allocation failed for local matrices.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    for (int i = 0; i < K; i += BLOCK_SIZE) {
        int block_size = (i + BLOCK_SIZE <= K) ? BLOCK_SIZE : (K - i);
        if (B != NULL) {
            MPI_Bcast(&B[i * N], block_size * N, MPI_INT, 0, MPI_COMM_WORLD);
        }
    }

    int rows_assigned = 0;
    while (rows_assigned < M) {
        int rows_to_assign = (rows_assigned + BLOCK_SIZE <= M) ? BLOCK_SIZE : (M - rows_assigned);

        if (rank == 0) {
            for (int p = 1; p < size; p++) {
                if (rows_assigned >= M) break;

                MPI_Send(&rows_to_assign, 1, MPI_INT, p, 0, MPI_COMM_WORLD);
                MPI_Send(&A[rows_assigned * K], rows_to_assign * K, MPI_INT, p, 0, MPI_COMM_WORLD);

                rows_assigned += rows_to_assign;
            }

            for (int i = 0; i < rows_to_assign; i++) {
                for (int j = 0; j < N; j++) {
                    for (int k = 0; k < K; k++) {
                        C[(rows_assigned + i) * N + j] += A[(rows_assigned + i) * K + k] * B[k * N + j];
                    }
                }
            }
        } else {
            MPI_Recv(&rows_to_assign, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(local_A, rows_to_assign * K, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int i = 0; i < rows_to_assign; i++) {
                for (int j = 0; j < N; j++) {
                    for (int k = 0; k < K; k++) {
                        local_C[i * N + j] += local_A[i * K + k] * B[k * N + j];
                    }
                }
            }

            MPI_Send(local_C, rows_to_assign * N, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }

    if (rank == 0) {
        for (int p = 1; p < size; p++) {
            MPI_Recv(&C[rows_assigned * N], rows_assigned * N, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

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
