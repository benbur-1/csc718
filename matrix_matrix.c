/*
 * Author: Ben Burgess
 * Date: 2024-10-31
 * Class: CSC 718 - Dakota State University
 * Email: ben.burgess@trojans.dsu.edu
 *
 * Description:
 * This program performs matrix-matrix multiplication using MPI with dynamic load balancing.
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_LOW(id, p, n) ((id)*(n)/(p))
#define BLOCK_HIGH(id, p, n) (BLOCK_LOW((id)+1, p, n)-1)
#define BLOCK_SIZE(id, p, n) (BLOCK_HIGH(id, p, n) - BLOCK_LOW(id, p, n) + 1)

int main(int argc, char *argv[]) {
    int rank, size, rows, cols;
    double elapsed_time;
    int *A = NULL;
    int *B = NULL;
    int *local_result = NULL;
    int *global_result = NULL;

    MPI_Init(&argc, &argv);
    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time = -MPI_Wtime();

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <rows> <cols>\n", argv[0]);
            fflush(stderr);
        }
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    rows = atoi(argv[1]);
    cols = atoi(argv[2]);

    if (rank == 0) {
        printf("Matrix A size: %dx%d\n", rows, cols);
        printf("Matrix B size: %dx%d\n", cols, rows);
        printf("Matrix C size: %dx%d\n", rows, rows);
        fflush(stdout);
    }

    if (rank == 0) {
        A = (int*) malloc(rows * cols * sizeof(int));
        B = (int*) malloc(cols * rows * sizeof(int));
        global_result = (int*) calloc(rows * rows, sizeof(int));

        if (A == NULL || B == NULL || global_result == NULL) {
            fprintf(stderr, "Memory allocation failed on master process\n");
            fflush(stderr);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        srand(time(NULL));
        for (int i = 0; i < rows * cols; i++) A[i] = rand() % 10 + 1;
        for (int i = 0; i < cols * rows; i++) B[i] = rand() % 10 + 1;
    } else {
        B = (int*) malloc(cols * rows * sizeof(int));
        if (B == NULL) {
            fprintf(stderr, "Memory allocation for B failed on process %d\n", rank);
            fflush(stderr);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    int local_rows = BLOCK_SIZE(rank, size, rows);
    int low_row = BLOCK_LOW(rank, size, rows);
    int high_row = BLOCK_HIGH(rank, size, rows);

    if (rank == 0) {
        printf("Process %d is handling rows %d to %d (%d rows)\n", rank, low_row, high_row, local_rows);
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            int proc_rows = BLOCK_SIZE(i, size, rows);
            printf("Sending %d rows to process %d\n", proc_rows, i);
            fflush(stdout);
            MPI_Send(&proc_rows, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Recv(NULL, 0, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        int received_rows;
        MPI_Recv(&received_rows, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d is handling rows %d to %d (%d rows)\n", rank, low_row, high_row, local_rows);
        fflush(stdout);
        MPI_Send(NULL, 0, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    int *local_A = (int*) malloc(local_rows * cols * sizeof(int));
    local_result = (int*) calloc(local_rows * rows, sizeof(int));

    if (local_A == NULL || local_result == NULL) {
        fprintf(stderr, "Memory allocation failed on process %d\n", rank);
        fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    MPI_Scatter(A, local_rows * cols, MPI_INT, local_A, local_rows * cols, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, cols * rows, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < rows; j++) {
            local_result[i * rows + j] = 0;
            for (int k = 0; k < cols; k++) {
                local_result[i * rows + j] += local_A[i * cols + k] * B[k * rows + j];
            }
        }
    }

    int *recv_counts = NULL;
    int *displs = NULL;

    if (rank == 0) {
        recv_counts = (int*) malloc(size * sizeof(int));
        displs = (int*) malloc(size * sizeof(int));
        
        int offset = 0;
        for (int i = 0; i < size; i++) {
            recv_counts[i] = BLOCK_SIZE(i, size, rows) * rows;
            displs[i] = offset;
            offset += recv_counts[i];
        }
    }

    MPI_Gatherv(local_result, local_rows * rows, MPI_INT, global_result, recv_counts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    elapsed_time += MPI_Wtime();

    if (rank == 0) {
        printf("Resulting matrix C:\n");
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < rows; j++) {
                printf("%.6f ", (double)global_result[i * rows + j]);
            }
            printf("\n");
        }
        printf("\nTotal elapsed time: %10.6f seconds\n", elapsed_time);
        fflush(stdout);
    }

    if (rank == 0) {
        free(A);
        free(global_result);
        free(recv_counts);
        free(displs);
    }
    free(B);
    free(local_A);
    free(local_result);

    MPI_Finalize();
    return 0;
}
