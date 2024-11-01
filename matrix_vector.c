/*
 * Author: Ben Burgess
 * Date: 2024-10-31
 * Class: CSC 718 - Dakota State University
 * Email: ben.burgess@trojans.dsu.edu
 *
 * Description:
 * This program performs matrix-vector multiplication with dynamic load balancing using MPI.
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
    int *matrix = NULL;
    int *vector = NULL;
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
        printf("Matrix size: %dx%d\n", rows, cols);
        printf("Number of processes: %d\n", size);
    }

    if (rank == 0) {
        matrix = (int*) malloc(rows * cols * sizeof(int));
        vector = (int*) malloc(cols * sizeof(int));
        global_result = (int*) calloc(rows, sizeof(int));

        if (matrix == NULL || vector == NULL || global_result == NULL) {
            fprintf(stderr, "Memory allocation failed on master process\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        srand(time(NULL));
        for (int i = 0; i < rows * cols; i++) matrix[i] = rand() % 100 + 1;
        for (int i = 0; i < cols; i++) vector[i] = rand() % 100 + 1;
    } else {
        vector = (int*) malloc(cols * sizeof(int));
        if (vector == NULL) {
            fprintf(stderr, "Memory allocation for vector failed on process %d\n", rank);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    int local_rows = BLOCK_SIZE(rank, size, rows);
    int low_row = BLOCK_LOW(rank, size, rows);
    int high_row = BLOCK_HIGH(rank, size, rows);

    // Print "handling" message for the root process
    if (rank == 0) {
        printf("Process %d is handling rows %d to %d (%d rows)\n", rank, low_row, high_row, local_rows);
    }

    // Synchronize to ensure rank 0 prints first
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            int proc_rows = BLOCK_SIZE(i, size, rows);
            printf("Sending %d rows to process %d\n", proc_rows, i);
            MPI_Send(&proc_rows, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Recv(NULL, 0, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        int received_rows;
        MPI_Recv(&received_rows, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d is handling rows %d to %d (%d rows)\n", rank, low_row, high_row, local_rows);
        MPI_Send(NULL, 0, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    int *local_matrix = (int*) malloc(local_rows * cols * sizeof(int));
    local_result = (int*) calloc(local_rows, sizeof(int));

    if (local_matrix == NULL || local_result == NULL) {
        fprintf(stderr, "Memory allocation failed on process %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    MPI_Scatter(matrix, local_rows * cols, MPI_INT, local_matrix, local_rows * cols, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(vector, cols, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < local_rows; i++) {
        local_result[i] = 0;
        for (int j = 0; j < cols; j++) {
            local_result[i] += local_matrix[i * cols + j] * vector[j];
        }
    }

    int *recv_counts = NULL;
    int *displs = NULL;

    if (rank == 0) {
        recv_counts = (int*) malloc(size * sizeof(int));
        displs = (int*) malloc(size * sizeof(int));
        
        int offset = 0;
        for (int i = 0; i < size; i++) {
            recv_counts[i] = BLOCK_SIZE(i, size, rows);
            displs[i] = offset;
            offset += recv_counts[i];
        }
    }

    MPI_Gatherv(local_result, local_rows, MPI_INT, global_result, recv_counts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    elapsed_time += MPI_Wtime();

    if (rank == 0) {
        printf("Resulting vector c:\n");
        for (int i = 0; i < rows; i++) {
            printf("%.6f ", (double)global_result[i]);
        }
        printf("\nTotal elapsed time: %10.6f seconds\n", elapsed_time);
    }

    if (rank == 0) {
        free(matrix);
        free(global_result);
        free(recv_counts);
        free(displs);
    }
    free(vector);
    free(local_matrix);
    free(local_result);

    MPI_Finalize();
    return 0;
}
