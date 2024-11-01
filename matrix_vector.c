// Filename: matrix_vector_multiplication_v6.c
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
        fflush(stdout);
        printf("Number of processes: %d\n", size);
        fflush(stdout);
    }

    if (rank == 0) {
        matrix = (int*) malloc(rows * cols * sizeof(int));
        vector = (int*) malloc(cols * sizeof(int));
        global_result = (int*) calloc(rows, sizeof(int));

        if (matrix == NULL || vector == NULL || global_result == NULL) {
            fprintf(stderr, "Memory allocation failed on master process\n");
            fflush(stderr);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        srand(time(NULL));
        for (int i = 0; i < rows * cols; i++) matrix[i] = rand() % 100 + 1;
        for (int i = 0; i < cols; i++) vector[i] = rand() % 100 + 1;
    } else {
        vector = (int*) malloc(cols * sizeof(int));
        if (vector == NULL) {
            fprintf(stderr, "Memory allocation for vector failed on process %d\n", rank);
            fflush(stderr);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    int local_rows = BLOCK_SIZE(rank, size, rows);
    int low_row = BLOCK_LOW(rank, size, rows);
    int high_row = BLOCK_HIGH(rank, size, rows);

    int *local_matrix = (int*) malloc(local_rows * cols * sizeof(int));
    local_result = (int*) calloc(local_rows, sizeof(int));

    if (local_matrix == NULL || local_result == NULL) {
        fprintf(stderr, "Memory allocation failed on process %d\n", rank);
        fflush(stderr);
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

    // Ensure ordered printing of each process's assigned row range
    for (int i = 0; i < size; i++) {
        if (rank == i) {
            printf("Process %d is handling rows %d to %d (%d rows)\n", rank, low_row, high_row, local_rows);
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);  // Synchronize before next process prints
    }

    if (rank == 0) {
        printf("Resulting vector c:\n");
        fflush(stdout);
        for (int i = 0; i < rows; i++) {
            printf("%.6f ", (double)global_result[i]);
            fflush(stdout);
        }
        printf("\nTotal elapsed time: %10.6f seconds\n", elapsed_time);
        fflush(stdout);
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
