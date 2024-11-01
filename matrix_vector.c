// Filename: matrix_vector_dynamic.c
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define REQUEST_ROW_TAG 1
#define ROW_ASSIGN_TAG 2
#define TERMINATE_TAG 3

int main(int argc, char *argv[]) {
    int rank, size, rows, cols;
    double elapsed_time;
    int *matrix = NULL;
    int *vector = NULL;
    int *global_result = NULL;

    MPI_Init(&argc, &argv);
    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time = -MPI_Wtime();

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <rows> <cols>\n", argv[0]);
        }
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    rows = atoi(argv[1]);
    cols = atoi(argv[2]);

    if (rank == 0) {
        printf("Matrix size: %dx%d\n", rows, cols);
        printf("Number of processes: %d\n", size);
        fflush(stdout);
    }

    // Master process initializes matrix and vector
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
        // Allocate vector on workers
        vector = (int*) malloc(cols * sizeof(int));
        if (vector == NULL) {
            fprintf(stderr, "Memory allocation for vector failed on process %d\n", rank);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    MPI_Bcast(vector, cols, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Master process: dynamic row distribution
        int current_row = 0;
        int row_chunk = 10; // Number of rows to assign at a time
        int completed_processes = 0;

        while (completed_processes < size - 1) {
            MPI_Status status;
            int request;

            // Wait for a request from any worker
            MPI_Recv(&request, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST_ROW_TAG, MPI_COMM_WORLD, &status);
            int worker_rank = status.MPI_SOURCE;

            if (current_row >= rows) {
                // Send termination signal if no rows are left
                MPI_Send(NULL, 0, MPI_INT, worker_rank, TERMINATE_TAG, MPI_COMM_WORLD);
                completed_processes++;
            } else {
                // Assign a chunk of rows to the worker
                int rows_to_assign = (current_row + row_chunk > rows) ? rows - current_row : row_chunk;
                MPI_Send(&rows_to_assign, 1, MPI_INT, worker_rank, ROW_ASSIGN_TAG, MPI_COMM_WORLD);
                MPI_Send(&matrix[current_row * cols], rows_to_assign * cols, MPI_INT, worker_rank, ROW_ASSIGN_TAG, MPI_COMM_WORLD);
                MPI_Send(&current_row, 1, MPI_INT, worker_rank, ROW_ASSIGN_TAG, MPI_COMM_WORLD);
                current_row += rows_to_assign;
            }
        }

        // Receive partial results from workers
        for (int i = 1; i < size; i++) {
            MPI_Status status;
            int start_row;
            int rows_to_receive;

            // Receive the start row index and the count of rows
            MPI_Recv(&start_row, 1, MPI_INT, i, ROW_ASSIGN_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows_to_receive, 1, MPI_INT, i, ROW_ASSIGN_TAG, MPI_COMM_WORLD, &status);

            // Receive the actual computed results
            MPI_Recv(&global_result[start_row], rows_to_receive, MPI_INT, i, ROW_ASSIGN_TAG, MPI_COMM_WORLD, &status);
        }

        elapsed_time += MPI_Wtime();

        printf("Resulting vector c:\n");
        for (int i = 0; i < rows; i++) {
            printf("%.6f ", (double)global_result[i]);
        }
        printf("\nTotal elapsed time: %10.6f seconds\n", elapsed_time);
        fflush(stdout);
    } else {
        // Worker processes
        while (1) {
            int request = 1;
            MPI_Send(&request, 1, MPI_INT, 0, REQUEST_ROW_TAG, MPI_COMM_WORLD);

            MPI_Status status;
            int rows_to_process;

            // Check for termination signal or row assignment
            MPI_Recv(&rows_to_process, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == TERMINATE_TAG) break;

            int start_row;
            MPI_Recv(&start_row, 1, MPI_INT, 0, ROW_ASSIGN_TAG, MPI_COMM_WORLD, &status);

            int *matrix_chunk = (int*) malloc(rows_to_process * cols * sizeof(int));
            int *result_chunk = (int*) malloc(rows_to_process * sizeof(int));

            if (matrix_chunk == NULL || result_chunk == NULL) {
                fprintf(stderr, "Memory allocation failed on worker process %d\n", rank);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }

            MPI_Recv(matrix_chunk, rows_to_process * cols, MPI_INT, 0, ROW_ASSIGN_TAG, MPI_COMM_WORLD, &status);

            // Perform matrix-vector multiplication on assigned rows
            for (int i = 0; i < rows_to_process; i++) {
                result_chunk[i] = 0;
                for (int j = 0; j < cols; j++) {
                    result_chunk[i] += matrix_chunk[i * cols + j] * vector[j];
                }
            }

            // Send partial result back to the master
            MPI_Send(&start_row, 1, MPI_INT, 0, ROW_ASSIGN_TAG, MPI_COMM_WORLD);
            MPI_Send(&rows_to_process, 1, MPI_INT, 0, ROW_ASSIGN_TAG, MPI_COMM_WORLD);
            MPI_Send(result_chunk, rows_to_process, MPI_INT, 0, ROW_ASSIGN_TAG, MPI_COMM_WORLD);

            free(matrix_chunk);
            free(result_chunk);
        }
    }

    if (rank == 0) {
        free(matrix);
        free(global_result);
    }
    free(vector);

    MPI_Finalize();
    return 0;
}
