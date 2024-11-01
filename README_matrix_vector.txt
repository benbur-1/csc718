README for Matrix-Vector Multiplication Program

Author: Ben Burgess
Date: 2024-10-31
Class: CSC 718 - Dakota State University
Email: ben.burgess@trojans.dsu.edu

Contents:
1. matrix_vector.c - MPI-based Matrix-Vector Multiplication Program with Dynamic Load Balancing

Instructions to Compile:

To compile the program, use the mpicc compiler:

1. Compile matrix_vector.c:
   mpicc -o matrix_vector matrix_vector.c

Steps to Run:

1. Run the program with mpirun:
   mpirun -np <number_of_processes> ./matrix_vector <rows> <cols>

Example:
   mpirun -np 4 ./matrix_vector 100 100
   This runs the program with a 100x100 matrix on 4 processes.

Average Running Time:

Average running time varies based on matrix size and process count:
- For a 100x100 matrix with 4 processes: approximately 0.0002 seconds.
- For a 1000x1000 matrix with 16 processes: approximately 0.01 seconds.

Expected Output:

- Matrix Size and Number of Processes:
  e.g., "Matrix size: 100x100" and "Number of processes: 4"
- Process Assignment:
  e.g., "Sending 25 rows to process 1" and "Process 1 is handling rows 25 to 49 (25 rows)"
- Resulting Vector `c`:
  e.g., "262020.000000 247811.000000 266285.000000 ..."
- Total Elapsed Time:
  e.g., "Total elapsed time: 0.000299 seconds"

Additional Information:

- Ensure mpicc and mpirun are available for MPI compilation and execution.
- Random matrix and vector values are generated each run.
- The program requires at least as many rows as processes for balanced load distribution.
