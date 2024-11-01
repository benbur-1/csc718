
README for Matrix-Matrix Multiplication Program

Author: Ben Burgess
Date: 2024-10-31
Class: CSC 718 - Dakota State University
Email: ben.burgess@trojans.dsu.edu

Contents:
1. matrix_matrix.c - Matrix-Matrix Multiplication Program using MPI with Dynamic Load Balancing

Instructions to Compile:

To compile the program, use the following `mpicc` command:

   mpicc -o matrix_matrix matrix_matrix.c

Steps to Run:

To run the program with different matrix dimensions, specify the matrix row and column count as arguments:

   mpirun -np <number_of_processes> -machinefile <machinefile> ./matrix_matrix <rows> <cols>

Example command:

   mpirun -np 4 ./matrix_matrix 1000 1000

This command will run the program with a 1000x1000 matrix `A` and `B`, multiplying them to produce matrix `C` of size 1000x1000.

Average Running Time:

The average running time depends on the size of the matrices and the number of processes used. For large matrices (e.g., 5000x5000 or 10000x10000) with 4, 8, or 16 processes, expect a time in the range of several seconds, with faster times on high-performance clusters.

Expected Output:

Example for a 1000x1000 Matrix-Matrix Multiplication:

Matrix A size: 1000x1000
Matrix B size: 1000x1000
Matrix C size: 1000x1000
Process 0 is handling rows 0 to 249 (250 rows)
Sending 250 rows to process 1
Process 1 is handling rows 250 to 499 (250 rows)
...

Resulting matrix C:
1000.000000 1200.000000 ...
2000.000000 1400.000000 ...
...
Total elapsed time: 0.123456 seconds

Additional Information:

- Ensure that `rows` and `cols` are the same for both `matrix A` and `matrix B`.
- Run on clusters with `machinefile` configuration to allocate processes across nodes, for example:

      mpirun -np 16 -machinefile machinefile ./matrix_matrix 5000 5000

- The program uses MPI for dynamic load balancing, allowing efficient resource utilization across multiple processes.
- Make sure to free memory on each process to avoid memory leaks.
