/*
 * Author: Ben Burgess
 * Date: 2024-11-23
 * Class: CSC 718 - Dakota State University
 * Email: ben.burgess@trojans.dsu.edu
 *
 * Description:
 * This program implements both a sequential and a parallel version of the merge sort algorithm using OpenMP.
 * The sequential version is used as a baseline, while the parallel version demonstrates performance improvements
 * with different thread counts. The program sorts an array of at least 1,000,000 integers and profiles the performance
 * of both versions to analyze the effects of parallelization.
 */

// File: merge_sort_openmp.c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define ARRAY_SIZE 1000000
#define TASK_SIZE 100  // Depth control for parallelization

// Function to merge two halves of an array
void merge(int arr[], int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;
    int *L = (int *)malloc(n1 * sizeof(int));
    int *R = (int *)malloc(n2 * sizeof(int));

    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (int i = 0; i < n2; i++)
        R[i] = arr[mid + 1 + i];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k++] = L[i++];
        } else {
            arr[k++] = R[j++];
        }
    }
    while (i < n1) {
        arr[k++] = L[i++];
    }
    while (j < n2) {
        arr[k++] = R[j++];
    }

    free(L);
    free(R);
}

// Function prototype for the sequential merge sort
void merge_sort(int arr[], int left, int right);

// Parallel merge sort using OpenMP
void parallel_merge_sort(int arr[], int left, int right, int depth) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        if (depth <= 0) {
            merge_sort(arr, left, right);
        } else {
            #pragma omp parallel sections
            {
                #pragma omp section
                parallel_merge_sort(arr, left, mid, depth - 1);
                #pragma omp section
                parallel_merge_sort(arr, mid + 1, right, depth - 1);
            }
            merge(arr, left, mid, right);
        }
    }
}

// Sequential merge sort
void merge_sort(int arr[], int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        merge_sort(arr, left, mid);
        merge_sort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

int main(int argc, char *argv[]) {
    int *arr = (int *)malloc(ARRAY_SIZE * sizeof(int));
    int *arr_copy = (int *)malloc(ARRAY_SIZE * sizeof(int));

    // Initialize the array with random values
    for (int i = 0; i < ARRAY_SIZE; i++) {
        arr[i] = rand() % 10000;
        arr_copy[i] = arr[i];
    }

    // Sequential merge sort
    double start_time, end_time, sequential_time;
    start_time = omp_get_wtime();
    merge_sort(arr, 0, ARRAY_SIZE - 1);
    end_time = omp_get_wtime();
    sequential_time = end_time - start_time;
    printf("Sequential Merge Sort Time: %f seconds\n", sequential_time);

    // Run parallel merge sort with different thread counts and collect performance data
    double parallel_times[4];
    int thread_counts[4] = {1, 2, 4, 8};

    for (int i = 0; i < 4; i++) {
        int threads = thread_counts[i];
        omp_set_num_threads(threads);
        for (int j = 0; j < ARRAY_SIZE; j++) {
            arr_copy[j] = arr[j];
        }
        start_time = omp_get_wtime();
        #pragma omp parallel
        {
            #pragma omp single
            parallel_merge_sort(arr_copy, 0, ARRAY_SIZE - 1, 4); // Adjust depth as needed
        }
        end_time = omp_get_wtime();
        parallel_times[i] = end_time - start_time;
        printf("Threads: %d | Parallel Merge Sort Time: %f seconds\n", threads, parallel_times[i]);
    }

    // Summary of performance
    printf("\nPerformance Summary:\n");
    printf("Threads | Sequential Time (s) | Parallel Time (s) | Speedup\n");
    printf("--------|---------------------|-------------------|---------\n");
    for (int i = 0; i < 4; i++) {
        double speedup = sequential_time / parallel_times[i];
        printf("%7d | %19.6f | %17.6f | %7.2f\n", thread_counts[i], sequential_time, parallel_times[i], speedup);
    }

    free(arr);
    free(arr_copy);
    return 0;
}
