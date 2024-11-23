#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define ARRAY_SIZE 1000000

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

// Sequential merge sort
void merge_sort(int arr[], int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        merge_sort(arr, left, mid);
        merge_sort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

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

int main() {
    int *arr = (int *)malloc(ARRAY_SIZE * sizeof(int));
    int *arr_copy = (int *)malloc(ARRAY_SIZE * sizeof(int));

    // Initialize the array with random values
    for (int i = 0; i < ARRAY_SIZE; i++) {
        arr[i] = rand() % 10000;
        arr_copy[i] = arr[i];
    }

    // Sequential merge sort
    double start_time = omp_get_wtime();
    merge_sort(arr, 0, ARRAY_SIZE - 1);
    double end_time = omp_get_wtime();
    printf("| Threads | Sequential Time (s) | Parallel Time (s) | Speedup |
");
    printf("|---------|---------------------|-------------------|---------|
");
    printf("| 1       | %f              | N/A               | N/A     |
", end_time - start_time);

    // Run parallel merge sort with different thread counts
    for (int threads = 1; threads <= 8; threads *= 2) {
        omp_set_num_threads(threads);
        for (int i = 0; i < ARRAY_SIZE; i++) {
            arr_copy[i] = arr[i];
        }
        start_time = omp_get_wtime();
        parallel_merge_sort(arr_copy, 0, ARRAY_SIZE - 1, 4); // Adjust depth as needed
        end_time = omp_get_wtime();
        double parallel_time = end_time - start_time;
        double speedup = (threads == 1) ? 1.0 : (end_time - start_time) / (end_time - start_time);
        printf("| %d       | %f              | %f             | %f    |
", threads, (threads == 1)
