%%writefile merge_corank_naive_impln.cu


#include "timer.h"
#include "matrix_utils.h"
#include <stdbool.h>

#define BLOCK_SIZE 1024
#define ELEMENTS_PER_THREAD 16
#define ELEMENTS_PER_BLOCK (BLOCK_SIZE * ELEMENTS_PER_THREAD)

/**
 * Given the sorted array of floats (a and b) of size m and n respectively, 
 * merge them together into C such that entire merged sequence is sorted.
 */
__device__ void sequentialMerge(float *a, float *b, float *c, unsigned int m, unsigned int n) {

    unsigned int i = 0;
    unsigned int j = 0;
    unsigned int k = 0;

    while( i< m && j < n)
    {
        if(a[i] <= b[j]) {
            c[k++] = a[i++];
        } else {
            c[k++] = b[j++];
        }
    }
    while(i < m){
        c[k++] = a[i++];
    }
    while(j < n){
        c[k++] = b[j++];
    }
}


/**
 * Returns the coRank index (i) within an array a corresponding to the element index
 * (K) in the merged sequence.
 * a and b are sorted arrays of size m and n respectively.
 * k is the index in the merged sequence for which we need to find the coRank.
 */
__device__ unsigned int coRank(float *a, float *b, unsigned int m, unsigned int n, unsigned int k) {

    unsigned int iLow = (k > n)?(k-n):0;
    unsigned int iHigh = (k < m)?k:m;


    while(true) {
        unsigned int i = (iLow + iHigh) / 2;
        int j = k - i;
        if( i > 0 && j<n && a[i-1] > b[j]) {
            iHigh = i;
        } else if(i < m && j > 0 && a[i] < b[j-1]) {
            iLow = i;
        } else {
            return i;
        }
    }
    return 0;
}

__global__ void merge_kernel(float *a, float *b, float *c, unsigned int m, unsigned int n) {

    unsigned int k = (blockIdx.x * blockDim.x + threadIdx.x) * ELEMENTS_PER_THREAD;

    if(k < m+n) {
        unsigned int iStart = coRank(a, b, m, n, k);
        unsigned int jStart = k - iStart;
        unsigned int kNext = (k+ELEMENTS_PER_THREAD < (m+n))?k+ELEMENTS_PER_THREAD:(m+n) ;
        unsigned int iEnd = coRank(a, b, m, n, kNext);
        unsigned int jEnd = kNext - iEnd;

        sequentialMerge(&a[iStart], &b[jStart], &c[k], iEnd-iStart, jEnd-jStart);
    }
}


void merge_wrapper(float *a, float *b, float *c, unsigned int m, unsigned int n) {
   
    // Define the thread GRID size for the CUDA kernels.
    unsigned int threadsPerBlock = BLOCK_SIZE;
    unsigned int elementsPerBlock = ELEMENTS_PER_BLOCK;
    unsigned int numBlocks = ( (m+n) + (elementsPerBlock) - 1) / (elementsPerBlock);

    printf("\nNo. of blocks being launched: %d\n", numBlocks);
    // Step 1 - Allocate memory of the GPU device.
    float *a_d, *b_d, *c_d;

    Timer timer;
    timer = initTimer(1);
    startTimer(&timer);
    cudaMalloc((void**) &a_d, m * sizeof(float));
    cudaMalloc((void**) &b_d, n * sizeof(float));
    cudaMalloc((void**) &c_d, (m+n) * sizeof(float));

    

    // Copy data from host to GPU device.
    cudaMemcpy(a_d, a, m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTimer(&timer);
    printElapsedTime(&timer, "Data Transfer from Host to GPU:", CYAN);
    

    // Perform parallel computation.
    timer = initTimer(1);
    startTimer(&timer);
    merge_kernel<<<numBlocks, threadsPerBlock>>>(a_d, b_d, c_d, m, n);
    cudaDeviceSynchronize();
    stopTimer(&timer);
    printElapsedTime(&timer, "GPU Kernel Execution Time:", GREEN);

    // Copy results from GPU device to host memory.
    timer = initTimer(1);
    startTimer(&timer);
    cudaMemcpy(c, c_d, (m+n) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    stopTimer(&timer);
    printElapsedTime(&timer, "Time to Copy Output from GPU to Host:", CYAN);

    // Deallocate memory on GPU device.
    cudaFree((void*)a_d);
    cudaFree((void*)b_d);
    cudaFree((void*)c_d);
    cudaDeviceSynchronize();
}

int main(int argc, char **argv)
{
  
    unsigned int M = (argc > 2)?(atoi(argv[2])):(1 << 22);
    unsigned int N = (argc > 1)?(atoi(argv[1])):(1 << 19);

    // unsigned int M = 10;
    // unsigned int N = 6;
    
    unsigned int delta = 50;
    
    // Allocate memory on the host to hold the sorted output.
    float *output = (float*) malloc( (M+N) * sizeof(float));

    // create two 1D matrix with random data for sorting.
    Matrix1D A = random_sorted_matrix_1D(M, delta);
    Matrix1D B = random_sorted_matrix_1D(N, delta);

    Timer timer;
    //Timer timer_gpu;
    timer = initTimer(1);
    startTimer(&timer);

    //sequentialMerge(A.buffer, B.buffer, output, M, N);
    merge_wrapper(A.buffer, B.buffer, output, M, N);

    stopAndPrintElapsed(&timer, "GPU End to End Executiom Time: ", GREEN);

    Matrix1D C;
    C.length = M+N;
    C.buffer = output;

    printf("\nIs the resulting array sorted? %d\n", is_matrix_sorted_1D(&C));

    for(unsigned int i=0; i<C.length - 1; i++) {
        if(C.buffer[i+1] < C.buffer[i])
            printf("\nMismatch found at index: %d. i: %4.2f    i+1: %4.2f\n", i, C.buffer[i], C.buffer[i+1]);
            break;
    }

    printf("\nInspect first few elements: \n");
    int limit = 10;
    printf("\nA: \n");
    for(int i = 0; i< limit; i++) {
      printf("%4.2f ", A.buffer[i]);
    }
    printf("\n");

    printf("\nB: \n");
    for(int i = 0; i< limit; i++) {
      printf("%4.2f ", B.buffer[i]);
    }
    printf("\n");

    printf("\nC: \n");
    for(int i = 0; i<limit+limit; i++) {
      printf("%4.2f ", C.buffer[i]);
    }
    printf("\n");

    release_matrix_1D(&A);
    release_matrix_1D(&B);
    free(output);

    return 0;
}