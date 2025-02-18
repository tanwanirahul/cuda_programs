#include "timer.h"

# define BLOCK_SIZE 1024
/**
 * Each block will have BLOCK_SIZE threads which will calculate the partial sums
 * for 2*BLOCK_SIZE elements of the input array.
 */
__global__ void reduce_kernel(float *input, float *block_wise_sums, unsigned int N) {

    unsigned int seg_start = (blockIdx.x * blockDim.x * 2);
    unsigned int seg_thread = seg_start + (threadIdx.x * 2);

    if (seg_thread < N - 1) {
        for (int step = 1; step <= blockDim.x; step*=2 ) {
            if( threadIdx.x % step == 0) {
                input[seg_thread] += input[seg_thread + step];
            }
            __syncthreads();
        }
        if(threadIdx.x == 0) {
            block_wise_sums[blockIdx.x] = input[seg_start];
        }
    }
}

void reduce_wrapper(float *input, float *result, unsigned int N)
{
    // Step 1 - Allocate memory of the GPU device.
    float *input_d, *blockWiseSums_d, *blockWiseSums, sum = 0.0;

    unsigned int threadsPerBlock = BLOCK_SIZE;
    unsigned int numBlocks = (N + (2 * threadsPerBlock) - 1) / (threadsPerBlock * 2);

    Timer timer;
    timer = initTimer(1);
    startTimer(&timer);
    cudaMalloc((void**) &input_d, N * sizeof(float) );
    cudaMalloc((void**) &blockWiseSums_d, numBlocks * sizeof(float) );

    // Copy data from host to GPU device.
    cudaMemcpy(input_d, input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTimer(&timer);
    printElapsedTime(&timer, "Data Transfer from Host to GPU:", CYAN);
    
    timer = initTimer(1);
    startTimer(&timer);

    // Perform parallel computation.
    reduce_kernel<<<numBlocks, threadsPerBlock>>>(input_d, blockWiseSums_d, N);
    cudaDeviceSynchronize();
    stopTimer(&timer);
    printElapsedTime(&timer, "GPU Kernel Execution Time:", GREEN);


    // Perform the total sum from the BlockWiseSums on the CPU.
    blockWiseSums = (float *) malloc(numBlocks * sizeof(float));
    
    // Copy results from GPU device to host memory.
    cudaMemcpy(blockWiseSums, blockWiseSums_d, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for(unsigned int i = 0; i < numBlocks; i++) {
        sum+= blockWiseSums[i];
    }
    *result = sum;
    // Deallocate memory on GPU device.
    cudaFree((void*)input_d);
    cudaFree((void*)blockWiseSums_d);

}

int main(int argc, char **argv)
{
    cudaDeviceSynchronize();

    unsigned int N = (argc > 1)?(atoi(argv[1])):(1 << 25);
    //unsigned int N = (argc > 1)?(atoi(argv[1])):(1 << 10);

    // Allocate memory on the host.
    float *input = (float*) malloc(N * sizeof(float));
    float sum;

    // create a random data for addition.
    for (unsigned int i = 0; i < N; i++) {
        //input[i] = rand();
        input[i] = 1.0;
    }

    Timer timer;
    //Timer timer_gpu;
    timer = initTimer(1);
    startTimer(&timer);
    reduce_wrapper(input, &sum, N);
    stopAndPrintElapsed(&timer, "GPU End to End Executiom Time: ", GREEN);

    printf("Sum returned: %20.2f\n", sum);
    printf("Does output match expected value? %d\n", (sum==N)?1:0);

    free(input);

    return 0;
}