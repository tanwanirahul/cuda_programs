#include "timer.h"

# define BLOCK_SIZE 1024
/**
 * Implements the Scan part of the Kogge Scone algorithm. Kogge Scone algorithm 
 * is implemented using two kernels  - scan and add. 
 */
__global__ void scan_kernel(float *input, float * output, float *block_wise_sums, unsigned int N) {

    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i < N) {
        output[i] = input[i];
        __syncthreads();

        for(int step = 1; step <= BLOCK_SIZE/2; step*=2) {
            float sum = 0.0;
            if(threadIdx.x >= step) {
                sum = output[i] + output[i - step];
            }
            __syncthreads();
            if(threadIdx.x >= step) {
                output[i] = sum;
            }
            __syncthreads();
        }
        if(threadIdx.x == BLOCK_SIZE - 1 || i == (N-1)) {
            block_wise_sums[blockIdx.x] = output[i];
        }
    }
}

/**
 * Adds total sum upto the prev block to every element in the array
 * except for elements in the first block.
 */
__global__ void add_kernel(float * output, float * block_wise_sums, unsigned int N) {
    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < N) {
        if(blockIdx.x > 0) {
            output[i] += block_wise_sums[blockIdx.x - 1];
        }
    }
}

void scan_wrapper_d(float *input_d, float * output_d, unsigned int N) {
    unsigned int threadsPerBlock = BLOCK_SIZE;
    unsigned int numBlocks = (N + (threadsPerBlock) - 1) / (threadsPerBlock);

    float * blockWiseSums_d;
    cudaMalloc((void**) &blockWiseSums_d, numBlocks * sizeof(float) );
    cudaDeviceSynchronize();

    scan_kernel<<<numBlocks, threadsPerBlock>>>(input_d, output_d, blockWiseSums_d, N);
    cudaDeviceSynchronize();

    float * blockWiseSums_CPU = (float *) malloc(sizeof(float) * numBlocks);
    cudaMemcpy(blockWiseSums_CPU, blockWiseSums_d, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    // Perform scan on the block wise sum and then add the sums to output elements.
    if (numBlocks > 1) {
        scan_wrapper_d(blockWiseSums_d, blockWiseSums_d, numBlocks);
        add_kernel<<<numBlocks, threadsPerBlock>>>(output_d, blockWiseSums_d, N);
    }
    cudaDeviceSynchronize();

    cudaFree((void*)blockWiseSums_d);
    cudaDeviceSynchronize();
}

void scan_wrapper(float *input, float *output, unsigned int N)
{
    // Step 1 - Allocate memory of the GPU device.
    float *input_d, *output_d;

    Timer timer;
    timer = initTimer(1);
    startTimer(&timer);
    cudaMalloc((void**) &input_d, N * sizeof(float));
    cudaMalloc((void**) &output_d, N * sizeof(float));

    // Copy data from host to GPU device.
    cudaMemcpy(input_d, input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTimer(&timer);
    printElapsedTime(&timer, "Data Transfer from Host to GPU:", CYAN);
    
    
    // Perform parallel computation.
    timer = initTimer(1);
    startTimer(&timer);
    scan_wrapper_d(input_d, output_d, N);
    cudaDeviceSynchronize();
    stopTimer(&timer);
    printElapsedTime(&timer, "GPU Kernel Execution Time:", GREEN);

    // Copy results from GPU device to host memory.
    timer = initTimer(1);
    startTimer(&timer);
    cudaMemcpy(output, output_d, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTimer(&timer);
    printElapsedTime(&timer, "Time to Copy Output from GPU to Host:", CYAN);

    // Deallocate memory on GPU device.
    cudaFree((void*)input_d);
    cudaFree((void*)output_d);
    cudaDeviceSynchronize();

}

int main(int argc, char **argv)
{
    unsigned int N = (argc > 1)?(atoi(argv[1])):(1 << 30);
    
    // Allocate memory on the host.
    float *input = (float*) malloc(N * sizeof(float));
    float *output = (float*) malloc(N * sizeof(float));

    // create a random data for addition.
    for (unsigned int i = 0; i < N; i++) {
        //input[i] = rand();
        input[i] = i;
    }

    Timer timer;
    //Timer timer_gpu;
    timer = initTimer(1);
    startTimer(&timer);
    scan_wrapper(input, output, N);
    stopAndPrintElapsed(&timer, "GPU End to End Executiom Time: ", GREEN);

    float sum = output[N-1];
    printf("No. of elements: %u\n", N);
    printf("Sum returned: %20.2f\n", sum);

    float expected = N-1;
    expected = (expected * (expected + 1))/2.0;
    printf("Exepcted answer: %20.2f\n",(expected)); 
    printf("Does output match expected value? %d\n", (sum==expected)?1:0);

    free(input);
    free(output);

    return 0;
}