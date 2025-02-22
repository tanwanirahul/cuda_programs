
#include "timer.h"

# define BLOCK_SIZE 1024
/**
 * Implements the Scan part of the Kogge Scone algorithm. Kogge Scone algorithm 
 * is implemented using two kernels  - scan and add. 
 */
__global__ void scan_kernel(float *input, float * output, float *block_wise_sums, unsigned int N) {

    unsigned int seg_start = blockIdx.x * blockDim.x * 2;

    __shared__ float input_s[BLOCK_SIZE * 2];

    // Load input data into the shared memory.
    if(seg_start + threadIdx.x < N) {
        input_s[threadIdx.x] = input[seg_start + threadIdx.x];
    }
    else {
        input_s[threadIdx.x] = 0.0;
    }
    if(seg_start + threadIdx.x + BLOCK_SIZE < N) {
        input_s[threadIdx.x + BLOCK_SIZE] = input[seg_start + threadIdx.x + BLOCK_SIZE];
    }
    else {
        input_s[threadIdx.x + BLOCK_SIZE] = 0.0;
    }
    __syncthreads();


    // Reduction Step.
    for(int step =1; step<=BLOCK_SIZE; step*=2 ) {
        unsigned int i = (threadIdx.x + 1) * 2 * step -1;
        if(i < 2 * BLOCK_SIZE) {
            input_s[i] += input_s[i - step];
        }
        __syncthreads();
    }

    // Post Reduction Step.
    for(int step = BLOCK_SIZE/2; step>=1; step/=2) {
        unsigned int i = (threadIdx.x + 1) * 2 * step -1;
        if(i + step < 2 * BLOCK_SIZE) {
            input_s[i + step] += input_s[i];
        }
        __syncthreads();
    }

    // Store the block-wise partial sums.
    if(threadIdx.x == 0) {
        block_wise_sums[blockIdx.x] = input_s[2 * BLOCK_SIZE - 1];
    }

    // Update the sums in the output array in Global memory from the shared memory.
    // Load input data into the shared memory.
    if(seg_start + threadIdx.x < N) {
        output[seg_start + threadIdx.x] = input_s[threadIdx.x];
    }
    if(seg_start + threadIdx.x + BLOCK_SIZE < N) {
        output[seg_start + threadIdx.x + BLOCK_SIZE] = input_s[threadIdx.x + BLOCK_SIZE];
    }

}

/**
 * Adds total sum upto the prev block to every element in the array
 * except for elements in the first block.
 */
__global__ void add_kernel(float * output, float * block_wise_sums, unsigned int N) {
    unsigned int i = (blockIdx.x * blockDim.x * 2) + threadIdx.x;
    // Updathe output with partial sums.
    if(blockIdx.x > 0) {
        if (i < N)
            output[i] += block_wise_sums[blockIdx.x - 1];
        if (i + BLOCK_SIZE < N)
            output[i + BLOCK_SIZE] += block_wise_sums[blockIdx.x - 1];

    }
}

void scan_wrapper_d(float *input_d, float * output_d, unsigned int N) {
    unsigned int threadsPerBlock = BLOCK_SIZE;
    unsigned int numBlocks = (N + (2 * threadsPerBlock) - 1) / (2 * threadsPerBlock);

    float * blockWiseSums_d;
    cudaMalloc((void**) &blockWiseSums_d, numBlocks * sizeof(float) );
    cudaDeviceSynchronize();

    scan_kernel<<<numBlocks, threadsPerBlock>>>(input_d, output_d, blockWiseSums_d, N);
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
        input[i] = 1.0;
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
    printf("Exepcted answer: %20.2f\n",N * 1.0 ); 
    printf("Does output match expected value? %d\n", (sum==(N * 1.0))?1:0);

    free(input);
    free(output);

    return 0;
}
