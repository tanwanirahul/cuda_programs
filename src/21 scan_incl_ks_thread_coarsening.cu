#include "timer.h"

# define BLOCK_SIZE 1024
# define COARSE_FACTOR 8
/**
 * Implements the Scan part of the Kogge Scone algorithm. Kogge Scone algorithm 
 * is implemented using two kernels  - scan and add. 
 */
__global__ void scan_kernel(float *input, float * output, float *block_wise_sums, unsigned int N) {

    unsigned int segment_start = (blockIdx.x * blockDim.x * COARSE_FACTOR);
    unsigned int segment_thread = segment_start + threadIdx.x;

    if (segment_thread < N) {
    // Load the BLOCK_SIZE * COARSE_FACTOR elements in the shared memory.
    // The threads are arranged such a way that access to global memory (input) is coalesced.
    __shared__ float input_coarse[COARSE_FACTOR * BLOCK_SIZE];
    for(int c=0; c <COARSE_FACTOR; c++) {
        int i = (segment_start + (c * BLOCK_SIZE) + threadIdx.x);
        if (i < N) {
            input_coarse[(c * BLOCK_SIZE) + threadIdx.x] = input[i];
        }
        else {
            input_coarse[(c * BLOCK_SIZE) + threadIdx.x] = 0.0;
        }
    }
    __syncthreads();

    // For each thread, perform the sequential sum over the COARSE_Factor elements.
    for(int c=1; c <COARSE_FACTOR; c++) {
        unsigned int i = threadIdx.x * COARSE_FACTOR;
        input_coarse[i + c] += input_coarse[ i + c - 1];
    }
    __syncthreads();

    __shared__ float buffer_1[BLOCK_SIZE];
    __shared__ float buffer_2[BLOCK_SIZE];

    float *input_s = buffer_1;
    float * output_s = buffer_2;
    
        input_s[threadIdx.x] = input_coarse[((threadIdx.x + 1) * COARSE_FACTOR)-1];
        __syncthreads();

        for(int step = 1; step <= BLOCK_SIZE/2; step*=2) {
            if(threadIdx.x >= step) {
                output_s[threadIdx.x] = input_s[threadIdx.x] + input_s[threadIdx.x - step];
            }
            else {
                output_s[threadIdx.x] = input_s[threadIdx.x];
            }
            __syncthreads();
            
            float * temp = input_s;
            input_s = output_s;
            output_s = temp;
        }

        // Perform the add kernel operation - adding over partial sums to each element.
        if(threadIdx.x > 0) {
            for(int c = 0; c < COARSE_FACTOR; c++) {
                input_coarse[threadIdx.x * COARSE_FACTOR + c] += input_s[threadIdx.x - 1];
            }
        }   

        if(threadIdx.x == BLOCK_SIZE - 1 || segment_thread == (N-1)) {
            block_wise_sums[blockIdx.x] = input_s[threadIdx.x];
        }

        __syncthreads();
        // Each thread need to update COARSE_FACTOR elements.
        for(int c = 0; c < COARSE_FACTOR; c++) {
            unsigned int i = (segment_start + (c * BLOCK_SIZE) + threadIdx.x);
            output[i] = input_coarse[(c * BLOCK_SIZE) + threadIdx.x];
        }
    }
}

/**
 * Adds total sum upto the prev block to every element in the array
 * except for elements in the first block.
 */
__global__ void add_kernel(float * output, float * block_wise_sums, unsigned int N) {
    unsigned int i = (blockIdx.x * blockDim.x * COARSE_FACTOR) + (threadIdx.x * COARSE_FACTOR);
        if(blockIdx.x > 0) {
            for(int c=0; c<COARSE_FACTOR; c++) {
                if ((i + c) < N) {
                    output[i + c] += block_wise_sums[blockIdx.x - 1];
                }
            }
        }
}

void scan_wrapper_d(float *input_d, float * output_d, unsigned int N) {
    unsigned int threadsPerBlock = BLOCK_SIZE;
    unsigned int numBlocks = (N + (threadsPerBlock * COARSE_FACTOR) - 1) / (threadsPerBlock * COARSE_FACTOR);

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
    unsigned int N = (argc > 1)?(atoi(argv[1])):((1 << 20));
    //unsigned int N = 14;
    
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
