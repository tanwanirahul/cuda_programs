#include "timer.h"

__global__ void vec_add_kernel(float *x_d, float *y_d, float *result_d, unsigned int N) {

    unsigned int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (thread_id < N) {
        result_d[thread_id] = x_d[thread_id] + y_d[thread_id];
    }
}

void vec_add_GPU(float *x, float *y, float *result, unsigned int N)
{
    // Step 1 - Allocate memory of the GPU device.
    // Q - How do we specify the device where we want this memory to be allocated?
    float *x_d, *y_d, *result_d;
    cudaMalloc((void**) &x_d, N * sizeof(float) );
    cudaMalloc((void**) &y_d, N * sizeof(float) );
    cudaMalloc((void**) &result_d, N * sizeof(float) );

    // Copy data from host to GPU device.
    cudaMemcpy(x_d, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, N*sizeof(float), cudaMemcpyHostToDevice);

    Timer timer;
    //timer = initTimer(1);
    startTimer(&timer);

    // Perform parallel computation.
    const unsigned int numThreadsPerBlock = 512;
    const unsigned int numBlocks = (N + numThreadsPerBlock - 1) / numThreadsPerBlock;
    vec_add_kernel<<<numBlocks, numThreadsPerBlock>>>(x_d, y_d, result_d, N);
    cudaDeviceSynchronize();
    stopTimer(&timer);
    printElapsedTime(&timer, "GPU Kernel Execution Time:", GREEN);

    // Copy results from GPU device to host memory.
    cudaMemcpy(result, result_d, N*sizeof(float), cudaMemcpyDeviceToHost);

    // Deallocate memory on GPU device.
    cudaFree((void*)x_d);
    cudaFree((void*)y_d);
    cudaFree((void*)result_d);

}

void vec_add_CPU(float *x, float *y, float *result, unsigned int N) {
    for(unsigned int i = 0; i<N; i++)
    {
        result[i] = x[i] + y[i];
    }
}

int main(int argc, char **argv)
{
    cudaDeviceSynchronize();

    unsigned int N = (argc > 1)?(atoi(argv[1])):(1 << 25);

    // Allocate memory on the host.
    float *x = (float*) malloc(N * sizeof(float));
    float *y = (float*) malloc(N * sizeof(float));
    float *result = (float*) malloc(N * sizeof(float));

    // create a random data for addition.
    for (unsigned int i = 0; i < N; i++) {
        x[i] = rand();
        y[i] = rand();
    }

    Timer timer;
    timer = initTimer(1);
    startTimer(&timer);
    vec_add_CPU(x, y, result, N);
    stopAndPrintElapsed(&timer, "CPU Execution Time: ", CYAN);

    //Timer timer_gpu;
    timer = initTimer(1);
    startTimer(&timer);
    vec_add_GPU(x, y, result, N);
    stopAndPrintElapsed(&timer, "GPU End to End Executiom Time: ", GREEN);

    free(x);
    free(y);
    free(result);

    return 0;
}