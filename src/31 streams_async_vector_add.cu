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
    unsigned int numSegments = 32;
    unsigned int segmentSize = (N + numSegments - 1) / numSegments;

    float *x_d, *y_d, *result_d;
    Timer timer;
    timer = initTimer(1);
    startTimer(&timer);
    cudaMalloc((void**) &x_d, N * sizeof(float) );
    cudaMalloc((void**) &y_d, N * sizeof(float) );
    cudaMalloc((void**) &result_d, N * sizeof(float) );
    stopTimer(&timer);
    printElapsedTime(&timer, "Time to allocate memory on GPU:", CYAN);

    timer = initTimer(1);
    startTimer(&timer);

    // Define CudaStreams.
    cudaStream_t streams[numSegments];

    // launch the streams.
    for(unsigned int i=0; i<numSegments; i++) {
        unsigned int segStart = (i * segmentSize);
        unsigned int segN = (segStart + segmentSize) < N? segmentSize: N - segStart;

        // Create/Init a cuda stream.
        cudaStreamCreate(&streams[i]);

        // Copy data from host to GPU device.
        cudaMemcpyAsync(&x_d[segStart], &x[segStart], segN * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(&y_d[segStart], &y[segStart], segN * sizeof(float), cudaMemcpyHostToDevice, streams[i]);

        // Perform parallel computation.
        const unsigned int numThreadsPerBlock = 64;
        const unsigned int numBlocks = (segN+ numThreadsPerBlock - 1) / numThreadsPerBlock;
        vec_add_kernel<<<numBlocks, numThreadsPerBlock, 0, streams[i]>>>(&x_d[segStart], &y_d[segStart], &result_d[segStart], segN);
    
        // Copy results from GPU device to host memory.
        cudaMemcpyAsync(&result[segStart], &result_d[segStart], segN*sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }    
    cudaDeviceSynchronize();
    stopTimer(&timer);
    printElapsedTime(&timer, "Time to copy data from Host to GPU, run CUDA kernel, and copy results back: ", GREEN);
    
    // After all the streams have been launched, we need to destroy the streams.
    for(unsigned int i=0; i<numSegments; i++) {
        // Synchronize stream to make sure all the tasks within the stream have finished. 
        cudaStreamSynchronize(streams[i]);
        // Destroy CUDA stream.
        cudaStreamDestroy(streams[i]);
    }

    // Deallocate memory on GPU device.
    timer = initTimer(1);
    startTimer(&timer);

    cudaFree((void*)x_d);
    cudaFree((void*)y_d);
    cudaFree((void*)result_d);

    cudaDeviceSynchronize();
    stopTimer(&timer);
    printElapsedTime(&timer, "Time to free up GPU allocated space: ", CYAN);

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
    float *result_GPU = (float*) malloc(N * sizeof(float));

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
    vec_add_GPU(x, y, result_GPU, N);

    stopAndPrintElapsed(&timer, "GPU End to End Executiom Time: ", GREEN);

    // Compare the output from CPU and GPU. They should match.
    for(unsigned int i=0; i<N; i++)
    {
        if(result[i] != result_GPU[i]){
            printf("\nMismatch found at %d. CPU result: %16.2f,   GPU result: %16.2f.\n", i, result[i], result_GPU[i]);
            break;
        }
    }

    for(int i=0; i<100; i++) {
        printf("%d: %f.   %f    %f\n", i, x[i], y[i], result_GPU[i]);
    }

    free(x);
    free(y);
    free(result);

    return 0;
}