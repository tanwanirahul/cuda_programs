#include "stdlib.h"
#include "img_utils.h"
#include "timer.h"

#define BLOCK_SIZE 1024
#define NUM_BINS 256

__global__ void histogram_kernel(unsigned char * image, unsigned int * bins, unsigned int width, unsigned int height) {

    unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < width * height) {
        unsigned char bin = image[i];
        atomicAdd(&bins[bin], 1);
    }

}

void histogram_wrapper(unsigned char * image, unsigned int* bins, unsigned int width, unsigned int height) {

    // Allocate memory on the GPU device.
    unsigned char *image_d;
    unsigned int *bins_d;

    size_t image_size = sizeof(unsigned char) * width * height;

    Timer timer;
    timer = initTimer(1);
    startTimer(&timer);

    cudaMalloc((void **) &image_d, image_size);
    cudaMalloc((void **) &bins_d, sizeof(unsigned int) * NUM_BINS);
    cudaDeviceSynchronize();
    stopAndPrintElapsed(&timer, "GPU Device Memory Allocation Time: ", CYAN);


    // Copy data from Host to GPU.
    timer = initTimer(1);
    startTimer(&timer);
    cudaMemcpy(image_d, image, image_size, cudaMemcpyHostToDevice);
    cudaMemset(bins_d, 0, sizeof(unsigned int) * NUM_BINS );
    cudaDeviceSynchronize();
    stopAndPrintElapsed(&timer, "Time To Copy Data to GPU DRAM: ", CYAN);

    // Do the computation on the device.
    timer = initTimer(1);
    startTimer(&timer);
    unsigned int threadsPerBlock  = BLOCK_SIZE;
    unsigned int numBlocks = ((width * height) + threadsPerBlock - 1) / threadsPerBlock; 
    histogram_kernel<<<numBlocks, threadsPerBlock>>>(image_d, bins_d, width, height);
    cudaDeviceSynchronize();
    stopAndPrintElapsed(&timer, "CUDA Kernel Execution Time: ", GREEN);

    // Copy the results back from GPU  to Host.
    timer = initTimer(1);
    startTimer(&timer);
    cudaMemcpy(bins, bins_d, sizeof(unsigned int) * NUM_BINS, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopAndPrintElapsed(&timer, "Time to COPY Results from GPU to HOST: ", CYAN);

    // Deallocate memory from the GPU device.
    cudaFree(image_d);
    cudaFree(bins_d);
}

int main(int argc, char** argv) {
    unsigned int width = 1024;
    unsigned int height = 1024;

    // Allocate memory on the host for holding the output histogram.
    unsigned int * bins = (unsigned int*) malloc(sizeof(unsigned int) * NUM_BINS);
    unsigned int * bins_CPU = (unsigned int*) malloc(sizeof(unsigned int) * NUM_BINS);
    memset(bins_CPU, 0, sizeof(unsigned int) * NUM_BINS);

    // Create a sample greyscale image.
    GreyImage image = create_sample_pgm_image(width, height);

    // Run the histogram to find the bins count.
    histogram_wrapper(image.grey, bins, width, height);
    cudaDeviceSynchronize();


    // CPU implementation of Histogram bins calculation for comparison.
    for(unsigned int i=0; i<(width*height); i++) {
        unsigned char b = image.grey[i];
        bins_CPU[b]++;
    }

    int misMatches = 0.0;
    // compare the output from GPU kernel to CPU.
    for(int i=0; i<NUM_BINS; i++) {
        if(bins_CPU[i] != bins[i]) {
            printf("Mismatch found for bin: %d. Expected: %d, Found: %d\n", i, bins_CPU[i], bins[i]);
            misMatches++;
        }
    }

    if(misMatches == 0) {
        printf("\nOutput matched with CPU implementation!\n");
    }
    else {
        printf("\nTotal mismatches found: %d\n", misMatches);
    }

    unload_pgm_image(&image);
    free(bins);
    free(bins_CPU);

    return 0;
}