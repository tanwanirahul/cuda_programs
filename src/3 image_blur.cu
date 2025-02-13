%%writefile image_blur.cu

#include "stdlib.h"
#include "img_utils.h"
#include "timer.h"

__global__ void image_blur_kernel(unsigned char * original, unsigned char * blur, unsigned int blur_size, unsigned int width, unsigned int height) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height and col < width) {
        unsigned int average = 0;
        for(int j = 0; j < (2 * blur_size + 1); j++) {
            for(int k = 0; k < (2 * blur_size + 1); k++) {
                int input_row = row - blur_size + j;
                int input_col = col - blur_size + k;
                if(input_row >= 0 && input_row < height && input_col >=0 && input_col < width) {
                    average+= original[input_row * width + input_col];
                }
            }
        }
      blur[row*width+col] = (unsigned char) (average / ((2 * blur_size + 1) * (2 * blur_size + 1))); 
    }
}

void image_blur_wrapper(unsigned char * original, unsigned char * blur, unsigned int blur_size, unsigned int width, unsigned int height) {

    // Allocate memory on the GPU device.
    unsigned char *original_d, *blur_d;
    size_t image_size = sizeof(unsigned char) * width * height;

    Timer timer;
    timer = initTimer(1);
    startTimer(&timer);

    cudaMalloc((void **) &original_d, image_size);
    cudaMalloc((void **) &blur_d, image_size);
    cudaDeviceSynchronize();
    stopAndPrintElapsed(&timer, "GPU Device Memory Allocation Time: ", CYAN);


    // Copy data from Host to GPU.
    timer = initTimer(1);
    startTimer(&timer);
    cudaMemcpy(original_d, original, image_size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopAndPrintElapsed(&timer, "Time To Copy Data to GPU DRAM: ", CYAN);

    // Do the computation on the device.
    timer = initTimer(1);
    startTimer(&timer);
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y ); 
    image_blur_kernel<<<numBlocks, threadsPerBlock>>>(original_d, blur_d, blur_size, width, height);
    cudaDeviceSynchronize();
    stopAndPrintElapsed(&timer, "CUDA Kernel Execution Time: ", GREEN);

    // Copy the results back from GPU  to Host.
    timer = initTimer(1);
    startTimer(&timer);
    cudaMemcpy(blur, blur_d, image_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopAndPrintElapsed(&timer, "Time to COPY Results from GPU to HOST: ", CYAN);

    // Deallocate memory from the GPU device.
    cudaFree(original_d);
    cudaFree(blur_d);
}

int main(int argc, char** argv) {
    unsigned int width = 10240;
    unsigned int height = 10240;
    unsigned int blur_size = 5;

    size_t image_size = sizeof(unsigned char) * width * height;

    // Allocate memory on the host for blur image.
    unsigned char * blur = (unsigned char*) malloc(image_size);

    // Create a sample greyscale image.
    GreyImage image = create_sample_pgm_image(width, height);

    // optionally, save the image for reference.
    write_pgm_image(&image, "original_grey_image.ppm");

    // Convert the image to grey;
    image_blur_wrapper(image.grey, blur, blur_size, width, height);
    cudaDeviceSynchronize();

    GreyImage blur_image;
    blur_image.maxVal = 255;
    blur_image.width = width;
    blur_image.height = height; 
    blur_image.grey = blur;

    // Save the blur image;
    write_pgm_image(&blur_image, "blur_grey_image.ppm");

    //printf("First 100 elements of the orignal and blur image:\n");
    //int limit = (height * width) > width? width: (height * width);
    //for(int i=0; i<limit; i++) {
    //  printf("%d: %d       %d\n", i, image.grey[i], blur[i]);
    //}

    // Free up allocated space on the host for original and blur image.
    unload_pgm_image(&image);
    free(blur);

    return 0;
}