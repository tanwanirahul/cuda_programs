#include "stdlib.h"
#include "img_utils.h"

__global__ void rgb_to_grey_kernel(unsigned char * red, unsigned char * green, unsigned char * blue, unsigned char * grey, unsigned int width, unsigned int height) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        unsigned int i = (row*width) + col;
        grey[i] = (red[i] * 0.3) + (green[i] * 0.6) + (blue[i] * 0.1);
    }
}

void rgb_to_grey_wrapper(unsigned char * red, unsigned char * green, unsigned char * blue, unsigned char * grey, unsigned int width, unsigned int height) {

    // Allocate memory on the GPU device.
    unsigned char *red_d, *green_d, *blue_d, *grey_d;
    size_t image_size = sizeof(unsigned char) * width * height;

    cudaMalloc((void **) &red_d, image_size);
    cudaMalloc((void **) &green_d, image_size);
    cudaMalloc((void **) &blue_d, image_size);
    cudaMalloc((void **) &grey_d, image_size);
    cudaDeviceSynchronize(); 

    // Copy data from Host to GPU.
    cudaMemcpy(red_d, red, image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(green_d, green, image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(blue_d, blue, image_size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize(); 
    
    // Do the computation on the device.
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y ); 
    rgb_to_grey_kernel<<< numBlocks, threadsPerBlock >>>(red_d, green_d, blue_d, grey_d, width, height);
    cudaDeviceSynchronize(); 
 
    
    // Copy the results back from GPU  to Host.
    cudaMemcpy(grey, grey_d, image_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Deallocate memory from the GPU device.
    cudaFree(red_d);
    cudaFree(green_d);
    cudaFree(blue_d);
    cudaFree(grey_d);
    cudaDeviceSynchronize();

}

int main(int argc, char** argv) {

    cudaDeviceSynchronize();
    unsigned int width = 1024;
    unsigned int height = 1024;

    // Create a sample color image.
    Image image = create_sample_ppm_image(width, height);

    // optionally, you can save the image.
    write_ppm_image(&image, "color_image.ppm");
    
    // Allocate memory on the host to hold output - grey image.
    size_t image_size = sizeof(unsigned char) * width * height;
    unsigned char * grey = (unsigned char*) malloc(image_size);

    //printf("Contents of grey before kernel call in main: \n");
    //for(int i=0; i < width; i++) {
    //  printf("%d: %d  %d %d    %d\n", i, image.red[i], image.green[i], image.blue[i], grey[i]);
    //}

    // Convert the image to grey;
    rgb_to_grey_wrapper(image.red, image.green, image.blue, grey, width, height);
    cudaDeviceSynchronize();

    //printf("Contents of grey in main: \n");
    //for(int i=0; i < width; i++) {
    //  printf("%d: %d  %d %d   %d\n", i, image.red[i], image.green[i], image.blue[i], grey[i]);
    //}

    // Create a GreyImage and save it in .pgm format.
    GreyImage grey_image;
    grey_image.width = width;
    grey_image.height = height;
    grey_image.maxVal = 255;
    grey_image.grey = grey;

    write_pgm_image(&grey_image, "grey_image.pgm");

    // Free up allocated space on the host for RGB and grey images.
    unload_ppm_image(&image);
    free(grey);

    return 0;
}