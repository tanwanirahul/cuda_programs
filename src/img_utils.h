#ifndef IMG_UTILS_H
#define IMG_UTILS_H

#include<stdio.h>
#include<stdlib.h>
#include <stdbool.h>

typedef struct ImageInfo {
    bool status;
    int width;
    int height;
    int maxVal;
    unsigned char * red, *green, *blue;
} Image;


/*
* Given the filename, loads the ppm image and returns the decoded Image structure.
*/ 
Image load_ppm_image(char * filename) {
    Image img;
    int c;
    FILE *fp;

    //open PPM file for reading
    fp = fopen(filename, "rb");
    if (!fp) {
        img.status = false;
        return img;
    }

    // Read the header of the PPM file to extract the height and width information.
    // The first two characteers should match the magic number P6.
    if (getc(fp) != 'P' ||getc(fp) != '6') {
        img.status = false;
        return img;
    }

    //Skip comments if present.
    c = getc(fp);
    while (c == '#') {
    while (getc(fp) != '\n') ;
         c = getc(fp);
    }
    ungetc(c, fp);


    // Read the width and height information from the header.
    //read image size information
    if (fscanf(fp, "%d %d", &img.width, &img.height) != 2) {
        img.status = false;
        return img;
    }

    //read max value of the image.
    if (fscanf(fp, "%d", &img.maxVal) != 1) {
        img.status = false;
        return img;
    }

    //Assert mac value is less than 255 to read the color information in a single byte.
    if (img.maxVal > 255) {
        img.status = false;
        return img;
    }

    // Skip through the remaining line.
    while (fgetc(fp) != '\n') ;

    // Allocate memory to read the image into Red, Green, Blue channel arrays.
    img.red = (unsigned char *) malloc(img.width * img.height * sizeof(char));
    img.green = (unsigned char *) malloc(img.width * img.height * sizeof(char));
    img.blue = (unsigned char *) malloc(img.width * img.height * sizeof(char));

    // Read width * height * 3 bytes from the file and load into
    // appropriate channels.
    for(int i = 0; i<img.width * img.height; i++) {
        img.red[i] = fgetc(fp);
        img.green[i] = fgetc(fp);
        img.blue[i] = fgetc(fp);
    }

    fclose(fp);

    img.status=true;
    return img;
}


/*
    Given the pointer to an Image, saves this image in the PPM format
    to the given filename.
*/
bool write_ppm_image(Image* img, char * filename) {
    FILE *out_file;
    out_file = fopen(filename, "wb");

    fprintf(out_file, "P6 %d %d %d\n", img->width, img->height, img->maxVal);
    for(int i = 0; i<img->width*img->height; i++) {
        fputc(img->red[i], out_file);
        fputc(img->green[i], out_file);
        fputc(img->blue[i], out_file);
    }
    fclose(out_file);
    return true;
}

Image create_sample_ppm_image(int width, int height) {
    Image img;
    img.maxVal = 255;
    img.width = width;
    img.height = height;

    img.red = (unsigned char*) malloc(img.width * img.height * sizeof(char));
    img.green = (unsigned char*) malloc(img.width * img.height * sizeof(char));
    img.blue = (unsigned char*) malloc(img.width * img.height * sizeof(char));

    for(int r=0; r<img.width; r++) {
        for(int g=0; g<img.height; g++) {
        img.red[r*img.width + g] = r%256;
        img.green[r*img.width + g] = g%256;
        img.blue[r*img.width + g] = 20;
        }
    }
    img.status = true;
    return img;
}
/*
 Given the image that was previously loaded, unloads it by freeing the dynamically
 allocated space to hold the image content.
*/
bool unload_ppm_image(Image *image) {
    free(image->red);
    free(image->green);
    free(image->blue);
    return true;
}

#endif