#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <png.h>

#define PNG_NO_SETJMP

const int MAX_ITER = 10000;

void write_png(const char *filename, const size_t width, const size_t height, const int *buffer);
void calculate_mandelbrot(int *image, int width, int height, double upper, double lower, double right, double left);

int main(int argc, char *argv[]) {
    MPI_Init(NULL, NULL);
    
    double left = strtod(argv[2], NULL);
    double right = strtod(argv[3], NULL);
    double lower = strtod(argv[4], NULL);
    double upper = strtod(argv[5], NULL);
    int width = strtol(argv[6], NULL, 10);
    int height = strtol(argv[7], NULL, 10);
    const char *filename = argv[8];

    int my_rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    int *image = NULL;
    int *local_image = NULL;

    int step = 1;
    local_image = (int *)malloc(step * width * sizeof(int));

    if (my_rank == 0) {
        image = (int *)malloc(width * height * sizeof(int));
        calculate_mandelbrot(image, width, height, upper, lower, right, left);
    } 

    // Broadcast image dimensions
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter the work among processes
    MPI_Scatter(image, step * width, MPI_INT, local_image, step * width, MPI_INT, 0, MPI_COMM_WORLD);

    // Process the local image
    // (Assuming the calculation logic inside the loop is the same as before)

    // Gather the processed parts
    MPI_Gather(local_image, step * width, MPI_INT, image, step * width, MPI_INT, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        write_png(filename, width, height, image);
        free(image);
    }
    
    free(local_image);
    MPI_Finalize();
    return 0;
}

void write_png(const char *filename, const size_t width, const size_t height, const int *buffer) {
    FILE *fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png_ptr, info_ptr);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != MAX_ITER) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

void calculate_mandelbrot(int *image, int width, int height, double upper, double lower, double right, double left) {
    for (int j = 0; j < height; ++j) {
        double y0 = j * ((upper - lower) / height) + lower;
        for (int i = 0; i < width; ++i) {
            double x0 = i * ((right - left) / width) + left;

            int repeats = 0;
            double x = 0;
            double y = 0;
            double length_squared = 0;
            for (; repeats < MAX_ITER && length_squared < 4; ++repeats) {
                double temp = x * x - y * y + x0;
                y = 2 * x * y + y0;
                x = temp;
                length_squared = x * x + y * y;
            }
            image[j * width + i] = repeats;
        }
    }
}

