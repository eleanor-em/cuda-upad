#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <unordered_map>
#include <chrono>
#include "image.h"
#include "memory.cu"
#include "unionfind.cu"


template<size_t block_size>
__global__ void local_ccl(uint8_t *image, uint16_t image_width, uint32_t *labelmap) {
    uint16_t tid = threadIdx.x + threadIdx.y * blockDim.x;
    uint16_t x = blockDim.x * blockIdx.x + threadIdx.x;
    uint16_t y = blockDim.y * blockIdx.y + threadIdx.y;

    // Initialise shared data
    __shared__ uint16_t labelset[block_size * block_size];
    __shared__ uint8_t subimage[block_size * block_size];
    make_set(labelset, tid);
    subimage[tid] = image[y * image_width + x];
    __syncthreads();

    // Scan rows
    if (tid > 0 && subimage[tid] == subimage[tid - 1]) {
        merge(labelset, tid, (uint16_t) (tid - 1));
    }
    __syncthreads();

    // Scan columns
    if (tid >= block_size && subimage[tid] == subimage[tid - block_size]) {
        merge(labelset, tid, (uint16_t) (tid - block_size));
    }
    __syncthreads();

    // Perform labelling
    uint32_t l = find_root(labelset, tid);
    l = blockIdx.x * blockDim.x + l % blockDim.x + (blockIdx.y * blockDim.y + l / blockDim.x) * image_width;
    labelmap[y * image_width + x] = l;
}

__global__ void seam_stitch(uint8_t *image, uint16_t image_width, uint32_t *labelmap) {
    // Stitch row seams
    uint16_t x = (blockIdx.x + 1) * blockDim.x;
    uint16_t y = blockIdx.y * blockDim.x + threadIdx.x;
    uint32_t dest = y * image_width + x;
    uint32_t src = dest - 1;
    if (image[dest] == image[src]) {
        merge(labelmap, dest, src);
    }
    __syncthreads();

    // Stitch column seams
    x = blockIdx.y * blockDim.x + threadIdx.x;
    y = (blockIdx.x + 1) * blockDim.x;
    dest = y * image_width + x;
    src = (y - 1) * image_width + x;
    if (image[dest] == image[src]) {
        merge(labelmap, dest, src);
    }
}

template <class uint32_t>
__global__ void find_all_roots(uint32_t *labelmap) {
    uint32_t tid = (blockIdx.x + gridDim.x * blockIdx.y) * blockDim.x * blockDim.y + threadIdx.x + blockDim.x * threadIdx.y;
    labelmap[tid] = find_root(labelmap, tid);
}

template <class T>
void print_label_map(T *labelmap, size_t width, size_t height) {
    int max = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (labelmap[y * width + x] > max) {
                max = labelmap[y * width + x];
            }
        }
    }

    int digits = ((int) std::log10(max)) + 1;
    char format[10];
    snprintf(format, 10, "%%%dd ", digits);
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            printf(format, labelmap[y * width + x]);
        }
        printf("\n");
    }
}



int main() {    
    const int block_size = 32;

    // Load the image
    Image image("sample512.png");
    assert(image.ok());
    Image mask(image);

    const uint16_t blocks_per_row = image.width() > block_size ? image.width() / block_size : 1;

    // Perform thresholding
    uint8_t buffer[image.width() * image.height()];
    int i = 0;
    for (const Pixel& pix : image) {
        buffer[i++] = pix.r / 255;
    }

    // Initialise the GPU memory
    uint8_t *vals;
    uint32_t *labelmap;
    auto result = cudaMalloc(&vals, image.width() * image.height() * sizeof(*vals));
    if (result != 0) {
        printf(cudaGetErrorName(result));
    }
    result = cudaMemcpy(vals, buffer, image.width() * image.height() * sizeof(*vals), cudaMemcpyHostToDevice);
    if (result != 0) {
        printf(cudaGetErrorName(result));
    }
    result = cudaMallocManaged(&labelmap, image.width() * image.height() * sizeof(*labelmap));
    if (result != 0) {
        printf(cudaGetErrorName(result));
    }

    // Run the local CCL kernel
    auto total_start = std::chrono::high_resolution_clock::now();
    auto start = std::chrono::high_resolution_clock::now();

    dim3 gridSize = dim3(blocks_per_row, blocks_per_row);
    dim3 blockSize = dim3(block_size, block_size);

    local_ccl<block_size><<<gridSize, blockSize>>>(vals, image.width(), labelmap);

    result = cudaDeviceSynchronize();
    if (result != 0) {
        printf(cudaGetErrorName(result));
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "local-CCL kernel ran in " << diff.count() * 1000 << "ms\n";

    // Run the seam-stitch kernel
    start = std::chrono::high_resolution_clock::now();

    gridSize = dim3(blocks_per_row - 1, blocks_per_row);
    blockSize = dim3(block_size, 1);
    seam_stitch<<<gridSize, blockSize>>>(vals, image.width(), labelmap);

    result = cudaDeviceSynchronize();
    if (result != 0) {
        printf(cudaGetErrorName(result));
    }
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "seam-stitch kernel ran in " << diff.count() * 1000 << "ms\n";

    // Find the root of all components
    start = std::chrono::high_resolution_clock::now();

    gridSize = dim3(blocks_per_row, blocks_per_row);
    blockSize = dim3(block_size, block_size);
    find_all_roots<<<gridSize, blockSize>>>(labelmap);

    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "find-roots kernel ran in in " << diff.count() * 1000 << "ms\n";

    end = std::chrono::high_resolution_clock::now();
    diff = end - total_start;
    std::cout << "total runtime: " << diff.count() * 1000 << "ms\n";

    // Produce result mask
    std::unordered_map<int, Pixel> colours;
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            uint32_t value = find_root(labelmap, (uint32_t) (y * image.width() + x));
            
            if (colours.count(value) == 0) {
                colours.insert(std::pair<int, Pixel>(value, Pixel(rand() % 256, rand() % 256, rand() % 256)));
            }
            mask.set(x, y, colours.find(value)->second);
        }
    }
    mask.save("result.png");
    printf("saved output\n");

    cudaFree(vals);
    cudaFree(labelmap);
    return 0;
}
