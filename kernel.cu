#ifndef _KERNEL_CU
#define _KERNEL_CU

#include <chrono>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include "memory.cu"
#include "unionfind.cu"

void finalise_kernel() {
    auto result = cudaDeviceSynchronize();
    if (result != 0) {
        printf(cudaGetErrorName(result));
    }
}

class Timer {
    public:
        Timer() {
            reset();
        }

        void reset() {
            start = std::chrono::high_resolution_clock::now();
        }

        void report(const char *name) {
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            printf("%s ran in %fms\n", name, diff.count() * 1000);
        }
    private:
        std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

template<size_t block_size>
__global__ void local_ccl(DevicePointer<uint8_t>::Raw image, uint16_t image_width, DevicePointer<uint32_t>::Raw labelmap) {
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

__global__ void seam_stitch(DevicePointer<uint8_t>::Raw image, uint16_t image_width, DevicePointer<uint32_t>::Raw labelmap) {
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

__global__ void find_all_roots(DevicePointer<uint32_t>::Raw labelmap) {
    uint32_t tid = (blockIdx.x + gridDim.x * blockIdx.y) * blockDim.x * blockDim.y + threadIdx.x + blockDim.x * threadIdx.y;
    labelmap[tid] = find_root(labelmap, tid);
}

__global__ void apply_threshold(DevicePointer<Pixel>::Raw in, uint16_t image_width, uint16_t threshold, DevicePointer<uint8_t>::Raw out) {
    uint16_t x = blockDim.x * blockIdx.x + threadIdx.x;
    uint16_t y = blockDim.y * blockIdx.y + threadIdx.y;

    uint16_t r, g, b;
    r = in[y * image_width + x].r;
    g = in[y * image_width + x].g;
    b = in[y * image_width + x].b;
    r += in[(y - 1) * image_width + x].r;
    g += in[(y - 1) * image_width + x].g;
    b += in[(y - 1) * image_width + x].b;
    r += in[(y + 1) * image_width + x].r;
    g += in[(y + 1) * image_width + x].g;
    b += in[(y + 1) * image_width + x].b;
    r += in[y * image_width + x - 1].r;
    g += in[y * image_width + x - 1].g;
    b += in[y * image_width + x - 1].b;
    r += in[(y - 1) * image_width + x - 1].r;
    g += in[(y - 1) * image_width + x - 1].g;
    b += in[(y - 1) * image_width + x - 1].b;
    r += in[(y + 1) * image_width + x - 1].r;
    g += in[(y + 1) * image_width + x - 1].g;
    b += in[(y + 1) * image_width + x - 1].b;
    r += in[y * image_width + x + 1].r;
    g += in[y * image_width + x + 1].g;
    b += in[y * image_width + x + 1].b;
    r += in[(y - 1) * image_width + x + 1].r;
    g += in[(y - 1) * image_width + x + 1].g;
    b += in[(y - 1) * image_width + x + 1].b;
    r += in[(y + 1) * image_width + x + 1].r;
    g += in[(y + 1) * image_width + x + 1].g;
    b += in[(y + 1) * image_width + x + 1].b;

    threshold *= 9;
    out[y * image_width + x] = !(r > threshold || g > threshold || b > threshold);
}

DevicePointer<Pixel> image_to_device(const Image& image) {
    Pixel *pixels = new Pixel[image.width() * image.height()];
    
    size_t i = 0;
    for (const Pixel& pix: image) {
        pixels[i++] = pix;
    }

    DevicePointer<Pixel> vals(image.width() * image.height());
    vals.copy_from_host(pixels);
    delete[] pixels;
    return vals;
}

DevicePointer<uint8_t> threshold_image(const DevicePointer<Pixel>& image, uint16_t width, uint16_t height, uint8_t threshold) {
    const uint8_t block_size = 32;
    const uint16_t blocks_per_row = width > block_size ? width / block_size : 1;
    const uint16_t blocks_per_col = height > block_size ? height / block_size : 1;
    const dim3 image_grid(blocks_per_row, blocks_per_col);
    const dim3 image_block(block_size, block_size);

    DevicePointer<uint8_t> result(width * height);

    Timer timer;
    apply_threshold<<<image_grid, image_block>>>(image.as_raw(), width, threshold, result.as_raw());
    finalise_kernel();
    timer.report("threshold kernel");

    return result;
}

DevicePointer<uint32_t> label_components(const DevicePointer<uint8_t>& image, uint16_t width, uint16_t height) {
    const uint8_t block_size = 32;
    const uint16_t blocks_per_row = width > block_size ? width / block_size : 1;
    const uint16_t blocks_per_col = height > block_size ? height / block_size : 1;
    const dim3 image_grid(blocks_per_row, blocks_per_col);
    const dim3 image_block(block_size, block_size);
    const dim3 seam_grid(blocks_per_row - 1, blocks_per_col);
    const dim3 seam_block(block_size, block_size);

    DevicePointer<uint32_t> labelmap(width * height);

    // Locally label connected components
    Timer timer, total_timer;
    local_ccl<block_size><<<image_grid, image_block>>>(image.as_raw(), width, labelmap.as_raw());
    finalise_kernel();
    timer.report("local-CCL kernel");

    // Stitch seams together
    timer.reset();
    seam_stitch<<<seam_grid, seam_block>>>(image.as_raw(), width, labelmap.as_raw());
    finalise_kernel();
    timer.report("seam-stitch kernel");

    // Find the root of all components
    timer.reset();
    find_all_roots<<<image_grid, image_block>>>(labelmap.as_raw());
    timer.report("find-all-roots kernel");

    total_timer.report("total");

    return labelmap;
}
#endif // _KERNEL_CU