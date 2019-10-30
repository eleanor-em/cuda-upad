#include <cstdio>
#include <cmath>
#include <unordered_map>
#include "image.h"
#include "kernel.cu"
#include "memory.cu"

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
    // Load the image
    Image image("sample2048.png");
    assert(image.ok());
    Image mask(image);

    // Perform thresholding
    uint8_t buffer[image.width() * image.height()];
    size_t i = 0;
    for (const Pixel& pix : image) {
        buffer[i++] = pix.r / 255;
    }
    
    DevicePointer<uint8_t> vals(image.width() * image.height());
    vals.copy_from_host(buffer);
    auto final_map = label_components(vals, image.width(), image.height());

    // Produce result mask
    std::unordered_map<int, Pixel> colours;
    for (auto y = 0; y < image.height(); ++y) {
        for (auto x = 0; x < image.width(); ++x) {
            auto value = final_map.at(x, y, image.width());
            
            if (colours.count(value) == 0) {
                colours.insert(std::pair<int, Pixel>(value, Pixel(rand() % 256, rand() % 256, rand() % 256)));
            }
            mask.set(x, y, colours.find(value)->second);
        }
    }
    mask.save("result.png");
    printf("saved output\n");
    return 0;
}
