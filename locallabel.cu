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

struct Entry {
    uint32_t r;
    uint32_t g;
    uint32_t b;
    uint64_t r2;
    uint64_t g2;
    uint64_t b2;
    uint16_t count;

    Entry(): r(0), g(0), b(0), r2(0), g2(0), b2(0), count(0) {}
};

int main() {
    cudaFree(nullptr);
    const Pixel distinct_colours[] = {Pixel(230, 25, 75), Pixel(60, 180, 75), Pixel(255, 225, 25), Pixel(0, 130, 200), Pixel(245, 130, 48), Pixel(145, 30, 180), Pixel(70, 240, 240), Pixel(240, 50, 230), Pixel(210, 245, 60), Pixel(250, 190, 190), Pixel(0, 128, 128), Pixel(230, 190, 255), Pixel(170, 110, 40), Pixel(255, 250, 200), Pixel(128, 0, 0), Pixel(170, 255, 195), Pixel(128, 128, 0), Pixel(255, 215, 180), Pixel(0, 0, 128), Pixel(128, 128, 128), Pixel(255, 255, 255), Pixel(0, 0, 0)};

    // Load the image
    Timer timer;
    Image image("practical.png");
    assert(image.ok());
    Image mask(image);

    auto loaded = image_to_device(image);
    timer.report("loading image");

    auto thresholded = threshold_image(loaded, image.width(), image.height(), 180);
    auto final_map = label_components(thresholded, image.width(), image.height());

    timer.reset();
    std::unordered_map<uint32_t, Entry> results;
    const int max_size = 25000;
    const int min_size = 5000;
    for (auto y = 0; y < image.height(); ++y) {
        for (auto x = 0; x < image.width(); ++x) {
            auto value = final_map.at(x, y, image.width());
            auto pixel = image.at(x, y);
            if (results.count(value) == 0) {
                results.insert(std::pair<uint32_t, Entry>(value, Entry()));
            }

            Entry& entry = results.find(value)->second;
            if (entry.count < max_size) {
                entry.r += pixel.r;
                entry.r2 += pixel.r * pixel.r;
                entry.g += pixel.g;
                entry.g2 += pixel.g * pixel.g;
                entry.b += pixel.b;
                entry.b2 += pixel.b * pixel.b;
                ++entry.count;
            }
        }
    }
    for (const auto& it : results) {
        auto entry = it.second;
        if (entry.count > min_size && entry.count < max_size) {
            double rmean = (entry.r / 255.0) / entry.count;
            double gmean = (entry.g / 255.0) / entry.count;
            double bmean = (entry.b / 255.0) / entry.count;
            printf("found a value: %d (%.2f, %.2f, %.2f)\n", entry.count, rmean, gmean, bmean);
        }
    }
    timer.report("extracting values");

    timer.reset();
    // Produce result mask
    int i = 0;
    std::unordered_map<int, Pixel> colours;
    for (auto y = 0; y < image.height(); ++y) {
        for (auto x = 0; x < image.width(); ++x) {
            auto value = final_map.at(x, y, image.width());
            if (results[value].count > min_size && results[value].count < max_size) {
                
                if (colours.count(value) == 0) {
                    colours.insert(std::pair<int, Pixel>(value, distinct_colours[++i]));//Pixel(rand() % 256, rand() % 256, rand() % 256)));
                }
                mask.set(x, y, colours.find(value)->second);
            }
        }
    }
    mask.save("result.png");
    timer.report("saving image");
    return 0;
}
