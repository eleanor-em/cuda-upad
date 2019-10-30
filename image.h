#ifndef __UPAD_IMAGE_H
#define __UPAD_IMAGE_H
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

struct Pixel {
    uint8_t r;
    uint8_t g;
    uint8_t b;

    Pixel(): r(0), g(0), b(0) {}
    Pixel(uint8_t r, uint8_t g, uint8_t b): r(r), g(g), b(b) {}
};

// Overload to allow printing a pixel
std::ostream& operator<<(std::ostream& os, const Pixel &p) {
    os << "(" << (int) p.r << ", " << (int) p.g << ", " << (int) p.b << ")";
    return os;
}

class Image {
    public:
        // Load the image buffer
        Image(const char *fname):
            fname(fname) {
            image = stbi_load(fname, &_width, &_height, &bpp, 3);
        }

        Image(const Image& other) {
            fname = other.fname;
            image = stbi_load(other.fname, &_width, &_height, &bpp, 3);
        }

        // Returns true if the image was successfully loaded
        bool ok() const {
            return image != NULL;
        }

        // Returns whether the given coordinate is inside the image bounds
        bool inside(int x, int y) const {
            return x >= 0 && x < _width && y >= 0 && y < _height;
        }

        // Returns the colour at the given coordinate
        Pixel at(int x, int y) const {
            Pixel pixel;
            pixel.r = image[y * _width * 3 + 3 * x];
            pixel.g = image[y * _width * 3 + 3 * x + 1];
            pixel.b = image[y * _width * 3 + 3 * x + 2];
            return pixel;
        }

        void set(int x, int y, Pixel pixel) {
            image[y * _width * 3 + 3 * x] = pixel.r;
            image[y * _width * 3 + 3 * x + 1] = pixel.g;
            image[y * _width * 3 + 3 * x + 2] = pixel.b;
        }

        void clear() {
            for (int i = 0; i < _width * _height * 3; ++i) {
                image[i] = 255;
            }
        }

        void save(const char *fname) {
            stbi_write_png(fname, _width, _height, 3, image, 0);
        }

        int width() const {
            return _width;
        }

        int height() const {
            return _height;
        }

        ~Image() {
            stbi_image_free(image);
        }

        class Iterator {
            public:
                Iterator(const Image& image)
                    : image(image), x(0), y(0) { }
                
                void set_end() {
                    x = 0;
                    y = image.height();
                }

                Iterator& operator++() {
                    ++x;
                    if (x == image.width()) {
                        x = 0;
                        ++y;
                    }

                    return *this;
                }

                bool operator==(const Iterator& rhs) const {
                    return x == rhs.x && y == rhs.y;
                }
                bool operator!=(const Iterator& rhs) const {
                    return !(*this == rhs);
                }

                Pixel operator*() const {
                    return image.at(x, y);
                }

            private:
                const Image& image;
                int x;
                int y;
        };

        Iterator begin() const {
            return Iterator(*this);
        }

        Iterator end() const {
            Iterator it(*this);
            it.set_end();
            return it;
        }
    
    private:
        const char *fname;
        uint8_t *image;
        int _width;
        int _height;
        int bpp;
};
#endif // __UPAD_IMAGE_H