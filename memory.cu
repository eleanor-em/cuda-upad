#ifndef _MEMORY_CU
#define _MEMORY_CU
#include <cstdio>
#include <cassert>
#include <cstring>
#include <utility>

template <class T>
static void check_result(T result) {
    if (result != 0) {
        printf("%s\n", cudaGetErrorName(result));
        assert(false);
    }
}

template <class T>
class DevicePointer;

template <class T>
class HostPointer {
    public:
        HostPointer() {
            size = 1;
            ptr = new T[1];
        }
        HostPointer(size_t size)
        : size(size) {
            ptr = new T[size];
        }

        HostPointer(const HostPointer<T>& other) {
            size = other.size;
            ptr = new T[size];
            // not necessarily safe if T is not a POD
            memcpy(ptr, other.ptr, size * sizeof(*ptr));
        }

        HostPointer(HostPointer<T>&& other) noexcept {
            ptr = other.ptr;
            size = other.size;
            other.ptr = nullptr;
            other.size = 0;
        }

        HostPointer<T>& operator= (const HostPointer<T>& other) {
            auto tmp(other);
            *this = std::move(tmp);
            return *this;
        }

        HostPointer<T>& operator= (HostPointer<T>&& other) {
            if (this != &other) {
                delete[] ptr;
                ptr = other.ptr;
                size = other.size;
                other.ptr = nullptr;
                other.size = 0;
            }
            return *this;
        }

        ~HostPointer() noexcept {
            delete[] ptr;
        }

        friend class DevicePointer<T>;

        const T& at(size_t i) const {
            assert(i < size);
            return ptr[i];
        }

        const T& at(size_t x, size_t y, size_t width) const {
            return at(y * width + x);
        }

    private:
        T *ptr;
        size_t size;
};

template <class T>
class DevicePointer {
    public:
        typedef T* Raw;

        DevicePointer() {
            size = 1;
            check_result(cudaMalloc(&ptr, sizeof(*ptr)));
        }

        DevicePointer(size_t elems) {
            size = elems;
            check_result(cudaMalloc(&ptr, size * sizeof(*ptr)));
        }

        DevicePointer(const DevicePointer<T>& other) {
            size = other.size;
            check_result(cudaMalloc(&ptr, size * sizeof(*ptr)));
            check_result(cudaMemcpy(ptr, other.ptr, size * sizeof(*ptr), cudaMemcpyDeviceToDevice));
        }

        DevicePointer(DevicePointer<T>&& other) noexcept {
            ptr = other.ptr;
            size = other.size;
            other.ptr = nullptr;
            other.size = 0;
        }

        DevicePointer<T>& operator= (const DevicePointer<T>& other) {
            auto tmp(other);
            *this = std::move(tmp);
            return *this;
        }

        DevicePointer<T>& operator= (DevicePointer<T>&& other) {
            if (this != &other) {
                check_result(cudaFree(ptr));
                ptr = other.ptr;
                size = other.size;
                other.ptr = nullptr;
                other.size = 0;
            }
            return *this;
        }

        ~DevicePointer() noexcept {
            check_result(cudaFree(ptr));
        }

        Raw as_raw() const {
            return ptr;
        }

        HostPointer<T> as_host() const {
            HostPointer<T> ret(size);
            check_result(cudaMemcpy(ret.ptr, ptr, size * sizeof(*ptr), cudaMemcpyDeviceToHost));
            return ret;
        }

        void copy_from_host(T *host) const {
            check_result(cudaMemcpy(ptr, host, size * sizeof(*ptr), cudaMemcpyHostToDevice));
        }

    private:
        T *ptr;
        size_t size;
};
#endif // _MEMORY_CU