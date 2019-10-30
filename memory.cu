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
        ~HostPointer() noexcept {
            delete[] ptr;
        }
        friend class DevicePointer<T>;


    private:
        HostPointer() {}
        T *ptr;
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
            HostPointer<T> ret;
            ret.ptr = new T[size];
            check_result(cudaMemcpy(ret, ptr, size * sizeof(*ptr), cudaMemcpyDeviceToHost));
            return ret;
        }

        void copy_from_host(T *host) const {
            check_result(cudaMemcpy(ptr, host, size * sizeof(*ptr), cudaMemcpyHostToDevice));
        }

    private:
        T *ptr;
        size_t size;
};