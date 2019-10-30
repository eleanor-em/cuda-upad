template <class T>
__device__ __host__ void make_set(T *parent, T value) {
    parent[value] = value;
}

template <class T>
__device__ __host__ T find_root(T *parent, T index) {
    T current = index;
    while (current != parent[current]) {
        index = parent[current];
        // Shorten the path a bit along the way
        // TODO: proper path compression?
        parent[current] = parent[index];
        current = index;
    }
    return current;
}

template <class T>
__device__ __host__ void merge(T *parent, T x, T y) {
    T x_root = find_root(parent, x);
    T y_root = find_root(parent, y);
    if (x_root < y_root) {
        parent[y_root] = x_root;
    }
    if (y_root < x_root) {
        parent[x_root] = y_root;
    }
}