#ifndef TENSOR_H
#define TENSOR_H

#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <vector>
#include <ostream>

using namespace std;

class Tensor {
private: 
    double* data_; //CREACION DEL ARRAY DINAMICO CONTIGUO
    size_t* ref_count_;
    vector<size_t> shape_;

    size_t numel() const;
    void release();
    void allocate_and_zero();
    void check_dimension_limit(const vector<size_t>& shape) const;
    static size_t product(const vector<size_t>& shape);

public:

    Tensor(const vector<size_t>& shape,
           const vector<double>& values);

    explicit Tensor(const vector<size_t>& shape);

    Tensor(const Tensor& other);

    Tensor(Tensor&& other) noexcept;

    Tensor& operator=(const Tensor& other);

    Tensor& operator=(Tensor&& other) noexcept;

    ~Tensor();

    static Tensor zeros(const vector<size_t>& shape);
    static Tensor ones(const vector<size_t>& shape);
    static Tensor random(const vector<size_t>& shape,
                         double min, double max);
    static Tensor arange(double start, double end);

    const vector<size_t>& shape() const;
    size_t size() const;
    size_t ndim() const;

    double& operator[](size_t index);
    const double& operator[](size_t index) const;

    double& at(size_t i);
    const double& at(size_t i) const;

    double& at(size_t i, size_t j);
    const double& at(size_t i, size_t j) const;

    double& at(size_t i, size_t j, size_t k);
    const double& at(size_t i, size_t j, size_t k) const;

    Tensor view(const vector<size_t>& new_shape) const;

    Tensor unsqueeze(size_t dim) const;

    static Tensor concat(const vector<Tensor>& tensors,
                         size_t dim);

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator*(double scalar) const;

    friend Tensor operator*(double scalar, const Tensor& tensor);

    Tensor relu() const;
    Tensor sigmoid() const;

    void print_shape(ostream& os) const;
    void print(ostream& os) const;
};

Tensor dot(const Tensor& a, const Tensor& b);
Tensor matmul(const Tensor& a, const Tensor& b);

#endif
