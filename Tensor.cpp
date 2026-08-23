#include "Tensor.h"

// Implementacion de los metodos de la clase Tensor y de las funciones algebraicas externas.

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>

using namespace std;

// Calcula el numero total de elementos multiplicando las dimensiones del shape.
size_t Tensor::product(const vector<size_t>& shape) {
    size_t result = 1;

    for (size_t d : shape) {
        if (d == 0) {
            throw invalid_argument("Las dimensiones deben ser mayores que cero.");
        }

        if (result > numeric_limits<size_t>::max() / d) {
            throw overflow_error("El tamano del tensor excede el limite.");
        }

        result *= d;
    }

    return result;
}

size_t Tensor::numel() const {
    return product(shape_);
}

// Verifica la restriccion de la tarea: el tensor puede tener como maximo 3 dimensiones.
void Tensor::check_dimension_limit(
    const vector<size_t>& shape) const {

    if (shape.empty() || shape.size() > 3) {
        throw invalid_argument(
            "Un Tensor debe tener entre 1 y 3 dimensiones.");
    }
}

// Reserva memoria dinamica usando double* e inicializa todos los elementos en cero.
void Tensor::allocate_and_zero() {
    const size_t n = numel();

    data_ = new double[n];
    ref_count_ = new size_t(1);

    fill(data_, data_ + n, 0.0);
}

// Libera la memoria dinamica cuando ningun tensor comparte los datos.
void Tensor::release() {
    if (ref_count_ != nullptr) {
        --(*ref_count_);

        if (*ref_count_ == 0) {
            delete[] data_;
            delete ref_count_;
        }
    }

    data_ = nullptr;
    ref_count_ = nullptr;
}

// Constructor principal: recibe shape y values y almacena los datos en memoria dinamica.
Tensor::Tensor(const vector<size_t>& shape,
               const vector<double>& values)
    : data_(nullptr), ref_count_(nullptr), shape_(shape) {

    check_dimension_limit(shape_);

    // Verifica que la cantidad de values coincida con el producto de las dimensiones.
    if (values.size() != product(shape_)) {
        throw invalid_argument(
            "La cantidad de valores no coincide con el tamano del tensor.");
    }

    data_ = new double[values.size()];
    ref_count_ = new size_t(1);

    copy(values.begin(), values.end(), data_);
}

// Constructor auxiliar para crear un tensor inicializado en cero.
Tensor::Tensor(const vector<size_t>& shape)
    : data_(nullptr), ref_count_(nullptr), shape_(shape) {

    check_dimension_limit(shape_);
    allocate_and_zero();
}

// Crea un tensor con todos sus elementos iguales a cero.
Tensor Tensor::zeros(const vector<size_t>& shape) {
    return Tensor(shape);
}

// Crea un tensor con todos sus elementos iguales a uno.
Tensor Tensor::ones(const vector<size_t>& shape) {
    Tensor result(shape);

    fill(result.data_, result.data_ + result.numel(), 1.0);

    return result;
}

// Crea un tensor con valores aleatorios uniformemente distribuidos en [min, max).
Tensor Tensor::random(const vector<size_t>& shape,
                      double min, double max) {

    if (min >= max) {
        throw invalid_argument(
            "En random(), min debe ser menor que max.");
    }

    Tensor result(shape);

    random_device rd;
    mt19937 generator(rd());
    uniform_real_distribution<double> distribution(min, max);

    for (size_t i = 0; i < result.numel(); ++i) {
        result.data_[i] = distribution(generator);
    }

    return result;
}

// Crea un tensor 1D con valores secuenciales desde start hasta antes de end.
Tensor Tensor::arange(double start, double end) {
    if (start >= end) {
        throw invalid_argument(
            "arange() requiere start < end.");
    }

    size_t n = static_cast<size_t>(
        ceil(end - start));

    while (n > 0 && start + static_cast<double>(n - 1) >= end) {
        --n;
    }

    if (n == 0) {
        throw invalid_argument(
            "arange() no puede crear un tensor vacio.");
    }

    Tensor result({n});

    for (size_t i = 0; i < n; ++i) {
        result.data_[i] = start + static_cast<double>(i);
    }

    return result;
}

// Constructor de copia: realiza una copia profunda de los datos del tensor original.
Tensor::Tensor(const Tensor& other)
    : data_(nullptr), ref_count_(nullptr), shape_(other.shape_) {

    const size_t n = other.numel();

    data_ = new double[n];
    ref_count_ = new size_t(1);

    copy(other.data_, other.data_ + n, data_);
}

// Constructor de movimiento: transfiere la propiedad de la memoria al nuevo tensor.
Tensor::Tensor(Tensor&& other) noexcept
    : data_(other.data_),
      ref_count_(other.ref_count_),
      shape_(move(other.shape_)) {

    other.data_ = nullptr;
    other.ref_count_ = nullptr;
    other.shape_.clear();
}

// Asignacion por copia: libera el estado anterior mediante una copia segura y copia profunda.
Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) {
        return *this;
    }

    Tensor copy(other);
    swap(data_, copy.data_);
    swap(ref_count_, copy.ref_count_);
    swap(shape_, copy.shape_);

    return *this;
}

// Asignacion por movimiento: libera los recursos actuales y toma la memoria del temporal.
Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    release();

    data_ = other.data_;
    ref_count_ = other.ref_count_;
    shape_ = move(other.shape_);

    other.data_ = nullptr;
    other.ref_count_ = nullptr;
    other.shape_.clear();

    return *this;
}

// Destructor: libera la memoria dinamica asociada al tensor.
Tensor::~Tensor() {
    release();
}

const vector<size_t>& Tensor::shape() const {
    return shape_;
}

size_t Tensor::size() const {
    return numel();
}

size_t Tensor::ndim() const {
    return shape_.size();
}

double& Tensor::operator[](size_t index) {
    if (index >= numel()) {
        throw out_of_range("Indice fuera de rango.");
    }

    return data_[index];
}

const double& Tensor::operator[](size_t index) const {
    if (index >= numel()) {
        throw out_of_range("Indice fuera de rango.");
    }

    return data_[index];
}

double& Tensor::at(size_t i) {
    if (shape_.size() != 1) {
        throw invalid_argument("at(i) requiere un tensor 1D.");
    }

    return (*this)[i];
}

const double& Tensor::at(size_t i) const {
    if (shape_.size() != 1) {
        throw invalid_argument("at(i) requiere un tensor 1D.");
    }

    return (*this)[i];
}

double& Tensor::at(size_t i, size_t j) {
    if (shape_.size() != 2) {
        throw invalid_argument("at(i,j) requiere un tensor 2D.");
    }

    if (i >= shape_[0] || j >= shape_[1]) {
        throw out_of_range("Indice fuera de rango.");
    }

    return data_[i * shape_[1] + j];
}

const double& Tensor::at(size_t i, size_t j) const {
    if (shape_.size() != 2) {
        throw invalid_argument("at(i,j) requiere un tensor 2D.");
    }

    if (i >= shape_[0] || j >= shape_[1]) {
        throw out_of_range("Indice fuera de rango.");
    }

    return data_[i * shape_[1] + j];
}

double& Tensor::at(size_t i, size_t j, size_t k) {
    if (shape_.size() != 3) {
        throw invalid_argument("at(i,j,k) requiere un tensor 3D.");
    }

    if (i >= shape_[0] || j >= shape_[1] || k >= shape_[2]) {
        throw out_of_range("Indice fuera de rango.");
    }

    return data_[
        i * shape_[1] * shape_[2]
        + j * shape_[2]
        + k
    ];
}

const double& Tensor::at(size_t i,
                         size_t j,
                         size_t k) const {
    if (shape_.size() != 3) {
        throw invalid_argument("at(i,j,k) requiere un tensor 3D.");
    }

    if (i >= shape_[0] || j >= shape_[1] || k >= shape_[2]) {
        throw out_of_range("Indice fuera de rango.");
    }

    return data_[
        i * shape_[1] * shape_[2]
        + j * shape_[2]
        + k
    ];
}

// view cambia el shape sin copiar los datos; los tensores resultante y original comparten memoria.
Tensor Tensor::view(const vector<size_t>& new_shape) const {
    check_dimension_limit(new_shape);

    if (product(new_shape) != numel()) {
        throw invalid_argument(
            "view() debe conservar la cantidad total de elementos.");
    }

    Tensor result(new_shape);
    result.release();

    result.data_ = data_;
    result.ref_count_ = ref_count_;

    ++(*ref_count_);

    return result;
}

// unsqueeze inserta una dimension de tamano 1 sin copiar los datos.
Tensor Tensor::unsqueeze(size_t dim) const {
    // La tarea permite un maximo de 3 dimensiones.
    if (shape_.size() >= 3) {
        throw invalid_argument(
            "unsqueeze() no puede superar 3 dimensiones.");
    }

    if (dim > shape_.size()) {
        throw out_of_range(
            "La dimension indicada para unsqueeze() no existe.");
    }

    vector<size_t> new_shape = shape_;
    new_shape.insert(new_shape.begin() + static_cast<ptrdiff_t>(dim), 1);

    return view(new_shape);
}

// concat une tensores en una dimension y reserva memoria nueva para el resultado.
Tensor Tensor::concat(const vector<Tensor>& tensors,
                      size_t dim) {

    if (tensors.empty()) {
        throw invalid_argument(
            "concat() requiere al menos un tensor.");
    }

    const size_t dimensions = tensors[0].ndim();

    if (dim >= dimensions) {
        throw out_of_range(
            "La dimension de concatenacion no existe.");
    }

    for (const Tensor& tensor : tensors) {
        if (tensor.ndim() != dimensions) {
            throw invalid_argument(
                "Todos los tensores deben tener el mismo numero de dimensiones.");
        }
    }

    vector<size_t> result_shape = tensors[0].shape();

    for (size_t i = 1; i < tensors.size(); ++i) {
        for (size_t d = 0; d < dimensions; ++d) {
            if (d != dim &&
                tensors[i].shape()[d] != result_shape[d]) {
                throw invalid_argument(
                    "Dimensiones incompatibles para concat().");
            }
        }

        result_shape[dim] += tensors[i].shape()[dim];
    }

    Tensor result(result_shape);

    const size_t outer =
        [&]() {
            size_t p = 1;
            for (size_t d = 0; d < dim; ++d) {
                p *= result_shape[d];
            }
            return p;
        }();

    const size_t inner =
        [&]() {
            size_t p = 1;
            for (size_t d = dim + 1; d < dimensions; ++d) {
                p *= result_shape[d];
            }
            return p;
        }();

    size_t result_offset = 0;

    for (size_t outer_i = 0; outer_i < outer; ++outer_i) {
        for (const Tensor& tensor : tensors) {
            const size_t block =
                tensor.shape()[dim] * inner;

            const size_t source_offset =
                outer_i * block;

            copy(
                tensor.data_ + source_offset,
                tensor.data_ + source_offset + block,
                result.data_ + result_offset
            );

            result_offset += block;
        }
    }

    return result;
}

// Suma elemento a elemento y tambien permite el broadcasting necesario para los bias de la red.
Tensor Tensor::operator+(const Tensor& other) const {

    if (shape_ == other.shape_) {
        Tensor result(shape_);

        for (size_t i = 0; i < numel(); ++i) {
            result.data_[i] = data_[i] + other.data_[i];
        }

        return result;
    }

    if (shape_.size() == 2 &&
        other.shape_.size() == 2 &&
        other.shape_[0] == 1 &&
        other.shape_[1] == shape_[1]) {

        Tensor result(shape_);

        for (size_t i = 0; i < shape_[0]; ++i) {
            for (size_t j = 0; j < shape_[1]; ++j) {
                result.at(i, j) =
                    at(i, j) + other.at(0, j);
            }
        }

        return result;
    }

    if (shape_.size() == 2 &&
        other.shape_.size() == 2 &&
        shape_[0] == 1 &&
        shape_[1] == other.shape_[1]) {

        return other + *this;
    }

    throw invalid_argument(
        "Dimensiones incompatibles para operator+.");
}

// Resta elemento a elemento verificando que las dimensiones sean compatibles.
Tensor Tensor::operator-(const Tensor& other) const {
    if (shape_ == other.shape_) {
        Tensor result(shape_);

        for (size_t i = 0; i < numel(); ++i) {
            result.data_[i] = data_[i] - other.data_[i];
        }

        return result;
    }

    if (shape_.size() == 2 &&
        other.shape_.size() == 2 &&
        other.shape_[0] == 1 &&
        other.shape_[1] == shape_[1]) {

        Tensor result(shape_);

        for (size_t i = 0; i < shape_[0]; ++i) {
            for (size_t j = 0; j < shape_[1]; ++j) {
                result.at(i, j) =
                    at(i, j) - other.at(0, j);
            }
        }

        return result;
    }

    throw invalid_argument(
        "Dimensiones incompatibles para operator-.");
}

// Multiplicacion elemento a elemento entre tensores de la misma forma.
Tensor Tensor::operator*(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw invalid_argument(
            "Dimensiones incompatibles para multiplicacion elementwise.");
    }

    Tensor result(shape_);

    for (size_t i = 0; i < numel(); ++i) {
        result.data_[i] = data_[i] * other.data_[i];
    }

    return result;
}

// Multiplicacion de todos los elementos del tensor por un escalar.
Tensor Tensor::operator*(double scalar) const {
    Tensor result(shape_);

    for (size_t i = 0; i < numel(); ++i) {
        result.data_[i] = data_[i] * scalar;
    }

    return result;
}

Tensor operator*(double scalar, const Tensor& tensor) {
    return tensor * scalar;
}

// Funcion de activacion ReLU: reemplaza los valores negativos por cero.
Tensor Tensor::relu() const {
    Tensor result(shape_);

    for (size_t i = 0; i < numel(); ++i) {
        result.data_[i] = max(0.0, data_[i]);
    }

    return result;
}

// Funcion de activacion Sigmoid utilizada en la salida de la red neuronal.
Tensor Tensor::sigmoid() const {
    Tensor result(shape_);

    for (size_t i = 0; i < numel(); ++i) {

        if (data_[i] >= 0.0) {
            result.data_[i] =
                1.0 / (1.0 + exp(-data_[i]));
        } else {
            const double e = exp(data_[i]);
            result.data_[i] = e / (1.0 + e);
        }
    }

    return result;
}

void Tensor::print_shape(ostream& os) const {
    os << "{";

    for (size_t i = 0; i < shape_.size(); ++i) {
        os << shape_[i];

        if (i + 1 < shape_.size()) {
            os << ", ";
        }
    }

    os << "}";
}

void Tensor::print(ostream& os) const {
    os << "shape = ";
    print_shape(os);
    os << "\n";

    os << fixed << setprecision(4);

    if (shape_.size() == 1) {
        os << "[";

        for (size_t i = 0; i < shape_[0]; ++i) {
            os << data_[i];

            if (i + 1 < shape_[0]) {
                os << ", ";
            }
        }

        os << "]\n";
        return;
    }

    if (shape_.size() == 2) {
        for (size_t i = 0; i < shape_[0]; ++i) {
            os << "[";

            for (size_t j = 0; j < shape_[1]; ++j) {
                os << data_[i * shape_[1] + j];

                if (j + 1 < shape_[1]) {
                    os << ", ";
                }
            }

            os << "]\n";
        }

        return;
    }

    for (size_t i = 0; i < shape_[0]; ++i) {
        os << "Bloque " << i << ":\n";

        for (size_t j = 0; j < shape_[1]; ++j) {
            os << "[";

            for (size_t k = 0; k < shape_[2]; ++k) {
                os << at(i, j, k);

                if (k + 1 < shape_[2]) {
                    os << ", ";
                }
            }

            os << "]\n";
        }
    }
}

// Calcula el producto punto de dos tensores con igual cantidad de elementos.
Tensor dot(const Tensor& a, const Tensor& b) {
    if (a.size() != b.size()) {
        throw invalid_argument(
            "dot() requiere tensores con igual cantidad de elementos.");
    }

    double sum = 0.0;

    for (size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }

    return Tensor({1}, {sum});
}

// Realiza multiplicacion matricial entre dos tensores bidimensionales compatibles.
Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.ndim() != 2 || b.ndim() != 2) {
        throw invalid_argument(
            "matmul() requiere tensores bidimensionales.");
    }

    const size_t m = a.shape()[0];
    const size_t k = a.shape()[1];

    const size_t kb = b.shape()[0];
    const size_t n = b.shape()[1];

    if (k != kb) {
        throw invalid_argument(
            "Dimensiones incompatibles para multiplicacion matricial.");
    }

    Tensor result({m, n});

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;

            for (size_t p = 0; p < k; ++p) {
                sum += a.at(i, p) * b.at(p, j);
            }

            result.at(i, j) = sum;
        }
    }

    return result;
}