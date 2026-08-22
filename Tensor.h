#ifndef TENSOR_H
#define TENSOR_H

#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <vector>
#include <ostream>

class Tensor {
private:
    double* data_;
    std::size_t* ref_count_;
    std::vector<std::size_t> shape_;

    std::size_t numel() const;
    void release();
    void allocate_and_zero();
    void check_dimension_limit(const std::vector<std::size_t>& shape) const;
    static std::size_t product(const std::vector<std::size_t>& shape);

public:
    // Constructor principal
    Tensor(const std::vector<std::size_t>& shape,
           const std::vector<double>& values);

    // Constructor auxiliar para crear un tensor de ceros.
    explicit Tensor(const std::vector<std::size_t>& shape);

    // Constructor de copia: deep copy.
    Tensor(const Tensor& other);

    // Constructor de movimiento.
    Tensor(Tensor&& other) noexcept;

    // Asignacion por copia.
    Tensor& operator=(const Tensor& other);

    // Asignacion por movimiento.
    Tensor& operator=(Tensor&& other) noexcept;

    // Destructor.
    ~Tensor();

    // Creadores predefinidos.
    static Tensor zeros(const std::vector<std::size_t>& shape);
    static Tensor ones(const std::vector<std::size_t>& shape);
    static Tensor random(const std::vector<std::size_t>& shape,
                         double min, double max);
    static Tensor arange(double start, double end);

    // Informacion.
    const std::vector<std::size_t>& shape() const;
    std::size_t size() const;
    std::size_t ndim() const;

    // Acceso lineal.
    double& operator[](std::size_t index);
    const double& operator[](std::size_t index) const;

    // Acceso 1D / 2D / 3D.
    double& at(std::size_t i);
    const double& at(std::size_t i) const;

    double& at(std::size_t i, std::size_t j);
    const double& at(std::size_t i, std::size_t j) const;

    double& at(std::size_t i, std::size_t j, std::size_t k);
    const double& at(std::size_t i, std::size_t j, std::size_t k) const;

    // Reinterpretacion de forma sin copiar datos.
    Tensor view(const std::vector<std::size_t>& new_shape) const;

    // Agrega una dimension de tamano 1 sin copiar datos.
    Tensor unsqueeze(std::size_t dim) const;

    // Concatenacion.
    static Tensor concat(const std::vector<Tensor>& tensors,
                         std::size_t dim);

    // Operadores elementwise.
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator*(double scalar) const;

    friend Tensor operator*(double scalar, const Tensor& tensor);

    // Funciones de activacion utiles para la red neuronal.
    Tensor relu() const;
    Tensor sigmoid() const;

    // Imprimir informacion y datos.
    void print_shape(std::ostream& os) const;
    void print(std::ostream& os) const;
};

// Funciones amigas solicitadas.
Tensor dot(const Tensor& a, const Tensor& b);
Tensor matmul(const Tensor& a, const Tensor& b);

#endif
