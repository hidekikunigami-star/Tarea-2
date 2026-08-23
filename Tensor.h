#ifndef TENSOR_H
#define TENSOR_H
//CARGA DE LIBERIAS
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
    void release(); //FUNCION AUXOILIAR PARA LIBERAR MEMORIA
    void allocate_and_zero(); //FUNCION AUXILIAR PARA LIBERAR MEMORIA
    void check_dimension_limit(const vector<size_t>& shape) const;
    static size_t product(const vector<size_t>& shape); 

public:
//CREACION DEL CONSTRUCTOR PRINCIPAL:
    Tensor(const vector<size_t>& shape,
           const vector<double>& values);

    explicit Tensor(const vector<size_t>& shape); //CREA UN TENSOR DE CEROS DE FORMA EXPLICITA

    Tensor(const Tensor& other); //CONSTRUCTOR DE COPIA

    Tensor(Tensor&& other) noexcept; //CONSTRUCTOR DE MOVIMIENTO

    Tensor& operator=(const Tensor& other); //ASIGNADOR DE COPIA

    Tensor& operator=(Tensor&& other) noexcept; //ASIGNADOR DE MOVIMIENTO

    ~Tensor(); //DESTRUCTOR
//METODOS ESTATICO PARA FACILITAR LA CREACION DE TENSORES
    static Tensor zeros(const vector<size_t>& shape);
    static Tensor ones(const vector<size_t>& shape);
    static Tensor random(const vector<size_t>& shape,
                         double min, double max);
    static Tensor arange(double start, double end);

    const vector<size_t>& shape() const; //RETORNA LAS DIMENSIONES
    size_t size() const;   //RETORNA NUMERO TOTAL
    size_t ndim() const;   //RETORNA NUMERO DE DIMENSIONES
//ACCESO POR CORDENADAS
    double& operator[](size_t index);
    const double& operator[](size_t index) const;

    double& at(size_t i);
    const double& at(size_t i) const;

    double& at(size_t i, size_t j);
    const double& at(size_t i, size_t j) const;

    double& at(size_t i, size_t j, size_t k);
    const double& at(size_t i, size_t j, size_t k) const;
//FIN DEL ACCESO POR COORDENATAS
    Tensor view(const vector<size_t>& new_shape) const; //REINTERPRETA LA FORMA DEL TENSOR SIN COPIAR LOS DATOS SUBYACENTES

    Tensor unsqueeze(size_t dim) const; //AGREGA UNA DIMENSION DE TAMAÑO 1 SIN MODIFICAR LOS DATOS

    static Tensor concat(const vector<Tensor>& tensors,
                         size_t dim); //UNE MULTIPLES TENSORES A LO LARGO DE UNA DIMENSION
//SOBRECARGA DE OPERADORES
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator*(double scalar) const;

    friend Tensor operator*(double scalar, const Tensor& tensor);
//FUNCIONES DE ACTIVACION
    Tensor relu() const;
    Tensor sigmoid() const;
//METODOS PARA EL PRINT
    void print_shape(ostream& os) const;
    void print(ostream& os) const;
};

Tensor dot(const Tensor& a, const Tensor& b); //PRODUCTO ENTRE TENSORES
Tensor matmul(const Tensor& a, const Tensor& b); //MULTIPLICACION MATRIZIAL BIDIMENSIONAL

#endif
