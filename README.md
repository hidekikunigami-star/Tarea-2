# Tensor++

Implementacion en C++ de la tarea de Programacion III.

## Archivos

- `Tensor.h`: declaracion de la clase `Tensor` y funciones amigas.
- `Tensor.cpp`: implementacion de la clase.
- `main.cpp`: pruebas de las funcionalidades y red neuronal solicitada.

## Compilacion

Con g++:

```bash
g++ -std=c++17 -Wall -Wextra -pedantic Tensor.cpp main.cpp -o tensor_app
```

## Ejecucion

Linux/macOS:

```bash
./tensor_app
```

Windows:

```bash
tensor_app.exe
```

## Funcionalidades

- Tensores 1D, 2D y 3D.
- Memoria dinamica con `double*`.
- Deep copy.
- Move constructor y move assignment.
- `zeros`, `ones`, `random`, `arange`.
- Operadores `+`, `-`, `*` y multiplicacion por escalar.
- `view` sin copiar los datos.
- `unsqueeze` sin copiar los datos.
- `concat`.
- `dot`.
- `matmul`.
- ReLU y Sigmoid para la aplicacion de red neuronal.
- Broadcasting para sumar los biases `1 x 100` y `1 x 10` a las matrices de lotes.

## Nota

La tarea exige que los datos se almacenen mediante memoria dinamica. Esta implementacion usa `double*` y un contador de referencias manual para permitir que `view` y `unsqueeze` compartan los mismos datos sin realizar una copia ni provocar doble liberacion de memoria.
