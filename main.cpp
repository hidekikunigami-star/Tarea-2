#include "Tensor.h"

#include <iostream>
#include <iomanip>
#include <exception>

using namespace std;
// Función auxiliar: muestra el tamaño y la forma de un tensor

void mostrar_etapa(const char* nombre, const Tensor& tensor) {
    cout << left << setw(28)
         << nombre << " size = " << tensor.size()
         << " shape = ";

    tensor.print_shape(cout);// imprime la forma del tensor
    cout << endl;
}

void pruebas_basicas() {
    cout << "***************PRUEBAS BASICAS **************\n\n";
    
    // Creación de tensores usando métodos estáticos (estilo NumPy)
    Tensor A = Tensor::zeros({2, 3});    //tensor 2x3 lleno de ceros
    Tensor B = Tensor::ones({2, 3});     //tensor 2x3 lleno de unos
    Tensor C = Tensor::random({2, 2}, 0.0, 1.0); //tensor 2x2 con valores aleatorios
    Tensor D = Tensor::arange(0, 6);     // tensor con valores [0,1,2,3,4]
    
    //muestra cada tensor creado
    cout << "A = zeros({2,3})\n";
    A.print(cout);

    cout << "\nB = ones({2,3})\n";
    B.print(cout);

    cout << "\nC = random({2,2}, 0, 1)\n";
    C.print(cout);

    cout << "\nD = arange(0,6)\n";
    D.print(cout);
    
    // Operaciones básicas con tensores
    Tensor suma = A + B;
    Tensor resta = B - A;
    Tensor producto = A * B;
    Tensor escalar = B * 2.0;
     
    // Muestra resultados de las operaciones
    cout << "\nA + B:\n";
    suma.print(cout);

    cout << "\nB - A:\n";
    resta.print(cout);

    cout << "\nA * B elementwise:\n";
    producto.print(cout);

    cout << "\nB * 2.0:\n";
    escalar.print(cout);

    cout << "\n*************COPIA Y MOVIMIENTO ***************\n\n";

    Tensor original = Tensor::arange(0, 6); // tensor [0,1,2,3,4,5]
    Tensor copia(original);  // constructor de copia (deep copy)

    copia[0] = 100; // modificar la copia no afecta al original

    cout << "Original despues de modificar copia:\n"; 
    original.print(cout);
    
    cout << "Copia:\n";
    copia.print(cout);
    
    // Constructor de movimiento: transfiere la propiedad de los dato
    Tensor movido(move(copia));

    cout << "\nTensor movido:\n";
    movido.print(cout);

    cout << "\n***********VIEW Y UNSQUEEZE *********\n\n";

    Tensor V = Tensor::arange(0, 12); // tensor 1D de 12 elementos
   
    // view: cambia la forma sin copiar datos (comparte memoria)
    Tensor V2 = V.view({3, 4});  // ahora es 3x4
    Tensor U0 = V.unsqueeze(0);  // agrega dimensión al inicio: shape {1,12}
    Tensor U1 = V.unsqueeze(1);  // agrega dimensión en la posición 1: shape {12,1}
   
    // Muestra formas resultantes
    mostrar_etapa("V", V);
    mostrar_etapa("V.view({3,4})", V2);
    mostrar_etapa("V.unsqueeze(0)", U0);
    mostrar_etapa("V.unsqueeze(1)", U1);

    // Verificar que view comparte memoria con el original
    V2.at(0, 0) = 999; // modificar V2 modifica V (porque comparten datos)

    cout << "\nV[0] despues de cambiar V2[0,0] = 999: "
         << V[0] << endl; // debe mostrar 999

    cout << "\n*********** CONCAT **********\n\n";

    Tensor C1 = Tensor::ones({2, 3}); // tensor 2x3 de unos
    Tensor C2 = Tensor::zeros({2, 3}); // tensor 2x3 de ceros
    
    // Concatenar a lo largo de la dimensión 0 (filas): resultado 4x3
    Tensor C0 = Tensor::concat({C1, C2}, 0);
    // Concatenar a lo largo de la dimensión 1 (columnas): resultado 2x6
    Tensor Cdim1 = Tensor::concat({C1, C2}, 1);

    mostrar_etapa("concat dim 0", C0);
    mostrar_etapa("concat dim 1", Cdim1);

    cout << "\n*********** DOT Y MATMUL **************\n\n";

    Tensor x = Tensor::arange(1, 4); // [1, 2, 3]
    Tensor y = Tensor::arange(4, 7); // [4, 5, 6]
    
    // Producto punto: 1*4 + 2*5 + 3*6 = 32
    Tensor producto_punto = dot(x, y);

    cout << "dot([1,2,3], [4,5,6]) = ";
    producto_punto.print(cout);
    
    // Multiplicación matricial: M1 (2x3) * M2 (3x2) = (2x2)
    Tensor M1({2, 3}, {
        1, 2, 3,
        4, 5, 6
    });

    Tensor M2({3, 2}, {
        1, 2,
        3, 4,
        5, 6
    });

    Tensor MM = matmul(M1, M2);

    cout << "\nM1 * M2:\n";
    MM.print(cout);
}

void red_neuronal() {
    cout << "\n\n********** RED NEURONAL ***********\n\n";
    
    // Tensor de entrada: 1000 muestras, cada una de 20x20
    Tensor input = Tensor::random({1000, 20, 20}, 0.0, 1.0);
    mostrar_etapa("1. Entrada", input);
   
    // Aplanar cada muestra: 1000x400 (cada imagen 20x20 = 400)
    Tensor X = input.view({1000, 400});
    mostrar_etapa("2. View", X);
   
    //Capa lineal 1: W1 de 400x100
    Tensor W1 = Tensor::random({400, 100}, -0.1, 0.1);
    Tensor Z1 = matmul(X, W1);
    mostrar_etapa("3. Matmul W1", Z1);
    //Sumar bias b1 (1x100) (broadcasting)
    Tensor b1 = Tensor::random({1, 100}, -0.1, 0.1);
    Tensor Z1_bias = Z1 + b1;
    mostrar_etapa("4. Suma bias b1", Z1_bias);
    
    //Activación ReLU (max(0, x))
    Tensor H = Z1_bias.relu();
    mostrar_etapa("5. ReLU", H);
    
    //Capa lineal 2: W2 de 100x10
    Tensor W2 = Tensor::random({100, 10}, -0.1, 0.1);
    Tensor Z2 = matmul(H, W2);
    mostrar_etapa("6. Matmul W2", Z2);
    
    //Sumar bias b2 (1x10)
    Tensor b2 = Tensor::random({1, 10}, -0.1, 0.1);
    Tensor Z2_bias = Z2 + b2;
    mostrar_etapa("7. Suma bias b2", Z2_bias);
    
    //Activación Sigmoid (1 / (1 + e^(-x)))
    Tensor output = Z2_bias.sigmoid();
    mostrar_etapa("8. Sigmoid", output);

    cout << "\nPrimeros 10 valores de la primera muestra:\n";

    for (size_t j = 0; j < 10; ++j) {
        cout << fixed << setprecision(6)
             << output.at(0, j) << " ";
    }

    cout << "\n";
}

int main() {
    try {
        // Ejecuta todas las pruebas
        pruebas_basicas();
        red_neuronal();

        cout << "\nPrograma ejecutado correctamente.\n";
    }
    catch (const exception& e) {
        // Capturar cualquier excepción y mostrar el error
        cerr << "\nERROR: " << e.what() << endl;
        return 1;
    }

    return 0;
}
