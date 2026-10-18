# Informe del Trabajo Parcial - [Computación Distribuida y Paralela]

## 1. Nombre de los Integrantes / Fecha

**Integrantes**:

* **Sanz Palomino Facundo Alejandro**
* **Johan Marquez Zuñiga**
* **Anthony Stefano Cervantes Cohaila**
* **Gabriel Matias Jara Zapana**
* **Fecha**: [16/10/2026]


## 1. Marco Teorico

### ¿De que trata el proyecto?

* La orientacion del proyecto va a la resolución de problemas de búsqueda adversarial mediante inteligencia artificial. En este caso, se implementa el algoritmo **Minimax con Poda Alfa-Beta** para el juego de 3 en raya (Tic-Tac-Toe). Para lograr una ejecución óptima y cumplir con los estándares de **Computación Distribuida y Paralela**, se utiliza la API **OpenMP** en C++, lo que permite paralelizar el cálculo del árbol de juego distribuyendo las distintas ramas de decisión entre los múltiples núcleos del procesador (arquitectura Apple Silicon M2). Esto reduce sustancialmente el tiempo de procesamiento en la toma de decisiones de la IA.

### Caracteristicas clave del proyecto

* **Exploración de árbol de juego con Minimax y Poda Alfa-Beta**
* **Paralelización de hilos en memoria compartida usando OpenMP**
* **Gestión dinámica de la carga de trabajo (`schedule(dynamic)`)**
* **Prevención de condiciones de carrera mediante bloques críticos (`omp critical`)**
* **Benchmark integrado para comparar el rendimiento Secuencial vs Paralelo**

---

## 2. Explicación del Avance del proyecto.

El primer enfoque del proyecto fue el migrar la lógica original de Python a **C++**, ya que Python posee un bloqueo global (GIL) que impide el verdadero paralelismo en CPU. La idea central que surgió fue la de paralelizar únicamente el **primer nivel del árbol de juego** utilizando *OpenMP*. Al hacer esto, cada hilo de la CPU toma una jugada inicial disponible diferente y explora toda la profundidad de esa rama de forma concurrente, logrando un tiempo de evaluación mucho más rápido (**O(b^d/núcleos)**) sin generar una sobrecarga excesiva por crear demasiados hilos anidados.

---

## 3. Implementación de OpenMP y el algoritmo Minimax.

* En los siguientes segmentos de código veremos la implementación principal de la paralelización y la protección de los datos en memoria compartida:

**Paralelización de la mejor jugada (OpenMP)**

```cpp
// Este segmento de codigo sacado de "main.cpp"
int mejor_jugada_paralela(const vector<char>& tablero, int& total_nodos_explorados) {
    // ... (código de inicialización) ...

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < disponibles.size(); i++) {
        int jugada = disponibles[i];
        
        // Copia local para aislar la memoria de cada hilo
        vector<char> tablero_local = tablero; 
        tablero_local[jugada] = 'O';
        
        int nodos_locales = 0;
        int valor = minimax_ab(tablero_local, 0, -1000, 1000, false, nodos_locales);

        // Bloque crítico para sincronización de variables globales
        #pragma omp critical 
        {
            total_nodos += nodos_locales;
            if (valor > mejor_valor) {
                mejor_valor = valor;
                mejor_jugada = jugada;
            }
        }
    }
    return mejor_jugada;
}

```

* En este segmento de código se inicializa el grupo de hilos con la directiva `#pragma omp parallel for`. Se utiliza `schedule(dynamic)` para repartir la carga equitativamente si una rama del árbol se poda más rápido que otra. Además, se crea un `tablero_local` para que los hilos no sobrescriban la memoria de los demás (aislamiento), y finalmente se usa `#pragma omp critical` para que la actualización de la variable `mejor_valor` se haga de forma segura, evitando *condiciones de carrera*.

**Lógica recursiva de Minimax (Secuencial por hilo)**

```cpp
// Este segmento de codigo sacado de "main.cpp" - función minimax_ab
    if (es_max) {
        int mejor = numeric_limits<int>::min();
        for (int i = 0; i < 9; i++) {
            if (tablero[i] == ' ') {
                tablero[i] = 'O';
                mejor = max(mejor, minimax_ab(tablero, profundidad + 1, alfa, beta, false, nodos_locales));
                tablero[i] = ' '; // Backtracking
                alfa = max(alfa, mejor);
                if (beta <= alfa) break; // Poda Alfa-Beta
            }
        }
        return mejor;
    }

```

* Este fragmento muestra cómo cada hilo recorre su sub-rama asignada. Aplica el algoritmo Minimax verificando la utilidad del tablero. Lo más importante aquí es la condición `if (beta <= alfa) break;`, la cual detiene la exploración de ramas inútiles (poda), haciendo que la IA sea extremadamente eficiente sin cambiar el resultado óptimo.

---

## 6. Output/Consola.

### Benchmark de Rendimiento **Terminal de macOS**

* Al ejecutar el código, lo primero que nos muestra es la prueba de rendimiento procesando el tablero 10,000 veces:

```py
Realizando Benchmark (10000 evaluaciones del árbol completo)...
Tiempo Secuencial: 185 ms
Tiempo Paralelo (OpenMP): 42 ms
Aceleracion: 4.4047x

--- INICIANDO JUEGO CONTRA LA IA (PARALELA) ---
Posiciones: 0 a 8

```

* **Comentario**: Como podemos ver, la implementación paralela con OpenMP demuestra una reducción sustancial en el tiempo de ejecución. La aceleración (Speedup) evidencia que el sistema está utilizando eficientemente los múltiples núcleos del chip M2 de Mac, cumpliendo con el criterio principal de la rúbrica del curso.

### Juego interactivo contra la IA **Terminal de macOS**

* Al comenzar el juego e ingresar una jugada, nos muestra lo siguiente:

```py
   |   |   
-----------
   |   |   
-----------
   |   |   

Tu turno (X) - Ingresa posicion: 4
La IA (O) esta calculando con OpenMP...
La IA jugo en la posicion 0 (Nodos explorados: 58)

 O |   |   
-----------
   | X |   
-----------
   |   |   

Tu turno (X) - Ingresa posicion: 

```

* **Comentario**: El sistema permite una interacción directa. La IA recibe el estado del tablero, ejecuta la distribución de hilos mediante OpenMP para evaluar los movimientos restantes y devuelve la jugada óptima (en este caso, la esquina 0) junto con la cantidad de nodos que tuvo que explorar, demostrando la correctitud funcional del programa en tiempo real.