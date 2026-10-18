#include <chrono>
#include <iostream>
#include <limits>
#include <omp.h>
#include <vector>

using namespace std;
using namespace std::chrono;

// Imprime el tablero
void imprimir_tablero(const vector<char> &tablero) {
  cout << "\n";
  for (int i = 0; i < 9; i += 3) {
    cout << " " << tablero[i] << " | " << tablero[i + 1] << " | "
         << tablero[i + 2] << " \n";
    if (i < 6)
      cout << "-----------\n";
  }
  cout << "\n";
}

char verificar_ganador(const vector<char> &tablero) {
  int lineas[8][3] = {
      {0, 1, 2}, {3, 4, 5}, {6, 7, 8}, // Filas
      {0, 3, 6}, {1, 4, 7}, {2, 5, 8}, // Columnas
      {0, 4, 8}, {2, 4, 6}             // Diagonales
  };
  for (auto &l : lineas) {
    if (tablero[l[0]] != ' ' && tablero[l[0]] == tablero[l[1]] &&
        tablero[l[0]] == tablero[l[2]]) {
      return tablero[l[0]];
    }
  }
  for (char c : tablero) {
    if (c == ' ')
      return 'N'; // Juego en progreso
  }
  return 'T'; // Empate
}

// Algoritmo Minimax con Poda Alfa-Beta (Secuencial para explorar las sub-ramas)
int minimax_ab(vector<char> &tablero, int profundidad, int alfa, int beta,
               bool es_max, int &nodos_locales) {
  nodos_locales++;
  char resultado = verificar_ganador(tablero);
  if (resultado == 'O')
    return 10 - profundidad; // Prefiere ganar rápido
  if (resultado == 'X')
    return profundidad - 10;
  if (resultado == 'T')
    return 0;

  if (es_max) {
    int mejor = numeric_limits<int>::min();
    for (int i = 0; i < 9; i++) {
      if (tablero[i] == ' ') {
        tablero[i] = 'O';
        mejor = max(mejor, minimax_ab(tablero, profundidad + 1, alfa, beta,
                                      false, nodos_locales));
        tablero[i] = ' ';
        alfa = max(alfa, mejor);
        if (beta <= alfa)
          break; // Poda
      }
    }
    return mejor;
  } else {
    int mejor = numeric_limits<int>::max();
    for (int i = 0; i < 9; i++) {
      if (tablero[i] == ' ') {
        tablero[i] = 'X';
        mejor = min(mejor, minimax_ab(tablero, profundidad + 1, alfa, beta,
                                      true, nodos_locales));
        tablero[i] = ' ';
        beta = min(beta, mejor);
        if (beta <= alfa)
          break; // Poda
      }
    }
    return mejor;
  }
}

int mejor_jugada_paralela(const vector<char> &tablero,
                          int &total_nodos_explorados) {
  vector<int> disponibles;
  for (int i = 0; i < 9; i++) {
    if (tablero[i] == ' ')
      disponibles.push_back(i);
  }

  if (disponibles.empty())
    return -1;

  int mejor_valor = numeric_limits<int>::min();
  int mejor_jugada = -1;
  int total_nodos = 0;

#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < disponibles.size(); i++) {
    int jugada = disponibles[i];

    vector<char> tablero_local = tablero;
    tablero_local[jugada] = 'O';

    int nodos_locales = 0;
    int valor = minimax_ab(tablero_local, 0, -1000, 1000, false, nodos_locales);

#pragma omp critical
    {
      total_nodos += nodos_locales;
      if (valor > mejor_valor) {
        mejor_valor = valor;
        mejor_jugada = jugada;
      }
    }
  }

  total_nodos_explorados = total_nodos;
  return mejor_jugada;
}

// Versión secuencial estricta para demostrar la diferencia de rendimiento
int mejor_jugada_secuencial(const vector<char> &tablero) {
  int mejor_valor = numeric_limits<int>::min();
  int mejor_jugada = -1;
  vector<char> copia_tablero = tablero;

  for (int i = 0; i < 9; i++) {
    if (copia_tablero[i] == ' ') {
      copia_tablero[i] = 'O';
      int nodos = 0;
      int valor = minimax_ab(copia_tablero, 0, -1000, 1000, false, nodos);
      copia_tablero[i] = ' ';
      if (valor > mejor_valor) {
        mejor_valor = valor;
        mejor_jugada = i;
      }
    }
  }
  return mejor_jugada;
}

// Ejecuta la prueba de carga para cumplir con "reducción sustancial de tiempo"
void ejecutar_benchmark() {
  vector<char> tablero_vacio(9, ' ');
  int iteraciones = 10000;
  int nodos_dummy = 0;

  cout << "Realizando Benchmark (" << iteraciones
       << " evaluaciones del árbol completo)...\n";

  // Prueba Secuencial
  auto inicio_seq = high_resolution_clock::now();
  for (int i = 0; i < iteraciones; i++)
    mejor_jugada_secuencial(tablero_vacio);
  auto fin_seq = high_resolution_clock::now();
  double tiempo_seq = duration_cast<milliseconds>(fin_seq - inicio_seq).count();

  // Prueba Paralela
  auto inicio_par = high_resolution_clock::now();
  for (int i = 0; i < iteraciones; i++)
    mejor_jugada_paralela(tablero_vacio, nodos_dummy);
  auto fin_par = high_resolution_clock::now();
  double tiempo_par = duration_cast<milliseconds>(fin_par - inicio_par).count();

  cout << "Tiempo Secuencial: " << tiempo_seq << " ms\n";
  cout << "Tiempo Paralelo (OpenMP): " << tiempo_par << " ms\n";
  cout << "Aceleracion: " << tiempo_seq / tiempo_par << "x\n\n";
}

int main() {
  ejecutar_benchmark();

  cout << " ---------Que comience el juego----------\n";
  vector<char> tablero(9, ' ');
  cout << "Posiciones: 0 a 8\n";

  while (verificar_ganador(tablero) == 'N') {
    imprimir_tablero(tablero);

    // Turno humano
    int jugada_humano;
    cout << "Tu turno (X) - Ingresa posicion: ";
    cin >> jugada_humano;

    if (tablero[jugada_humano] != ' ') {
      cout << "Posicion ocupada!\n";
      continue;
    }
    tablero[jugada_humano] = 'X';

    if (verificar_ganador(tablero) != 'N')
      break;

    // Turno IA Paralela
    cout << "La IA (O) esta calculando con OpenMP...\n";
    int nodos_explorados = 0;
    int jugada_ia = mejor_jugada_paralela(tablero, nodos_explorados);
    tablero[jugada_ia] = 'O';
    cout << "La IA jugo en la posicion " << jugada_ia
         << " (Nodos explorados: " << nodos_explorados << ")\n";
  }

  imprimir_tablero(tablero);
  char ganador = verificar_ganador(tablero);
  if (ganador == 'T')
    cout << "Empate!\n";
  else
    cout << "Ganador: " << ganador << "!\n";

  return 0;
}