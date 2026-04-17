#include <random>
#include <fstream>
#include <chrono>
//#include <iomanip>
#include "Matrix.h"

using namespace std;

//Generate random square matrix
static Matrix<double> random_matrix(size_t n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    Matrix<double> m(n);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            m(i, j) = dist(gen);
        }
    }
    return m;
}

/*
//Read matrix from file
static Matrix<double> read_matrix(const string& filepath) {
    ifstream input(filepath);
    if (!input.is_open()) {
        throw runtime_error("Can't open: " + filepath);
    }
    size_t n;
    input >> n;
    Matrix<double> matrix(n);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            input >> matrix(i, j);
        }
    }
    return matrix;
}
*/

//Write matrix from file
static void write_matrix(const string& filepath, const Matrix<double>& matrix) {
    ofstream output(filepath);
    if (!output.is_open()) {
        throw runtime_error("Can't open: " + filepath);
    }
    output << matrix.get_size() << "\n";

    //Need this if not using rounding in .py for comparison
    //output << std::setprecision(17) << matrix;

    output << matrix;
}

int main() {
    vector<int> sizes = { 200, 400, 800, 1200, 1600, 2000 };
    vector<int> threads = { 1, 2, 4, 8 };

    try {
        Matrix<double> A, B, C;

        for (int size : sizes) {
            cout << "\nGenerating random matrices. Size: " << size << "x" << size << "\n\n";
            A = random_matrix(size);
            B = random_matrix(size);

            double time_1_thread = 0.0;

            for (int t : threads) {
                auto start = chrono::high_resolution_clock::now();
                //TODO MPI
                //C = A.multiply_omp(B, t);
                auto end = chrono::high_resolution_clock::now();

                double elapsed_sec = chrono::duration<double>(end - start).count();

                double acceleration = 1.0;
                if (t == 1) {
                    time_1_thread = elapsed_sec;
                }
                else {
                    acceleration = time_1_thread / elapsed_sec;
                }

                double efficiency = acceleration / t;

                cout << "Size: " << size << "\n";
                cout << "Threads: " << t << "\n";
                cout << "Elapsed time: " << elapsed_sec << "\n";
                cout << "Accelaration: " << acceleration << "\n";
                cout << "Efficiency: " << efficiency << "\n";

                if (size == 1200 && t == 4) {
                    cout << "Saving sample for verifing, size: 1200*1200 for 4 threads.\n";
                    write_matrix("InputA.txt", A);
                    write_matrix("InputB.txt", B);
                    write_matrix("Output.txt", C);
                }
            }
        }
        cout << "Finished.\n";
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return -1;
    }

    return 0;
}