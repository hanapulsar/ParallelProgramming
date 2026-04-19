#include <random>
#include <fstream>
#include <chrono>
//#include <iomanip>
#include "Matrix.cuh"

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
    vector<int> sizes = { 200, 400, 800, 1200, 1600, 2000, 10000};
    vector<int> block_sizes = { 8, 16, 32 };

    Matrix<double> saveA, saveB, saveC;

    //Warm up GPU
    Matrix<double> wA = random_matrix(10);
    Matrix<double> wB = random_matrix(10);
    Matrix<double> wC = wA.multiply_cuda(wB, 8);

    try {
        Matrix<double> A, B, C;

        for (int size : sizes) {
            cout << endl << "Generating random matrices. Size: " << size << "x" << size << endl << endl;
            A = random_matrix(size);
            B = random_matrix(size);

            for (int block : block_sizes) {
                auto start = chrono::high_resolution_clock::now();
                C = A.multiply_cuda(B, block);
                auto end = chrono::high_resolution_clock::now();

                double elapsed_sec = chrono::duration<double>(end - start).count();

                cout << "  Block Size: " << block << "x" << block
                    << " - GPU Time: " << elapsed_sec << " s"
                    << endl;

                if (size == 1200 && block == 16) {
                    saveA = A;
                    saveB = B;
                    saveC = C;
                }
            }
        }

        cout << endl << "Saving sample for verifing, size: 1200*1200 for 16*16 block." << endl;
        write_matrix("InputA.txt", saveA);
        write_matrix("InputB.txt", saveB);
        write_matrix("Output.txt", saveC);

        cout << endl << "Finished." << endl;
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return -1;
    }

    return 0;
}