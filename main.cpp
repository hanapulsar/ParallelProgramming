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

int main(int argc, char* argv[]) {
    try {
        Matrix<double> A, B, C;
        string fileC;

        if (argc == 4) {
            cout << "Getting matrices from files.\n";
            string fileA = argv[1];
            string fileB = argv[2];
            fileC = argv[3];

            A = read_matrix(fileA);
            B = read_matrix(fileB);
        }
        else {
            size_t size = 100;
            cout << "Generating random matrices. Size: " << size << "x" << size << "\n";

            string fileA = "InputA.txt";
            string fileB = "InputB.txt";
            fileC = "Output.txt";

            A = random_matrix(size);
            B = random_matrix(size);

            write_matrix(fileA, A);
            write_matrix(fileB, B);
            cout << "Generated matrices saved to: " << fileA << " and " << fileB << "\n";
        }

        cout << "Matrix A size: " << A.get_size() << "x" << A.get_size() << "\n";
        cout << "Matrix B size: " << B.get_size() << "x" << B.get_size() << "\n";
        cout << "Start calculating." << endl;

        auto start = chrono::high_resolution_clock::now();
        C = A * B;
        auto end = chrono::high_resolution_clock::now();

        double elapsed_sec = chrono::duration<double>(end - start).count();
        cout << "Done in: " << elapsed_sec << " sec\n";

        write_matrix(fileC, C);
        cout << "Result saved to: " << fileC << "\n";
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return -1;
    }

	return 0;
}
