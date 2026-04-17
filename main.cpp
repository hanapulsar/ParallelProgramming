#include <random>
#include <fstream>
#include <chrono>
#include <mpi.h>
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    vector<int> sizes = { 200, 400, 800, 1200, 1600, 2000 };

    try {
        for (int size : sizes) {
            Matrix<double> A, B, C;

            if (rank == 0) {
                //Generate matrices only once
                cout << "\nGenerating random matrices. Size: " << size << "x" << size << endl;
                A = random_matrix(size);
                B = random_matrix(size);
                C = Matrix<double>(size);
            }
            else {
                //Allocate memory for B matrix
                B = Matrix<double>(size);
            }

            //Broadcast B matrix
            MPI_Bcast(B.get_data(), size * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            //Allocate memory for local A and C matrices
            int rows_for_proc = size / num_procs;
            int local_matrix_size = rows_for_proc * size;
            vector<double> local_A(local_matrix_size);
            vector<double> local_C(local_matrix_size);

            //Scatter A matrix
            double* send_ptr_A = nullptr;
            if (rank == 0) {
                send_ptr_A = A.get_data();
            }
            MPI_Scatter(send_ptr_A, local_matrix_size, MPI_DOUBLE, //From
                local_A.data(), local_matrix_size, MPI_DOUBLE, //To
                0, MPI_COMM_WORLD);

            //Sync
            MPI_Barrier(MPI_COMM_WORLD);
            double start_time = MPI_Wtime();

            //Multiple local matrices
            for (int i = 0; i < rows_for_proc; ++i) {
                for (size_t j = 0; j < size; ++j) {
                    double sum = 0.0;
                    for (size_t k = 0; k < size; ++k) {
                        sum += local_A[i * size + k] * B(k, j);
                    }
                    local_C[i * size + j] = sum;
                }
            }

            //Sync
            MPI_Barrier(MPI_COMM_WORLD);
            double end_time = MPI_Wtime();

            //Gather C matrix
            double* recv_ptr_C = nullptr;
            if (rank == 0) {
                recv_ptr_C = C.get_data();
            }
            MPI_Gather(local_C.data(), local_matrix_size, MPI_DOUBLE,
                recv_ptr_C, local_matrix_size, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

            //Display results
            if (rank == 0) {
                double elapsed_sec = end_time - start_time;
                cout << "Size: " << size << " Procs: " << num_procs << " Time: " << elapsed_sec << " s" << endl;
                
                if (size == 1200 && num_procs == 2) {
                    cout << "Saving sample for verifing, size: 1200*1200 for 4 processes." << endl;
                    write_matrix("InputA.txt", A);
                    write_matrix("InputB.txt", B);
                    write_matrix("Output.txt", C);
                }
            }
        }
        if (rank == 0) cout << "Finished." << endl;
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}