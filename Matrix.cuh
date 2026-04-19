#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>

template <typename T>
class Matrix {
private:
	size_t size;
	T* data = nullptr;

public:
	Matrix();
	Matrix(size_t size);
	~Matrix();
	Matrix(const Matrix& src);
	Matrix& operator=(const Matrix& src);

	T& operator()(size_t row, size_t column);
	const T& operator()(size_t row, size_t column) const;

	Matrix operator*(const Matrix& src) const;
	Matrix multiply_cuda(const Matrix& src, int block_size) const;

	size_t get_size() const;
};

//Default constructor
template <typename T>
Matrix<T>::Matrix() : size(0) {
}

//Constructor
template <typename T>
Matrix<T>::Matrix(size_t size) : size(size) {
	if (size == 0) {
		data = nullptr;
		return;
	}

	data = new T[size * size]();
}

//Destructor
template <typename T>
Matrix<T>::~Matrix() {
	delete[] data;
}

//Copy constructor
template <typename T>
Matrix<T>::Matrix(const Matrix& src) : size(src.size) {
	if (size > 0) {
		data = new T[size * size];
		for (size_t i = 0; i < size * size; ++i) {
			data[i] = src.data[i];
		}
	}
}

//Assigment operator
template <typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix& src) {
	if (this == &src) {
		return *this;
	}

	delete[] data;

	size = src.size;

	if (size > 0) {
		data = new T[size * size];
		for (size_t i = 0; i < size * size; ++i) {
			data[i] = src.data[i];
		}
	}
	else {
		data = nullptr;
	}

	return *this;
}


//Functor operator (write)
template <typename T>
T& Matrix<T>::operator()(size_t row, size_t column) {
	if (row >= size || column >= size) {
		throw std::out_of_range("Index out of range.");
	}

	return data[row * size + column];
}


//Functor operator const (read)
template <typename T>
const T& Matrix<T>::operator()(size_t row, size_t column) const {
	if (row >= size || column >= size) {
		throw std::out_of_range("Index out of range.");
	}

	return data[row * size + column];
}

//Multiplication operator
template <typename T>
Matrix<T> Matrix<T>::operator*(const Matrix& src) const {
	if (size != src.size) {
		throw std::invalid_argument("Matrix dimensions different.");
	}

	Matrix<T> result(size);

	for (size_t i = 0; i < size; ++i) {
		for (size_t j = 0; j < size; ++j) {
			T sum = 0;
			for (size_t k = 0; k < size; ++k) {
				sum += (*this)(i, k) * src(k, j);
			}
			result(i, j) = sum;
		}
	}

	return result;
}

//Kernel multiply
template <typename T>
__global__ void KernelMultiplication(const T* A, const T* B, T* C, int N) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < N && col < N) {
		T sum = 0;
		for (int k = 0; k < N; ++k) {
			sum += A[row * N + k] * B[k * N + col];
		}
		C[row * N + col] = sum;
	}
}

//Multiplication operator - CUDA
template <typename T>
Matrix<T> Matrix<T>::multiply_cuda(const Matrix& src, int block_size) const {
	if (size != src.size) {
		throw std::invalid_argument("Matrix dimensions different.");
	}

	Matrix<T> result(size);
	int n = static_cast<int>(size);
	//Memory needed
	size_t bytes = n * n * sizeof(T);

	T* A, * B, * C;
	
	//Allocate memory
	cudaMalloc(&A, bytes);
	cudaMalloc(&B, bytes);
	cudaMalloc(&C, bytes);

	//Matrix copy
	cudaMemcpy(A, data, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(B, src.data, bytes, cudaMemcpyHostToDevice);

	//Grid and blocks
	dim3 threadsPerBlock(block_size, block_size);
	dim3 blocksPerGrid((n + block_size - 1) / block_size, (n + block_size - 1) / block_size);

	//Multiply
	KernelMultiplication << <blocksPerGrid, threadsPerBlock >> > (A, B, C, n);

	//Sync
	cudaDeviceSynchronize();

	//Copy result
	cudaMemcpy(result.data, C, bytes, cudaMemcpyDeviceToHost);

	//Free memory
	cudaFree(A);
	cudaFree(B);
	cudaFree(C);

	return result;
}

//Getter
template <typename T>
size_t Matrix<T>::get_size() const {
	return size;
}

//Output operator
template <typename T>
std::ostream& operator<<(std::ostream& ostream, const Matrix<T>& src) {
	size_t N = src.get_size();
	for (size_t i = 0; i < N; ++i) {
		for (size_t j = 0; j < N; ++j) {
			ostream << src(i, j);
			if (j < N - 1) {
				ostream << " ";
			}
		}
		ostream << "\n";
	}
	return ostream;
}

#endif