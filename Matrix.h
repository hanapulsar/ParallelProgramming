#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <stdexcept>

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

	size_t get_size() const;

	T* get_data() { return data; }
	const T* get_data() const { return data; }
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