import argparse
import sys

import numpy as np


def parse_arguments() -> argparse.Namespace:
    """
    Adds and parses command-line arguments
    """
    parser = argparse.ArgumentParser(description="Verify Matrix multiplication from C++ via NumPy.")
    parser.add_argument('--inputA', '-a', default='InputA.txt', help='Path to first matrix file. Default: InputA.txt')
    parser.add_argument('--inputB', '-b', default='InputB.txt', help='Path to second matrix file. Default: InputB.txt')
    parser.add_argument('--output', '-o', default='Output.txt', help='Path to result matrix file. Default: Output.txt')
    return parser.parse_args()


def read_matrix_from_file(path: str) -> np.ndarray:
    """
    Read matrices from file
    :param path: path to file with matrix
    """
    try:
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
            tokens = content.split()
            if not tokens:
                raise ValueError("File is empty")
            n = int(tokens[0])
            values = [float(x) for x in tokens[1:]]
            matrix = np.array(values).reshape(n, n)
            return matrix
    except IOError as err:
        raise IOError(f"Wasn't able to read file at {path}: {err}")
    except Exception as err:
        raise RuntimeError(f"Error while parsing file at {path}: {err}")


def verify(args: argparse.Namespace) -> None:
    """
    Verifies CPP result matrix by using NumPy
    :param args: parsed command-line arguments
    """
    try:
        print(f"Matrices to verify")
        print(f"First input matrix: {args.inputA}")
        print(f"Second input matrix: {args.inputB}")
        print(f"Output matrix: {args.output}")

        a = read_matrix_from_file(args.inputA)
        b = read_matrix_from_file(args.inputB)
        o = read_matrix_from_file(args.output)

        o_expected = np.matmul(a, b)

        print(f"Comparing results.")
        # We can use more precision if we add std::setprecision in write_matrix in main.cpp
        # is_correct = np.allclose(C_expected, C_cpp, rtol=1e-05, atol=1e-08)
        is_correct = np.allclose(o_expected, o, rtol=1e-02, atol=1e-02)

        if is_correct:
            print(f"Result matches.")
        else:
            print(f"Result does not match.")
            diff = np.max(np.abs(o_expected - o))
            print(f"Max absolute difference: {diff}")
    except Exception as e:
        print(f"Something went wrong: {e}")
        sys.exit(1)


def main() -> None:
    """
    Main function
    """
    args = parse_arguments()
    verify(args)


if __name__ == "__main__":
    main()
