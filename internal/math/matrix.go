// Package math provides linear algebra helper functions wrapping gonum operations.
package math

import (
	"nnlearn/internal/types"

	"gonum.org/v1/gonum/mat"
)

// MulMatrixVector multiplies a matrix by a vector and returns the resulting vector using gonum.
func MulMatrixVector(m types.Matrix, v types.Vector) types.Vector {
	var result mat.VecDense
	result.MulVec(m, v)
	return &result
}

// AddVectors adds two vectors and returns the resulting vector.
func AddVectors(a, b types.Vector) types.Vector {
	var result mat.VecDense
	result.AddVec(a, b)
	return &result
}

// SubVectors subtracts two vectors and returns the resulting vector.
func SubVectors(a, b types.Vector) types.Vector {
	var result mat.VecDense
	result.SubVec(a, b)
	return &result
}

// ApplyFuncToVector applies a function to each vector component
func ApplyFuncToVector(v types.Vector, f func(float64) float64) types.Vector {
	result := mat.NewVecDense(v.Len(), nil)
	for i := range v.Len() {
		result.SetVec(i, f(v.AtVec(i)))
	}
	return result
}

// MulElemVec performs element-wise (Hadamard) multiplication of two vectors.
func MulElemVec(a, b types.Vector) types.Vector {
	var result mat.VecDense
	result.MulElemVec(a, b)
	return &result
}

// Transpose returns the transpose of a matrix.
func Transpose(m types.Matrix) types.Matrix {
	return m.T()
}

// OuterProduct computes the outer product of two vectors, returning a matrix.
// The result is an m×n matrix where m = a.Len() and n = b.Len(),
// with result[i][j] = a[i] * b[j].
func OuterProduct(a, b types.Vector) types.Matrix {
	var result mat.Dense
	result.Outer(1, a, b)
	return &result
}
