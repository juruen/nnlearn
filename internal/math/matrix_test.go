package math

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gonum.org/v1/gonum/mat"
)

func TestMulMatrixVector(t *testing.T) {
	t.Run("basic multiplication", func(t *testing.T) {
		// [1 2]   [1]   [1*1 + 2*2]   [5]
		// [3 4] × [2] = [3*1 + 4*2] = [11]
		m := mat.NewDense(2, 2, []float64{1, 2, 3, 4})
		v := mat.NewVecDense(2, []float64{1, 2})

		result := MulMatrixVector(m, v)

		require.Equal(t, 2, result.Len())
		assert.InDelta(t, 5.0, result.AtVec(0), 1e-10)
		assert.InDelta(t, 11.0, result.AtVec(1), 1e-10)
	})

	t.Run("non-square matrix", func(t *testing.T) {
		// [1 2 3]   [1]   [1+4+9]    [14]
		// [4 5 6] × [2] = [4+10+18] = [32]
		//            [3]
		m := mat.NewDense(2, 3, []float64{1, 2, 3, 4, 5, 6})
		v := mat.NewVecDense(3, []float64{1, 2, 3})

		result := MulMatrixVector(m, v)

		require.Equal(t, 2, result.Len())
		assert.InDelta(t, 14.0, result.AtVec(0), 1e-10)
		assert.InDelta(t, 32.0, result.AtVec(1), 1e-10)
	})

	t.Run("identity matrix", func(t *testing.T) {
		m := mat.NewDense(3, 3, []float64{
			1, 0, 0,
			0, 1, 0,
			0, 0, 1,
		})
		v := mat.NewVecDense(3, []float64{7, 8, 9})

		result := MulMatrixVector(m, v)

		require.Equal(t, 3, result.Len())
		assert.InDelta(t, 7.0, result.AtVec(0), 1e-10)
		assert.InDelta(t, 8.0, result.AtVec(1), 1e-10)
		assert.InDelta(t, 9.0, result.AtVec(2), 1e-10)
	})

	t.Run("zero vector", func(t *testing.T) {
		m := mat.NewDense(2, 2, []float64{1, 2, 3, 4})
		v := mat.NewVecDense(2, nil)

		result := MulMatrixVector(m, v)

		require.Equal(t, 2, result.Len())
		assert.Equal(t, 0.0, result.AtVec(0))
		assert.Equal(t, 0.0, result.AtVec(1))
	})
}

func TestAddVectors(t *testing.T) {
	t.Run("basic addition", func(t *testing.T) {
		a := mat.NewVecDense(3, []float64{1, 2, 3})
		b := mat.NewVecDense(3, []float64{4, 5, 6})

		result := AddVectors(a, b)

		require.Equal(t, 3, result.Len())
		assert.InDelta(t, 5.0, result.AtVec(0), 1e-10)
		assert.InDelta(t, 7.0, result.AtVec(1), 1e-10)
		assert.InDelta(t, 9.0, result.AtVec(2), 1e-10)
	})

	t.Run("negative values", func(t *testing.T) {
		a := mat.NewVecDense(2, []float64{3, -1})
		b := mat.NewVecDense(2, []float64{-3, 1})

		result := AddVectors(a, b)

		require.Equal(t, 2, result.Len())
		assert.Equal(t, 0.0, result.AtVec(0))
		assert.Equal(t, 0.0, result.AtVec(1))
	})

	t.Run("zero vector", func(t *testing.T) {
		a := mat.NewVecDense(3, []float64{1, 2, 3})
		b := mat.NewVecDense(3, nil)

		result := AddVectors(a, b)

		require.Equal(t, 3, result.Len())
		assert.InDelta(t, 1.0, result.AtVec(0), 1e-10)
		assert.InDelta(t, 2.0, result.AtVec(1), 1e-10)
		assert.InDelta(t, 3.0, result.AtVec(2), 1e-10)
	})
}

func TestSubVectors(t *testing.T) {
	t.Run("basic subtraction", func(t *testing.T) {
		a := mat.NewVecDense(3, []float64{5, 7, 9})
		b := mat.NewVecDense(3, []float64{1, 2, 3})

		result := SubVectors(a, b)

		require.Equal(t, 3, result.Len())
		assert.InDelta(t, 4.0, result.AtVec(0), 1e-10)
		assert.InDelta(t, 5.0, result.AtVec(1), 1e-10)
		assert.InDelta(t, 6.0, result.AtVec(2), 1e-10)
	})

	t.Run("same vectors gives zero", func(t *testing.T) {
		a := mat.NewVecDense(3, []float64{1, 2, 3})
		b := mat.NewVecDense(3, []float64{1, 2, 3})

		result := SubVectors(a, b)

		require.Equal(t, 3, result.Len())
		assert.Equal(t, 0.0, result.AtVec(0))
		assert.Equal(t, 0.0, result.AtVec(1))
		assert.Equal(t, 0.0, result.AtVec(2))
	})

	t.Run("negative result", func(t *testing.T) {
		a := mat.NewVecDense(2, []float64{1, 2})
		b := mat.NewVecDense(2, []float64{3, 5})

		result := SubVectors(a, b)

		require.Equal(t, 2, result.Len())
		assert.InDelta(t, -2.0, result.AtVec(0), 1e-10)
		assert.InDelta(t, -3.0, result.AtVec(1), 1e-10)
	})
}

func TestApplyVectorFunc(t *testing.T) {
	t.Run("double each element", func(t *testing.T) {
		v := mat.NewVecDense(3, []float64{1, 2, 3})

		result := ApplyFuncToVector(v, func(x float64) float64 { return x * 2 })

		require.Equal(t, 3, result.Len())
		assert.InDelta(t, 2.0, result.AtVec(0), 1e-10)
		assert.InDelta(t, 4.0, result.AtVec(1), 1e-10)
		assert.InDelta(t, 6.0, result.AtVec(2), 1e-10)
	})

	t.Run("square each element", func(t *testing.T) {
		v := mat.NewVecDense(3, []float64{2, 3, 4})

		result := ApplyFuncToVector(v, func(x float64) float64 { return x * x })

		require.Equal(t, 3, result.Len())
		assert.InDelta(t, 4.0, result.AtVec(0), 1e-10)
		assert.InDelta(t, 9.0, result.AtVec(1), 1e-10)
		assert.InDelta(t, 16.0, result.AtVec(2), 1e-10)
	})

	t.Run("apply math.Abs", func(t *testing.T) {
		v := mat.NewVecDense(3, []float64{-1, 2, -3})

		result := ApplyFuncToVector(v, math.Abs)

		require.Equal(t, 3, result.Len())
		assert.InDelta(t, 1.0, result.AtVec(0), 1e-10)
		assert.InDelta(t, 2.0, result.AtVec(1), 1e-10)
		assert.InDelta(t, 3.0, result.AtVec(2), 1e-10)
	})

	t.Run("does not mutate input", func(t *testing.T) {
		v := mat.NewVecDense(2, []float64{5, 10})

		ApplyFuncToVector(v, func(x float64) float64 { return x + 1 })

		assert.Equal(t, 5.0, v.AtVec(0))
		assert.Equal(t, 10.0, v.AtVec(1))
	})
}

func TestMulElemVec(t *testing.T) {
	t.Run("basic element-wise multiplication", func(t *testing.T) {
		a := mat.NewVecDense(3, []float64{2, 3, 4})
		b := mat.NewVecDense(3, []float64{5, 6, 7})

		result := MulElemVec(a, b)

		require.Equal(t, 3, result.Len())
		assert.InDelta(t, 10.0, result.AtVec(0), 1e-10)
		assert.InDelta(t, 18.0, result.AtVec(1), 1e-10)
		assert.InDelta(t, 28.0, result.AtVec(2), 1e-10)
	})

	t.Run("multiply by zero vector", func(t *testing.T) {
		a := mat.NewVecDense(3, []float64{1, 2, 3})
		b := mat.NewVecDense(3, nil)

		result := MulElemVec(a, b)

		require.Equal(t, 3, result.Len())
		assert.Equal(t, 0.0, result.AtVec(0))
		assert.Equal(t, 0.0, result.AtVec(1))
		assert.Equal(t, 0.0, result.AtVec(2))
	})

	t.Run("multiply by ones", func(t *testing.T) {
		a := mat.NewVecDense(3, []float64{7, 8, 9})
		b := mat.NewVecDense(3, []float64{1, 1, 1})

		result := MulElemVec(a, b)

		require.Equal(t, 3, result.Len())
		assert.InDelta(t, 7.0, result.AtVec(0), 1e-10)
		assert.InDelta(t, 8.0, result.AtVec(1), 1e-10)
		assert.InDelta(t, 9.0, result.AtVec(2), 1e-10)
	})

	t.Run("negative values", func(t *testing.T) {
		a := mat.NewVecDense(2, []float64{-2, 3})
		b := mat.NewVecDense(2, []float64{4, -5})

		result := MulElemVec(a, b)

		require.Equal(t, 2, result.Len())
		assert.InDelta(t, -8.0, result.AtVec(0), 1e-10)
		assert.InDelta(t, -15.0, result.AtVec(1), 1e-10)
	})
}

func TestTranspose(t *testing.T) {
	t.Run("square matrix", func(t *testing.T) {
		// [1 2]T   [1 3]
		// [3 4]  = [2 4]
		m := mat.NewDense(2, 2, []float64{1, 2, 3, 4})

		result := Transpose(m)

		r, c := result.Dims()
		assert.Equal(t, 2, r)
		assert.Equal(t, 2, c)
		assert.Equal(t, 1.0, result.At(0, 0))
		assert.Equal(t, 3.0, result.At(0, 1))
		assert.Equal(t, 2.0, result.At(1, 0))
		assert.Equal(t, 4.0, result.At(1, 1))
	})

	t.Run("non-square matrix", func(t *testing.T) {
		// [1 2 3]T   [1 4]
		// [4 5 6]  = [2 5]
		//             [3 6]
		m := mat.NewDense(2, 3, []float64{1, 2, 3, 4, 5, 6})

		result := Transpose(m)

		r, c := result.Dims()
		assert.Equal(t, 3, r)
		assert.Equal(t, 2, c)
		assert.Equal(t, 1.0, result.At(0, 0))
		assert.Equal(t, 4.0, result.At(0, 1))
		assert.Equal(t, 2.0, result.At(1, 0))
		assert.Equal(t, 5.0, result.At(1, 1))
		assert.Equal(t, 3.0, result.At(2, 0))
		assert.Equal(t, 6.0, result.At(2, 1))
	})

	t.Run("double transpose is identity", func(t *testing.T) {
		m := mat.NewDense(2, 3, []float64{1, 2, 3, 4, 5, 6})

		result := Transpose(Transpose(m))

		r, c := result.Dims()
		assert.Equal(t, 2, r)
		assert.Equal(t, 3, c)
		for i := range r {
			for j := range c {
				assert.Equal(t, m.At(i, j), result.At(i, j))
			}
		}
	})
}

func TestOuterProduct(t *testing.T) {
	t.Run("basic outer product", func(t *testing.T) {
		// [1]           [1*4 1*5]   [4  5]
		// [2] × [4 5] = [2*4 2*5] = [8 10]
		// [3]           [3*4 3*5]   [12 15]
		a := mat.NewVecDense(3, []float64{1, 2, 3})
		b := mat.NewVecDense(2, []float64{4, 5})

		result := OuterProduct(a, b)

		r, c := result.Dims()
		assert.Equal(t, 3, r)
		assert.Equal(t, 2, c)
		assert.InDelta(t, 4.0, result.At(0, 0), 1e-10)
		assert.InDelta(t, 5.0, result.At(0, 1), 1e-10)
		assert.InDelta(t, 8.0, result.At(1, 0), 1e-10)
		assert.InDelta(t, 10.0, result.At(1, 1), 1e-10)
		assert.InDelta(t, 12.0, result.At(2, 0), 1e-10)
		assert.InDelta(t, 15.0, result.At(2, 1), 1e-10)
	})

	t.Run("same length vectors", func(t *testing.T) {
		// [2]           [2*3 2*4]   [6  8]
		// [5] × [3 4] = [5*3 5*4] = [15 20]
		a := mat.NewVecDense(2, []float64{2, 5})
		b := mat.NewVecDense(2, []float64{3, 4})

		result := OuterProduct(a, b)

		r, c := result.Dims()
		assert.Equal(t, 2, r)
		assert.Equal(t, 2, c)
		assert.InDelta(t, 6.0, result.At(0, 0), 1e-10)
		assert.InDelta(t, 8.0, result.At(0, 1), 1e-10)
		assert.InDelta(t, 15.0, result.At(1, 0), 1e-10)
		assert.InDelta(t, 20.0, result.At(1, 1), 1e-10)
	})

	t.Run("zero vector", func(t *testing.T) {
		a := mat.NewVecDense(2, []float64{1, 2})
		b := mat.NewVecDense(3, nil)

		result := OuterProduct(a, b)

		r, c := result.Dims()
		assert.Equal(t, 2, r)
		assert.Equal(t, 3, c)
		for i := range r {
			for j := range c {
				assert.Equal(t, 0.0, result.At(i, j))
			}
		}
	})
}
