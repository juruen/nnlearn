// Package mnist loads the MNIST handwritten digit dataset from IDX files.
package mnist

import (
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"

	"gonum.org/v1/gonum/mat"

	"nnlearn/internal/types"
)

// Dataset holds the loaded MNIST images and labels as vectors.
type Dataset struct {
	Images []types.Vector
	Labels []types.Vector
}

// Load reads the MNIST dataset from the given directory.
// It expects the uncompressed IDX files:
//   - train-images-idx3-ubyte / t10k-images-idx3-ubyte
//   - train-labels-idx1-ubyte / t10k-labels-idx1-ubyte
func Load(dir string, prefix string) (*Dataset, error) {
	images, err := loadImages(filepath.Join(dir, prefix+"-images-idx3-ubyte"))
	if err != nil {
		return nil, fmt.Errorf("failed to load images: %w", err)
	}

	labels, err := loadLabels(filepath.Join(dir, prefix+"-labels-idx1-ubyte"))
	if err != nil {
		return nil, fmt.Errorf("failed to load labels: %w", err)
	}

	if len(images) != len(labels) {
		return nil, fmt.Errorf("image count %d != label count %d", len(images), len(labels))
	}

	return &Dataset{Images: images, Labels: labels}, nil
}

func loadImages(path string) ([]types.Vector, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close() //nolint:errcheck

	var magic, count, rows, cols int32
	if err := binary.Read(f, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if magic != 2051 {
		return nil, fmt.Errorf("invalid image magic number: %d", magic)
	}

	if err := binary.Read(f, binary.BigEndian, &count); err != nil {
		return nil, err
	}
	if err := binary.Read(f, binary.BigEndian, &rows); err != nil {
		return nil, err
	}
	if err := binary.Read(f, binary.BigEndian, &cols); err != nil {
		return nil, err
	}

	pixelCount := int(rows) * int(cols)
	images := make([]types.Vector, count)
	buf := make([]byte, pixelCount)

	for i := range images {
		if _, err := f.Read(buf); err != nil {
			return nil, fmt.Errorf("failed to read image %d: %w", i, err)
		}
		data := make([]float64, pixelCount)
		for j, b := range buf {
			data[j] = float64(b) / 255.0
		}
		images[i] = mat.NewVecDense(pixelCount, data)
	}

	return images, nil
}

func loadLabels(path string) ([]types.Vector, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close() //nolint:errcheck

	var magic, count int32
	if err := binary.Read(f, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if magic != 2049 {
		return nil, fmt.Errorf("invalid label magic number: %d", magic)
	}

	if err := binary.Read(f, binary.BigEndian, &count); err != nil {
		return nil, err
	}

	buf := make([]byte, count)
	if _, err := f.Read(buf); err != nil {
		return nil, err
	}

	labels := make([]types.Vector, count)
	for i, b := range buf {
		// One-hot encode: label 3 → [0,0,0,1,0,0,0,0,0,0]
		data := make([]float64, 10)
		data[b] = 1.0
		labels[i] = mat.NewVecDense(10, data)
	}

	return labels, nil
}
