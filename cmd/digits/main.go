// Package main provides the digits CLI for training a feedforward neural network on MNIST.
package main

import (
	"fmt"
	"math/rand/v2"
	"os"
	"time"

	"nnlearn/internal/mnist"
	"nnlearn/internal/nn"
	"nnlearn/internal/types"

	"github.com/spf13/cobra"
)

func main() {
	var (
		dataDir      string
		outputPath   string
		hiddenLayers []int
		inputLen     int
		outputLen    int
		learningRate float64
		batchSize    int
		epochs       int
		seed         uint64
	)

	rootCmd := &cobra.Command{
		Use:   "digits",
		Short: "Train a feedforward neural network on MNIST handwritten digits",
		RunE: func(_ *cobra.Command, _ []string) error {
			fmt.Println("Loading training data...")
			trainData, err := mnist.Load(dataDir, "train")
			if err != nil {
				return fmt.Errorf("failed to load training data: %w", err)
			}
			fmt.Printf("Loaded %d training samples\n", len(trainData.Images))

			fmt.Println("Loading test data...")
			testData, err := mnist.Load(dataDir, "t10k")
			if err != nil {
				return fmt.Errorf("failed to load test data: %w", err)
			}
			fmt.Printf("Loaded %d test samples\n", len(testData.Images))

			opts := []nn.Option{nn.WithSeed(seed)}
			network := nn.NewFeedForward(inputLen, hiddenLayers, outputLen, opts...)

			fmt.Printf("Network: %d -> %v -> %d\n", inputLen, hiddenLayers, outputLen)
			fmt.Printf("Learning rate: %f, Batch size: %d, Epochs: %d\n", learningRate, batchSize, epochs)

			for epoch := range epochs {
				start := time.Now()

				// Shuffle training data
				perm := rand.Perm(len(trainData.Images))

				// Train in mini-batches
				for batchStart := 0; batchStart < len(trainData.Images); batchStart += batchSize {
					batchEnd := batchStart + batchSize
					if batchEnd > len(trainData.Images) {
						batchEnd = len(trainData.Images)
					}

					inputs := make([]types.Vector, 0, batchEnd-batchStart)
					outputs := make([]types.Vector, 0, batchEnd-batchStart)
					for _, idx := range perm[batchStart:batchEnd] {
						inputs = append(inputs, trainData.Images[idx])
						outputs = append(outputs, trainData.Labels[idx])
					}

					batch := types.TrainingBatch{
						Inputs:       inputs,
						Outputs:      outputs,
						LearningRate: learningRate,
					}

					if _, err := network.Train(batch); err != nil {
						return fmt.Errorf("training failed at epoch %d: %w", epoch, err)
					}
				}

				// Evaluate on test data
				correct := 0
				for i := range testData.Images {
					predicted, err := network.Predict(testData.Images[i])
					if err != nil {
						return fmt.Errorf("prediction failed: %w", err)
					}
					if argmax(predicted) == argmax(testData.Labels[i]) {
						correct++
					}
				}

				elapsed := time.Since(start)
				accuracy := float64(correct) / float64(len(testData.Images)) * 100
				fmt.Printf("Epoch %d: %d/%d correct (%.2f%%) [%s]\n",
					epoch+1, correct, len(testData.Images), accuracy, elapsed.Round(time.Millisecond))
			}

			if outputPath != "" {
				if err := network.Save(outputPath); err != nil {
					return fmt.Errorf("failed to save network: %w", err)
				}
				fmt.Printf("Network saved to %s\n", outputPath)
			}

			return nil
		},
	}

	flags := rootCmd.Flags()
	flags.StringVar(&dataDir, "data-dir", "data", "directory containing MNIST data files")
	flags.StringVarP(&outputPath, "output", "o", "", "path to save the trained network (JSON)")
	flags.IntSliceVar(&hiddenLayers, "hidden", []int{30}, "hidden layer sizes (comma-separated)")
	flags.IntVar(&inputLen, "input", 784, "input layer size (28×28 pixels)")
	flags.IntVar(&outputLen, "output-len", 10, "output layer size (10 digits)")
	flags.Float64VarP(&learningRate, "learning-rate", "l", 3.0, "learning rate (η)")
	flags.IntVarP(&batchSize, "batch-size", "b", 10, "mini-batch size")
	flags.IntVarP(&epochs, "epochs", "e", 30, "number of training epochs")
	flags.Uint64Var(&seed, "seed", 42, "random seed for weight initialization")

	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}

func argmax(v types.Vector) int {
	maxIdx := 0
	maxVal := v.AtVec(0)
	for i := 1; i < v.Len(); i++ {
		if v.AtVec(i) > maxVal {
			maxVal = v.AtVec(i)
			maxIdx = i
		}
	}
	return maxIdx
}
