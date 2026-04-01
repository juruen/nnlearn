.PHONY: audit test data build

MNIST_URL := https://storage.googleapis.com/cvdf-datasets/mnist

build:
	go build -o bin/digits ./cmd/digits

audit:
	golangci-lint run ./...
	go test ./... -v

test:
	go test ./... -v

data:
	mkdir -p data
	curl -L -o data/train-images-idx3-ubyte.gz $(MNIST_URL)/train-images-idx3-ubyte.gz
	curl -L -o data/train-labels-idx1-ubyte.gz $(MNIST_URL)/train-labels-idx1-ubyte.gz
	curl -L -o data/t10k-images-idx3-ubyte.gz $(MNIST_URL)/t10k-images-idx3-ubyte.gz
	curl -L -o data/t10k-labels-idx1-ubyte.gz $(MNIST_URL)/t10k-labels-idx1-ubyte.gz
	gunzip -kf data/*.gz
