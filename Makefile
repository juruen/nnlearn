.PHONY: audit test data build build-web build-wasm build-pages bundle-web

MNIST_URL := https://storage.googleapis.com/cvdf-datasets/mnist

build:
	go build -o bin/digits ./cmd/digits

build-web: bundle-web
	go build -o bin/nnlearn-web .

build-wasm:
	GOOS=js GOARCH=wasm go build -o web/app.wasm ./cmd/recognizer-wasm

bundle-web: build-wasm
	go run ./cmd/build-static-web

build-pages: bundle-web

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
