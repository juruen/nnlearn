# nnlearn

`nnlearn` contains a feedforward neural network implementation plus a small MNIST-focused training CLI.

## WebAssembly digit recognizer

The repository can now serve a browser UI that loads `model/model.json`, lets you draw inside a 28×28 frame, and recognizes the digit with the existing Go `FeedForward.Predict` implementation compiled to WebAssembly.

Build the web assets:

```sh
make build-wasm
go run .
```

Then open `http://localhost:8080`.

The page also loads and shows the first 100 examples from the MNIST training set served from `data/train-*`.

If you want a standalone web server binary too:

```sh
make build-web
./bin/nnlearn-web
```
