# nnlearn

`nnlearn` contains a feedforward neural network implementation plus a small MNIST-focused training CLI.

## WebAssembly digit recognizer

The repository can now serve a browser UI that loads `model/model.json`, lets you draw inside a 28×28 frame, and recognizes the digit with the existing Go `FeedForward.Predict` implementation compiled to WebAssembly.

Build the static web bundle:

```sh
make build-pages
go run .
```

Then open `http://localhost:8080`.

The page also loads and shows the first 100 examples from the MNIST training set, exported into `web/training-samples.json`.

For GitHub Pages, publish the contents of `web/`. The app now uses relative URLs, so it can run from a repository subpath.

If you want a standalone web server binary too:

```sh
make build-web
./bin/nnlearn-web
```
