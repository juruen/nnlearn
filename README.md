# nnlearn

`nnlearn` is a learning project to understand AI from the ground up.

I am starting with feed forward neural networks. The first goal is the canonical handwritten-digit example: training a neural network on the mythical MNIST dataset and using it to recognize digits.

This repository includes a native Go implementation of a feed forward network together with the supporting pieces needed to train it and inspect the results.

My main focus here is understanding the math behind the model. Because of that, I did not use AI to implement the fundamentals such as cost, error, and backpropagation. I wanted to make sure I could understand the equations and translate them into code myself, without looking at existing implementations or relying on prebuilt neural-network libraries.

## WebAssembly digit recognizer

The project also includes a browser demo that runs the Go `FeedForward.Predict` implementation through WebAssembly. It loads `model/model.json`, lets you draw inside a 28×28 frame, and shows the recognized digit.

You can try it here:

[https://juruen.github.io/nnlearn/web](https://juruen.github.io/nnlearn/web)

Source code:

[https://github.com/juruen/nnlearn](https://github.com/juruen/nnlearn)

## References

- Rumelhart, D. E., Hinton, G. E., and Williams, R. J. (1986). *Learning representations by back-propagating errors*. Nature, 323(6088), 533-536. https://doi.org/10.1038/323533a0
- Efron, Bradley, and Trevor Hastie. *Computer Age Statistical Inference: Algorithms, Evidence, and Data Science*. Cambridge University Press, 2016. See p. 351, where Chapter 18, *Neural Networks and Deep Learning*, begins.
