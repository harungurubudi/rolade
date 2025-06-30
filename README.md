# Rolade 🧠

**Rolade** is a minimalistic neural network toolkit written in Go. It is designed to be simple, understandable, and customizable. Rolade is particularly suitable for learning purposes, small-scale experiments, and use cases where Go is preferred over Python or other machine learning languages.

---

## 🚀 Features

* Fully connected multi-layer neural network
* Custom activation, loss, and optimizer support
* Parallel batch training with mergeable deltas
* Simple model save/load support

---

## 📦 Installation

```bash
go get github.com/harungurubudi/rolade
```

---

## 🥺 Quick Start Example

```go
package main

import (
	"github.com/harungurubudi/rolade/network"
	"github.com/harungurubudi/rolade/activation"
	"github.com/harungurubudi/rolade/loss"
	"github.com/harungurubudi/rolade/optimizer"
)

func main() {
	features := []network.Vector{
		{0, 0}, {0, 1}, {1, 0}, {1, 1},
	}
	targets := []network.Vector{
		{0}, {1}, {1}, {0},
	}

	samples, _ := network.NewSamples(features, targets)

	props := &network.Props{
		ErrLimit:  0.001,
		MaxEpoch:  10000,
		Optimizer: optimizer.NewSGDWithLearningRate(0.5),
		Loss:      &loss.RMSE{},
	}

	net, _ := network.NewNetwork(2, 1, activation.NewSigmoid(), props)
	net.AddHidden(2, activation.NewSigmoid())

	_ = net.Train(samples)
}
```

---

## 📃 Model Persistence

```go
_ = net.Save("./model")
loadedNet, _ := network.Load("./model")
```

---

## 📄 API Structure

* `Vector`: Alias for `[]float64`, used for features and targets
* `Samples`: Slice of `Sample` (input + target pair)
* `Network`: Main struct managing the architecture and training
* `Props`: Contains training hyperparameters (optimizer, loss, epoch settings)
* `Activation`, `Loss`, `Optimizer`: Interfaces for customizing behavior

---

## 🎓 Learning Resources

Check the `/network` package for detailed GoDoc-style comments. To inspect inline:

```bash
go doc github.com/harungurubudi/rolade/network
```

Or view it online via [pkg.go.dev](https://pkg.go.dev/github.com/harungurubudi/rolade)

---

## 🏆 Why Rolade?

Rolade was built to revisit and understand the fundamentals of how neural networks operate under the hood, particularly in Go. While it's not optimized for large-scale production ML, it provides a transparent and hackable playground for enthusiasts and learners.

---

## ✉️ License

MIT License
