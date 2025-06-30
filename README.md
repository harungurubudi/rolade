# Rolade

**A Minimal Neural Network Toolkit in Go**

---

## Disclaimer

This project was originally built for fun and learning purposes. While it has evolved with improvements like batch-parallel training, delta merging, and better test coverage, it is **still experimental**. Use it at your own risk.

---

## Installation

```bash
go get github.com/harungurubudi/rolade
```

---

## Quick Example

```go
nt, _ := network.NewNetwork(4, 2, &activation.Sigmoid{}, nil)
nt.AddHiddenLayer(4, &activation.ReLU{})

features := []network.Vector{
	{0, 0, 0, 1},
	{1, 1, 0, 0},
}
targets := []network.Vector{
	{0, 1},
	{1, 0},
}

samples, _ := network.NewSamples(features, targets)
err := nt.Train(samples)
```

---

## Defining a Network

```go
nt := network.NewNetwork(4, 2, &activation.ReLU{}, nil)
nt.AddHiddenLayer(4, &activation.Tanh{})
nt.AddHiddenLayer(3, &activation.Tanh{})
```

---

## Configuring Network Properties

```go
nt.SetProps(network.Props{
    Optimizer: &optimizer.SGD{Alpha: 0.01},
    ErrLimit:  0.005,
    MaxEpoch:  10000,
})
```

### Available Properties

| Property  | Description                         | Type                   | Default |
| --------- | ----------------------------------- | ---------------------- | ------- |
| Optimizer | Algorithm for weight updates        | `optimizer.IOptimizer` | SGD     |
| Loss      | Loss function                       | `loss.ILoss`           | RMSE    |
| ErrLimit  | Target error to stop training early | `float64`              | 0.001   |
| MaxEpoch  | Maximum training epochs             | `int`                  | 10000   |
| Patience  | Epochs to wait without improvement  | `int`                  | 1000    |

---

## Data Structures

### Vector

A type alias for `[]float64`. Used for both features and targets.

### Sample

```go
type Sample struct {
	Feature Vector
	Target  Vector
}
```

### Samples

```go
type Samples []Sample
```

Create samples:

```go
samples, _ := network.NewSamples(features, targets)
```

Split into mini-batches:

```go
batches := samples.Split(batchSize)
```

---

## Training

```go
err := nt.Train(samples)
```

* Trains using configurable epochs, learning rate, loss, optimizer.
* Supports concurrent batch training and delta merging.

---

## Testing

```go
output, binary, err := nt.Test(input)
```

* `output`: the raw output vector
* `binary`: each output value thresholded (e.g., > 0.5 → 1)
* `err`: if computation failed

---

## Activation Functions

* `*activation.Sigmoid`
* `*activation.Tanh`
* `*activation.ReLU`

---

## Optimizers

* **SGD** (`*optimizer.SGD`)

  * `Alpha`: learning rate
  * `Momentum`: optional
  * `IsNesterov`: optional

---

## Loss Functions

* `*loss.RMSE`

---

## Saving and Loading Models

```go
nt.Save("model_dir")
nt2, err := network.Load("model_dir")
```

---

## Example Usage

See the `example/` directory for full examples, including the Iris dataset classification.

---

## License

MIT
