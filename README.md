# Rolade

**A Minimal Neural Network Toolkit in Go**

---

## Disclaimer

This project was built for fun and learning purposes. There is no guarantee regarding the validity of the methods used, and it is not actively maintained or supported. It is **not recommended** for serious or production use. Use it at your own risk.

---

## Getting Started

This section outlines the basic usage of the Rolade toolkit. Some parts may be incomplete and are subject to future updates.

---

### Defining a Network

First, import the package:

```go
import "github.com/harungurubudi/rolade/network"
```

To create a new network, define the input and output sizes, and specify an activation function. For example, a network with 4 input features, 2 output targets, and ReLU as the activation function:

```go
nt := network.NewNetwork(4, 2, &activation.ReLU{})
```

You can add hidden layers using the AddHiddenLayer method. For example, to add two hidden layers of size 4 and 3 using Tanh activation:

```go
nt.AddHiddenLayer(4, &activation.Tanh{})
nt.AddHiddenLayer(3, &activation.Tanh{})
```

### Configuring Network Properties

You can customize the network using the ``SetProps`` method:

```go
nt.SetProps(network.Props{
    Optimizer: &optimizer.SGD{
        Alpha: 0.01,
    },
    ErrLimit: 0.005,
    MaxEpoch: 10000,
})
```

#### Available Properties

| Options       | Description                                       | Type                      | Default Value           |
|---------------|---------------------------------------------------|---------------------------|-------------------------|
| Optimizer     | Algoritma Optimizer                               | optimizer.IOptimizer      | *optimizer.SGD          |
| Loss          | Fungsi Loss                                       | loss.ILoss                | *loss.RMSE              |
| ErrLimit      | Batas error yang perlu dicapai saat training      | float64                   | 0.001                   |
| MaxEpoch      | Epoch maksimal saat training                      | int                       | 1000                    |

### Data Types

All input and output data use ``network.DataArray``, a custom type based on ``[]float64``.

```go
singleData := network.DataArray{0, 0, 0, 1}
```

### Training the Network

To train the network:

```go
nt.Train(inputs []DataArray, targets []DataArray) error
```

- ``inputs``: Slice of input ``DataArrays``
- ``targets``: Slice of target ``DataArrays``

### Testing the Network

To test the network:

```go
nt.Test(input DataArray) (DataArray, []int, error)
```

Returns:

1. The raw output (DataArray)
2. The binary result after thresholding (values > 0.5 become 1, else 0)
3. An error object if testing fails

### Activation Functions

Currently supported activation functions:

- ``*activation.Sigmoid``
- ``*activation.Tanh``
- ``*activation.ReLU``

### Optimizers

Currently only one optimizer is supported:

- SGD (``*optimizer.SGD``)
  - Alpha: Learning rate
  - M: Momentum
  - IsNesterov: Use Nesterov momentum
  - V: Velocity

### Loss Functions

Only one loss function is available:
- RMSE (``*loss.RMSE``)

### Examples
A simple usage example is available in the ``example`` directory.