package network

import (
	"fmt"
	"math/rand"
	"encoding/json"
	"io/ioutil"

	"github.com/harungurubudi/rolade/activation"
	"github.com/harungurubudi/rolade/loss"
	"github.com/harungurubudi/rolade/optimizer"
	"github.com/harungurubudi/rolade/profile"
)

// NewNetwork creates and initializes a new feedforward neural network with a single layer,
// using the given input and output sizes and a shared activation function for the layer.
//
// The network is configured with default training properties: RMSE loss,
// SGD optimizer, a max error threshold of 0.001, and up to 1000 training epochs.
//
// Returns the initialized *Network or an error if layer creation fails.
func NewNetwork(inputSize int, outputSize int, activation activation.IActivation) (nt *Network, err error) {
	var synaptics []synaptic
	sy, err := generateSynaptic(inputSize, outputSize, activation)
	if err != nil {
		return nt, fmt.Errorf("Got error while add layer: %v", err)
	}

	synaptics = append(synaptics, sy)

	nt = &Network{
		inputSize: inputSize,
		outputSize: outputSize,
		props: Props{
			Loss:       &loss.RMSE{},
			Optimizer:  optimizer.NewSGD(),
			ErrLimit:  	0.001,
			MaxEpoch: 	1000,
		},
		synaptics: synaptics,
	}
	
	return nt, nil
}

// Load reads a serialized network profile from the specified directory and reconstructs
// the full network configuration, including layers, weights, activation functions,
// loss function, and optimizer.
//
// The profile file must be located at `path + "/rolade.profile"`.
//
// Returns the restored *Network or an error if deserialization or instantiation fails.
func Load(path string) (nt *Network, err error) {
	ntb, err := ioutil.ReadFile(path + "/rolade.profile")
    if err != nil {
		return nil, fmt.Errorf("Got error while load model file: %v", err)
	}

	var pr profile.Network
	err = json.Unmarshal(ntb, &pr)
	if err != nil {
		return nil, fmt.Errorf("Got error while unmarshalling model: %v", err)
	}

	var syn []synaptic
	for _, sySource := range pr.Synaptics {
		a, err := activation.Generate(&sySource.Activation)
		if err != nil {
			return nil, fmt.Errorf("Got error while load model: %v", err)
		}

		syn = append(syn, synaptic{
			sourceSize: sySource.SourceSize,
			targetSize: sySource.TargetSize,
			weight: weight{
				w: sySource.Weight.W,
				b: sySource.Weight.B,
			},
			activation: a, 
		})
	}

	l, err := loss.Generate(&pr.Props.Loss)
	if err != nil {
		return nil, fmt.Errorf("Got error while load model: %v", err)
	}

	o, err := optimizer.Generate(&pr.Props.Optimizer)
	if err != nil {
		return nil, fmt.Errorf("Got error while load model: %v", err)
	}

	return &Network{
		inputSize: pr.InputSize,
		outputSize: pr.OutputSize,
		props: Props{
			Loss: l,
			Optimizer: o, 
			ErrLimit: pr.Props.ErrLimit,
			MaxEpoch: pr.Props.MaxEpoch,
		},
		synaptics: syn,
	}, nil
}

func generateSynaptic(sourceSize int, targetSize int, activation activation.IActivation) (sy synaptic, err error) {
	var w [][]float64 
	for i := 0; i < sourceSize; i++ {
		var tmp []float64
		for j := 0; j < targetSize; j++ {
			tmp = append(tmp, getRandomFloat(-0.5, 0.5))
		}
		w = append(w, tmp)
	}

	var b []float64
	for j := 0; j < targetSize; j++ {
		b = append(b, getRandomFloat(-0.5, 0.5))
	}
	
	result := synaptic{
		sourceSize: sourceSize,
		targetSize: targetSize,
		weight: weight{
			w: w,
			b: b,
		},
		activation: activation,
	}
	return result, nil
} 

func getRandomFloat(min, max float64) float64 {
	return min + rand.Float64()*(max-min)
}

func mean(vals DataArray) (result float64, err error) {
	if len(vals) == 0 {
		return result, fmt.Errorf("Got error while calculating mean : division by zero")
	}

	var sum float64 
	for _, val := range vals {
		sum += val
	}

	return sum / float64(len(vals)), nil
}

func mergeWeights(a []weight, b []weight) []weight {
	for i := 0; i < len(b); i++ {
		for j := 0; j < len(b[i].w); j++ {
			for k := 0; k < len(b[i].w[j]); k++ {
				a[i].w[j][k] = a[i].w[j][k] + b[i].w[j][k]
			}
		}

		for j := 0; j < len(b[i].b); j++ {
			a[i].b[j] = a[i].b[j] + b[i].b[j]
		}
	}
	
	return a
}