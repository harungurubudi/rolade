package network

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"

	"github.com/harungurubudi/rolade/activation"
	"github.com/harungurubudi/rolade/loss"
	"github.com/harungurubudi/rolade/model"
	"github.com/harungurubudi/rolade/optimizer"
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
		return nt, fmt.Errorf("got error while add layer: %v", err)
	}

	synaptics = append(synaptics, sy)

	nt = &Network{
		inputSize:  inputSize,
		outputSize: outputSize,
		props: Props{
			Loss:      &loss.RMSE{},
			Optimizer: optimizer.NewSGD(0.01),
			ErrLimit:  0.001,
			MaxEpoch:  1000,
			Patience:  1000,
		},
		synaptics: synaptics,
	}

	return nt, nil
}

// Load restores a previously saved neural network model from a profile file.
//
// The file must be located at the provided `path`, and is expected to be named `rolade.profile`.
// It will unmarshal the network structure (synaptic layers, weights, activation functions),
// along with training properties (loss function, optimizer, and training parameters).
//
// Parameters:
//   - path: the directory where the `rolade.profile` file is located.
//
// Returns:
//   - *Network: a pointer to the loaded network instance
//   - error: if any step in reading, unmarshaling, or reconstruction fails
func Load(path string) (*Network, error) {
	// Read the serialized network file
	data, err := os.ReadFile(path + "/rolade.profile")
	if err != nil {
		return nil, fmt.Errorf("error reading model file: %w", err)
	}

	// Unmarshal into the intermediate model.Network struct
	var profile model.Network
	if err := json.Unmarshal(data, &profile); err != nil {
		return nil, fmt.Errorf("error unmarshalling model: %w", err)
	}

	// Reconstruct synaptic layers
	var synaptics []synaptic
	for _, src := range profile.Synaptics {
		act, err := activation.Load(&src.Activation)
		if err != nil {
			return nil, fmt.Errorf("error loading activation: %w", err)
		}

		synaptics = append(synaptics, synaptic{
			sourceSize: src.SourceSize,
			targetSize: src.TargetSize,
			weight: weight{
				weight: src.Weight.Weight,
				bias:   src.Weight.Bias,
			},
			activation: act,
		})
	}

	// Load training configuration
	lossFunc, err := loss.Load(&profile.Props.Loss)
	if err != nil {
		return nil, fmt.Errorf("error loading loss function: %w", err)
	}

	optimizerFunc, err := optimizer.Generate(&profile.Props.Optimizer)
	if err != nil {
		return nil, fmt.Errorf("error loading optimizer: %w", err)
	}

	// Return the fully reconstructed network
	return &Network{
		inputSize:  profile.InputSize,
		outputSize: profile.OutputSize,
		props: Props{
			Loss:      lossFunc,
			Optimizer: optimizerFunc,
			ErrLimit:  profile.Props.ErrLimit,
			MaxEpoch:  profile.Props.MaxEpoch,
			Patience:  profile.Props.Patience,
		},
		synaptics: synaptics,
	}, nil
}

func generateSynaptic(sourceSize int, targetSize int, activation activation.IActivation) (sy synaptic, err error) {
	var w [][]float64
	for range sourceSize {
		var tmp []float64
		for range targetSize {
			tmp = append(tmp, getRandomFloat(-0.5, 0.5))
		}
		w = append(w, tmp)
	}

	var b []float64
	for range targetSize {
		b = append(b, getRandomFloat(-0.5, 0.5))
	}

	result := synaptic{
		sourceSize: sourceSize,
		targetSize: targetSize,
		weight: weight{
			weight: w,
			bias:   b,
		},
		activation: activation,
	}
	return result, nil
}

func getRandomFloat(min, max float64) float64 {
	return min + rand.Float64()*(max-min)
}

func mean(vals Vector) (result float64, err error) {
	if len(vals) == 0 {
		return result, fmt.Errorf("got error while calculating mean : division by zero")
	}

	var sum float64
	for _, val := range vals {
		sum += val
	}

	return sum / float64(len(vals)), nil
}
