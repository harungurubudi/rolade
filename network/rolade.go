package network

import (
	"os"
	"log"
	"fmt"
	"time"
	"encoding/json"
	
	"github.com/harungurubudi/rolade/activation"
	"github.com/harungurubudi/rolade/loss"
	"github.com/harungurubudi/rolade/optimizer"
	"github.com/harungurubudi/rolade/profile"
)

type DataArray []float64

type (
	// Network
	Props struct {
		Loss       loss.ILoss
		Optimizer  optimizer.IOptimizer
		ErrLimit   float64
		MaxEpoch   int
	}

	weight struct {
		w [][]float64
		b []float64
	}

	synaptic struct {
		sourceSize int
		targetSize int
		weight weight
		activation activation.IActivation
	}
	
	Network struct {
		inputSize int
		outputSize int
		props Props
		synaptics []synaptic 
	}

	deltas []weight
)

// AddLayer - Add single hidden layer
func (nt *Network) AddLayer(size int, activation activation.IActivation) error {
	netSize := len(nt.synaptics)
	if netSize > 0 {
		lastLayer := nt.synaptics[netSize - 1]
		sy, err := generateSynaptic(lastLayer.sourceSize, size, lastLayer.activation)
		if err != nil {
			return fmt.Errorf("Got error while add layer: %v", err)
		}
		nt.synaptics[netSize - 1] = sy
	}

	sy, err := generateSynaptic(size, nt.outputSize, activation)
	if err != nil {
		return fmt.Errorf("Got error while add layer: %v", err)
	}
	nt.synaptics = append(nt.synaptics, sy)
	return nil
}

// SetProps - Self defined
func (nt *Network) SetProps(props Props) {
	if props.Loss != nil {
		nt.props.Loss = props.Loss
	}
	if props.Optimizer != nil {
		nt.props.Optimizer = props.Optimizer
	}
	if props.ErrLimit != float64(0) {
		nt.props.ErrLimit = props.ErrLimit
	}
	if props.MaxEpoch != 0 {
		nt.props.MaxEpoch = props.MaxEpoch
	}
}

// Test neural network
func (nt *Network) Test(input DataArray) (DataArray, []int, error) {
	output, err := nt.forward(input)
	if err != nil {
		return nil, nil, err
	}

	var result []int
	for _, item := range output[len(output) - 1] {
		if item > 0.5 {
			result = append(result, 1)
		} else {
			result = append(result, 0)
		}
	}
	
	return output[len(output) - 1], result, nil
}

// forward - Do single forward feed
func (nt *Network) forward(input DataArray) ([]DataArray, error) {
	inputSize := len(input)
	if inputSize != nt.inputSize {
		return nil, fmt.Errorf("Feature input doesn't fit in network feature size. Expect %d nodes, but got %d nodes", nt.inputSize, inputSize)
	}
	
	var err error
	var output []DataArray
	for i := 0; i < len(nt.synaptics); i++ {
		input, err = nt.propagate(input, i)
		if (err != nil) {
			return nil, err
		}
		output = append(output, input)
	}
	
	return output, nil
}

func (nt *Network) propagate(input DataArray, synapticIndex int) (DataArray, error) {
	inputSize := len(input)
	if inputSize != nt.synaptics[synapticIndex].sourceSize {
		return nil, fmt.Errorf("Propagate input doesn't fit. Expect %d nodes, but got %d nodes", nt.synaptics[synapticIndex].sourceSize, inputSize)
	}
	output := make(DataArray, nt.synaptics[synapticIndex].targetSize)
	for j := range output {
		var sum float64
		for i := 0; i < nt.synaptics[synapticIndex].sourceSize; i++ {
			sum += nt.synaptics[synapticIndex].weight.w[i][j] * input[i]
		}
		sum += nt.synaptics[synapticIndex].weight.b[j]
		output[j] = nt.synaptics[synapticIndex].activation.Activate(sum)
	}

	return output, nil
}

// Train - Train neural network
func (nt *Network) Train(inputs []DataArray, targets []DataArray) (error) {
	inputSize := len(inputs)
	targetSize := len(targets)
	if inputSize != targetSize {
		return fmt.Errorf("Input size should be same with target's. Have %d inputs, but %d output", inputSize, targetSize)
	}

	checkPointBatch := nt.props.MaxEpoch / 20
	var lastLoss float64
	start := time.Now()
	for epoch := 0; epoch < nt.props.MaxEpoch; epoch++ {
		errMean, err := nt.trainSet(inputs, targets)
		if err != nil {
			return fmt.Errorf("Got error while training in epoch %d : %v", epoch, err)
		}

		loss := nt.props.Loss.Calculate(errMean)
		lastLoss = loss
		
		if epoch % checkPointBatch == 0 {
			log.Printf("Epoch %d got loss (%s) : %f\n", epoch, nt.props.Loss.CallMe(), loss)
		}

		if loss <= nt.props.ErrLimit {
			log.Printf("Training finished at epoch %d due minimum error has been reached at loss (%s) : %f\n", epoch, nt.props.Loss.CallMe(), loss)
			return nil
		}
	}
	end := time.Now()
	duration := end.Sub(start)
	log.Printf("Training finished in %s due maximum epoch (%d) has been reached at loss (%s) : %f\n", duration.String(), nt.props.MaxEpoch, nt.props.Loss.CallMe(), lastLoss)
	return nil
}

func (nt *Network) trainSet(inputs []DataArray, targets []DataArray) (errMean DataArray, err error) {
	for i, input := range inputs {
		outputs, err := nt.forward(input)

		if err != nil {
			return errMean, fmt.Errorf("Got an error while train with data %d : %v", i, err)
		}

		// Shift nodes
		nodes := []DataArray{input}
		nodes = append(nodes, outputs[:len(outputs) - 1]...)

		// Count error : expected target - final output
		tErr := make(DataArray, len(targets[i]))
		grad := make(DataArray, len(targets[i]))

		for j, n := range targets[i] {
			y := outputs[len(outputs) - 1][j]
			tErr[j] = n - y
			grad[j] = tErr[j] * nt.synaptics[len(nt.synaptics) - 1].activation.Derivate(y)
		}

		delta, err := nt.calculateDelta(grad, nodes)
		if err != nil {
			return errMean, fmt.Errorf("Got an error while train with data %d : %v", i, err)
		}

		nt.updateWeight(delta)
		tmpTErr, err := mean(tErr)
		if err != nil {
			return errMean, fmt.Errorf("Got an error while train with data %d : %v", i, err)
		} 
		errMean = append(errMean, tmpTErr)
	}

	return errMean, nil
} 

func (nt *Network) calculateDelta(grad DataArray, nodes []DataArray) (deltas, error) {
	result := make([]weight, len(nodes))
	for i := len(nodes) - 1; i >= 0; i-- {
		newGrad, d, err := nt.backPropagate(grad, nodes[i], nt.synaptics[i])
		if err != nil {
			return nil, fmt.Errorf("Error calculating delta : %v", err)
		}
		result[i] = d
		grad = newGrad
	}

	return deltas(result), nil
}

func (nt *Network) backPropagate(grad DataArray, node DataArray, sy synaptic) (DataArray, weight, error) {
	var wDelta [][]float64
	newGrad := make(DataArray, len(node))

	for i := 0; i < len(node); i++ {
		var tErrLocalSum float64
		var wDeltaLocal []float64
		for j := 0; j < len(grad); j++ {
			tErrLocalSum += (grad[j] * sy.weight.w[i][j])
			wDeltaLocal = append(wDeltaLocal, nt.props.Optimizer.CalculateDelta(node[i] * grad[j]))
		} 
		newGrad[i] = tErrLocalSum * sy.activation.Derivate(node[i])
		wDelta = append(wDelta, wDeltaLocal)
	}

	var bDelta []float64
	for j := 0; j < len(grad); j++ {
		bDelta = append(bDelta, nt.props.Optimizer.CalculateDelta(grad[j]))
	}

	dSet := weight{
		w: wDelta,
		b: bDelta,
	}

	return newGrad, dSet, nil
}

func (nt *Network) updateWeight(d deltas) {
	for i := 0; i < len(d); i++ {
		for j := 0; j < len(d[i].w); j++ {
			for k := 0; k < len(d[i].w[j]); k++ {
				nt.synaptics[i].weight.w[j][k] = nt.synaptics[i].weight.w[j][k] + d[i].w[j][k]
			}
		}

		for j := 0; j < len(d[i].b); j++ {
			nt.synaptics[i].weight.b[j] = nt.synaptics[i].weight.b[j] + d[i].b[j]
		}
	}
}

func (nt *Network) Save(path string) (err error) {
	var sy []profile.Synaptic
	for _, sySource := range nt.synaptics {
		ajs, err := json.Marshal(sySource.activation) 
		if err != nil {
			return fmt.Errorf("Got error while marshalling activation: %v", err)
		}
		sy = append(sy, profile.Synaptic{
			SourceSize: sySource.sourceSize,
			TargetSize: sySource.targetSize,
			Weight: profile.Weight{
				W: sySource.weight.w,
				B: sySource.weight.b,
			},
			Activation: profile.Attr{
				Name: sySource.activation.CallMe(),
				Props: string(ajs),
			},
		})
	}

	ljs, err := json.Marshal(nt.props.Loss) 
	if err != nil {
		return fmt.Errorf("Got error while marshalling loss: %v", err)
	}

	ojs, err := json.Marshal(nt.props.Optimizer) 
	if err != nil {
		return fmt.Errorf("Got error while marshalling optimizer: %v", err)
	}

	r := profile.Network{
		InputSize: nt.inputSize,
		OutputSize: nt.outputSize,
		Synaptics: sy,
		Props: profile.Props{
			Loss: profile.Attr{
				Name: nt.props.Loss.CallMe(),
				Props: string(ljs),
			},
			Optimizer: profile.Attr{
				Name: nt.props.Optimizer.CallMe(),
				Props: string(ojs),
			},
			ErrLimit: nt.props.ErrLimit,
			MaxEpoch: nt.props.MaxEpoch,
		},
	}

	b, err := json.Marshal(r)
	if err != nil {
		return fmt.Errorf("Got error while marshalling model: %v", err)
	}

	f, err := os.Create(path + "/rolade.profile")
	if err != nil {
		return fmt.Errorf("Got error while creating model file: %v", err)
	}
	defer f.Close()

	_, err = f.Write(b)
	if err != nil {
		return fmt.Errorf("Got error while writing model to file: %v", err)
	}
	return nil
}