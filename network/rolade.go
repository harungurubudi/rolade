package network

import (
	"os"
	"log"
	"fmt"
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

	weightset struct {
		sourceSize int
		targetSize int
		weight [][]float64
		bias []float64
		activation activation.IActivation
	}
	
	Network struct {
		inputSize int
		outputSize int
		props Props
		weights []weightset 
	}

	// Deltas
	deltaSet struct {
		weight [][]float64
		bias []float64
	}

	deltas []deltaSet
)

func (nt *Network) Save(path string) (err error) {
	var w []profile.Weightset
	for _, wit := range nt.weights {
		ajs, err := json.Marshal(wit.activation) 
		if err != nil {
			return fmt.Errorf("Got error while marshalling activation: %v", err)
		}
		w = append(w, profile.Weightset{
			SourceSize: wit.sourceSize,
			TargetSize: wit.targetSize,
			Weight: wit.weight,
			Bias: wit.bias,
			Activation: profile.Attr{
				Name: wit.activation.CallMe(),
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
		Weights: w,
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

// AddLayer - Add single hidden layer
func (nt *Network) AddLayer(size int, activation activation.IActivation) error {
	wLen := len(nt.weights)
	if wLen > 0 {
		lastWeight := nt.weights[wLen - 1]
		w, err := generateWeight(lastWeight.sourceSize, size, lastWeight.activation)
		if err != nil {
			return fmt.Errorf("Got error while add layer: %v", err)
		}
		nt.weights[wLen - 1] = w
	}

	w, err := generateWeight(size, nt.outputSize, activation)
	if err != nil {
		return fmt.Errorf("Got error while add layer: %v", err)
	}
	nt.weights = append(nt.weights, w)
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
	for i := 0; i < len(nt.weights); i++ {
		input, err = nt.propagate(input, i)
		if (err != nil) {
			return nil, err
		}
		output = append(output, input)
	}
	
	return output, nil
}

func (nt *Network) propagate(input DataArray, weightIndex int) (DataArray, error) {
	weight := nt.weights[weightIndex]
	inputSize := len(input)
	if inputSize != weight.sourceSize {
		return nil, fmt.Errorf("Propagate input doesn't fit in weight number %d. Expect %d nodes, but got %d nodes", weightIndex, weight.sourceSize, inputSize)
	}
	output := make(DataArray, weight.targetSize)
	for j := range output {
		var sum float64
		for i := 0; i < weight.sourceSize; i++ {
			sum += weight.weight[i][j] * input[i]
		}
		sum += weight.bias[j]
		output[j] = weight.activation.Activate(sum)
	}

	return output, nil
}

// Train - Train neural network
func (nt *Network) Train(inputs []DataArray, targets []DataArray) (error) {
	inputLength := len(inputs)
	targetLength := len(targets)
	if inputLength != targetLength {
		return fmt.Errorf("Input length should be same with target's. Have %d inputs, but %d output", inputLength, targetLength)
	}

	checkPointBatch := nt.props.MaxEpoch / 20
	var lastLoss float64
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
	log.Printf("Training finished due maximum epoch (%d) has been reached at loss (%s) : %f\n", nt.props.MaxEpoch, nt.props.Loss.CallMe(), lastLoss)
	return nil
}

func (nt *Network) trainSet(inputs []DataArray, targets []DataArray) (errMean DataArray, err error) {
	for i, input := range inputs {
		outputs, err := nt.forward(input)

		// fmt.Println(outputs)
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
			grad[j] = tErr[j] * nt.weights[len(nt.weights) - 1].activation.Derivate(y)
		}

		d, err := nt.calculateDelta(grad, nodes)
		if err != nil {
			return errMean, fmt.Errorf("Got an error while train with data %d : %v", i, err)
		}

		nt.updateWeight(d)
		tmpTErr, err := mean(tErr)
		if err != nil {
			return errMean, fmt.Errorf("Got an error while train with data %d : %v", i, err)
		} 
		errMean = append(errMean, tmpTErr)
	}

	return errMean, nil
} 

func (nt *Network) calculateDelta(grad DataArray, nodes []DataArray) (deltas, error) {
	result := make([]deltaSet, len(nodes))
	for i := len(nodes) - 1; i >= 0; i-- {
		newGrad, d, err := nt.backPropagate(grad, nodes[i], nt.weights[i])
		if err != nil {
			return nil, fmt.Errorf("Error calculating delta : %v", err)
		}
		result[i] = d
		grad = newGrad
	}

	return deltas(result), nil
}

func (nt *Network) backPropagate(grad DataArray, node DataArray, wset weightset) (DataArray, deltaSet, error) {
	var wDelta [][]float64
	newGrad := make(DataArray, len(node))

	for i := 0; i < len(node); i++ {
		var tErrLocalSum float64
		var wDeltaLocal []float64
		for j := 0; j < len(grad); j++ {
			tErrLocalSum += (grad[j] * wset.weight[i][j])
			wDeltaLocal = append(wDeltaLocal, nt.props.Optimizer.CalculateDelta(node[i] * grad[j]))
		} 
		newGrad[i] = tErrLocalSum * wset.activation.Derivate(node[i])
		wDelta = append(wDelta, wDeltaLocal)
	}

	var bDelta []float64
	for j := 0; j < len(grad); j++ {
		bDelta = append(bDelta, nt.props.Optimizer.CalculateDelta(grad[j]))
	}

	dSet := deltaSet{
		weight: wDelta,
		bias: bDelta,
	}

	return newGrad, dSet, nil
}

func (nt *Network) updateWeight(d deltas) error {
	for i := 0; i < len(d); i++ {
		for j := 0; j < len(d[i].weight); j++ {
			for k := 0; k < len(d[i].weight[j]); k++ {
				nt.weights[i].weight[j][k] = nt.weights[i].weight[j][k] + d[i].weight[j][k]
			}
		}

		for j := 0; j < len(d[i].bias); j++ {
			nt.weights[i].bias[j] = nt.weights[i].bias[j] + d[i].bias[j]
		}
	}
	
	return nil
}

