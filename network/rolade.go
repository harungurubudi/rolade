package network

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"sync"

	"github.com/harungurubudi/rolade/activation"
	"github.com/harungurubudi/rolade/loss"
	"github.com/harungurubudi/rolade/model"
	"github.com/harungurubudi/rolade/optimizer"
)

const asyncProcessThreshold = 128

type (
	// Props defines the configuration for training the neural network,
	// including the loss function, optimizer, error limit, and maximum number of epochs.
	Props struct {
		Loss      loss.ILoss
		Optimizer optimizer.IOptimizer
		ErrLimit  float64
		MaxEpoch  int
		Patience  int
	}

	// weight contains the weights and biases of a layer in the neural network.
	weight struct {
		weight [][]float64
		bias   []float64
	}

	// synaptic defines a layer's structure, including the number of input/output neurons,
	// weights, and activation function used during forward and backward passes.
	synaptic struct {
		sourceSize int
		targetSize int
		weight     weight
		activation activation.IActivation
	}

	// Network represents a feedforward neural network composed of fully connected layers.
	// It maintains the structure of the network (input/output sizes, layer weights, activations)
	// and provides methods for forward propagation, backpropagation, and training.
	Network struct {
		inputSize     int
		outputSize    int
		props         Props
		synaptics     []synaptic
		lossHistories []float64
	}

	// deltas is a slice of weight updates used during backpropagation.
	deltas []weight
)

// AddLayer - Add single hidden layer
func (nt *Network) AddLayer(size int, activation activation.IActivation) error {
	netSize := len(nt.synaptics)
	if netSize > 0 {
		lastLayer := nt.synaptics[netSize-1]
		sy, err := generateSynaptic(lastLayer.sourceSize, size, lastLayer.activation)
		if err != nil {
			return fmt.Errorf("got error while add layer: %v", err)
		}
		nt.synaptics[netSize-1] = sy
	}

	sy, err := generateSynaptic(size, nt.outputSize, activation)
	if err != nil {
		return fmt.Errorf("got error while add layer: %v", err)
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
	if props.Patience != 0 {
		nt.props.Patience = props.Patience
	}
}

// Test neural network
func (nt *Network) Test(input Vector) (Vector, []int, error) {
	output, err := nt.forward(input)
	if err != nil {
		return nil, nil, err
	}

	var result []int
	for _, item := range output[len(output)-1] {
		if item > 0.5 {
			result = append(result, 1)
		} else {
			result = append(result, 0)
		}
	}

	return output[len(output)-1], result, nil
}

// forward performs a full forward pass through the network given an input vector.
// It returns the output of each layer (including the final output layer) as a slice of Vector,
// or an error if the input size does not match the expected input size of the network.
func (nt *Network) forward(input Vector) (layerActivations []Vector, err error) {
	layerActivations = make([]Vector, 0, len(nt.synaptics))
	for i := 0; i < len(nt.synaptics); i++ {
		input, err = nt.propagate(input, i)
		if err != nil {
			return nil, err
		}
		layerActivations = append(layerActivations, input)
	}

	return layerActivations, nil
}

// propagate computes the output of a single layer (synaptic connection) in the network.
// It applies the layer's weights, biases, and activation function to the input vector,
// returning the resulting output vector or an error if the input size does not match
// the expected number of source nodes.
func (nt *Network) propagate(input Vector, synapticIndex int) (Vector, error) {
	sy := nt.synaptics[synapticIndex]
	if len(input) != sy.sourceSize {
		return nil, fmt.Errorf("propagate input doesn't fit. Expect %d nodes, but got %d nodes", sy.sourceSize, len(input))
	}

	result := make(Vector, sy.targetSize)
	weights := sy.weight.weight
	biases := sy.weight.bias
	activationFn := sy.activation

	if sy.targetSize > asyncProcessThreshold {
		var wg sync.WaitGroup
		for j := range result {
			wg.Add(1)
			go func(j int) {
				defer wg.Done()
				result[j] = nt.computeNeuronActivation(input, weights, biases, j, activationFn)
			}(j) // <- pass j explicitly
		}
		wg.Wait()
	} else {
		for j := range result {
			result[j] = nt.computeNeuronActivation(input, weights, biases, j, activationFn)
		}
	}

	return result, nil
}

// computeNeuronActivation calculates the output of a single neuron.
//
// It performs a weighted sum of inputs, adds the bias, and applies the activation function.
//
// Parameters:
//   - input: the input feature vector
//   - weights: 2D slice of weights for the layer
//   - biases: bias terms for the layer
//   - j: the index of the output neuron
//   - act: the activation function
//
// Returns:
//   - the activation output of neuron j
func (nt *Network) computeNeuronActivation(input Vector, weights [][]float64, biases []float64, j int, act activation.IActivation) float64 {
	var sum float64
	for i := 0; i < len(input); i++ {
		sum += weights[i][j] * input[i]
	}
	sum += biases[j]
	return act.Activate(sum)
}

// Train runs the training process over the given input and target data using the configured
// optimizer and loss function. It trains for a maximum number of epochs or until the loss
// falls below the configured error limit (ErrLimit).
//
// During training, it performs forward and backward passes for each data pair,
// applies optimizer updates, and logs progress at checkpoints.
//
// Returns an error if the input and target sizes do not match, or if an error occurs during training.
func (nt *Network) Train(samples Samples) error {
	var bestLoss = math.MaxFloat64
	var epochsWithoutImprovement = 0
	for epoch := 0; epoch < nt.props.MaxEpoch; epoch++ {
		errMean, err := nt.trainEpoch(samples)
		if err != nil {
			return err
		}

		loss := nt.props.Loss.Calculate(errMean)
		nt.lossHistories = append(nt.lossHistories, loss)

		if nt.props.MaxEpoch > 20 && epoch%(nt.props.MaxEpoch/20) == 0 {
			log.Printf("Training in epoch %d with loss: %f\n", epoch, loss)
		}

		// Implements early stopping logic to terminate training when no improvement occurs.
		//
		// This mechanism tracks the best loss value seen so far and counts how many consecutive epochs
		// have passed without an improvement (i.e., a lower loss value). If the number of such epochs
		// exceeds the configured `Patience`, training stops early to prevent overfitting or wasted computation.
		//
		// Additionally, if the loss drops below the predefined error threshold (`ErrLimit`), training
		// also stops immediately.
		//
		// - bestLoss stores the lowest loss observed so far.
		// - epochsWithoutImprovement counts how many epochs have passed since the last best loss.
		// - Patience defines the maximum tolerated epochs without improvement before stopping.
		//
		// This strategy improves training efficiency and avoids overfitting by halting when further
		// progress is unlikely.
		if loss < bestLoss {
			bestLoss = loss
			epochsWithoutImprovement = 0
		} else {
			epochsWithoutImprovement++
		}

		if loss <= nt.props.ErrLimit {
			log.Printf("Stopping early: loss (%f) is below threshold", loss)
			return nil
		}

		if epochsWithoutImprovement >= nt.props.Patience {
			log.Printf("Stopping early: no improvement in last %d epochs", nt.props.Patience)
			return nil
		}
	}
	log.Printf("Training completed with final loss: %f", bestLoss)
	return nil
}

// trainEpoch performs one full training epoch over the provided dataset.
//
// The dataset is split into batches which are processed in parallel using goroutines.
// Each batch goes through forward and backward propagation, computing local weight deltas and errors.
// After all batches are processed, the resulting deltas are merged and applied to the network weights.
// The function returns a vector of error values (one per sample) or an error if the training fails.
//
// Note:
// - Parallelism is limited to a fixed number of goroutines (maxGoroutines).
// - mergeDeltas ensures stability by averaging gradients across batches.
//
// Parameters:
//   - samples: the full set of training samples for this epoch.
//
// Returns:
//   - errMean: a vector of per-sample error values.
//   - err: any error that occurred during batch training.
func (nt *Network) trainEpoch(samples Samples) (errMean Vector, err error) {
	// TODO: make this constants dynamic
	const maxGoroutines = 10

	batchSize := samples.Len() / maxGoroutines
	if batchSize == 0 {
		batchSize = 1
	}

	batches := samples.Split(batchSize)

	var (
		wg        sync.WaitGroup
		mutex     sync.Mutex
		allErrs   []float64
		allDeltas []deltas
	)

	for _, batch := range batches {
		wg.Add(1)
		go func(batch Samples) {
			defer wg.Done()

			batchErr, batchDelta, err := nt.trainBatch(batch)
			if err != nil {
				return // Could log or collect failed batch info here
			}

			mutex.Lock()
			allErrs = append(allErrs, batchErr...)
			allDeltas = append(allDeltas, batchDelta)
			mutex.Unlock()
		}(batch)
	}

	wg.Wait()

	finalDelta := mergeDeltas(allDeltas)
	nt.updateWeight(finalDelta)

	return allErrs, nil
}

// trainBatch performs training on a single batch of samples.
//
// For each sample in the batch, it performs:
//   - Forward propagation to compute the output.
//   - Error and gradient calculation at the output layer.
//   - Backward propagation to compute weight deltas.
//   - Error accumulation for loss reporting.
//
// All deltas from individual samples are collected and merged into a single delta
// which is returned for later weight updates (outside this function).
//
// Parameters:
//   - batch: a slice of samples representing a mini-batch.
//
// Returns:
//   - errMean: slice of mean errors for each sample in the batch.
//   - delta: merged weight/bias deltas to be applied later.
//   - err: error if training fails at any point in the batch.
func (nt *Network) trainBatch(batch Samples) (errMean []float64, delta deltas, err error) {
	var returnTrainingError = func(index int, err error) error {
		return fmt.Errorf("got an error while train with data %d: %v", index, err)
	}

	var deltasInBatch []deltas
	for i, sample := range batch {
		outputs, err := nt.forward(sample.Feature)
		if err != nil {
			return errMean, delta, returnTrainingError(i, err)
		}

		// Build node layers for backpropagation (input + hidden layers)
		nodes := append([]Vector{sample.Feature}, outputs[:len(outputs)-1]...)

		// Compute output layer error and gradient
		targetError := make(Vector, len(sample.Target))
		grad := make(Vector, len(sample.Target))
		for j, expected := range sample.Target {
			actual := outputs[len(outputs)-1][j]
			targetError[j] = expected - actual
			grad[j] = targetError[j] * nt.synaptics[len(nt.synaptics)-1].activation.Derivate(actual)
		}

		// Calculate deltas for backpropagation
		d, err := nt.calculateDelta(grad, nodes)
		if err != nil {
			return errMean, delta, returnTrainingError(i, err)
		}
		deltasInBatch = append(deltasInBatch, d)

		// Compute and accumulate per-sample error
		tmpTErr, err := mean(targetError)
		if err != nil {
			return errMean, delta, returnTrainingError(i, err)
		}
		errMean = append(errMean, tmpTErr)
	}

	delta = mergeDeltas(deltasInBatch)
	return errMean, delta, nil
}

// MergeDeltas combines multiple deltas (from different batches)
// into a single averaged delta to be applied once.
//
// Each delta corresponds to a layer (len = numLayers)
func mergeDeltas(all []deltas) deltas {
	if len(all) == 0 {
		return nil
	}

	// Initialize merged with deep copy of the first deltas
	merged := make(deltas, len(all[0]))
	for i := range all[0] {
		merged[i] = weight{
			weight: make([][]float64, len(all[0][i].weight)),
			bias:   make([]float64, len(all[0][i].bias)),
		}
		for j := range all[0][i].weight {
			merged[i].weight[j] = make([]float64, len(all[0][i].weight[j]))
		}
	}

	// Accumulate
	for _, d := range all {
		for i, w := range d {
			for j := range w.weight {
				for k := range w.weight[j] {
					merged[i].weight[j][k] += w.weight[j][k]
				}
			}
			for j := range w.bias {
				merged[i].bias[j] += w.bias[j]
			}
		}
	}

	// Average
	n := float64(len(all))
	for i := range merged {
		for j := range merged[i].weight {
			for k := range merged[i].weight[j] {
				merged[i].weight[j][k] /= n
			}
		}
		for j := range merged[i].bias {
			merged[i].bias[j] /= n
		}
	}

	return merged
}

// calculateDelta performs the full backpropagation pass through the network.
//
// Starting from the output gradient, it iterates backward through all layers,
// computing the weight and bias deltas needed for each layer.
//
// Parameters:
//   - grad: the gradient of the loss with respect to the final output layer
//   - nodes: a slice of FeatureVectors representing the activation at each layer,
//     including the original input and all hidden layers (but excluding output)
//
// Returns:
//   - a slice of weight updates (deltas), one per layer
//   - an error if backpropagation fails at any layer
func (nt *Network) calculateDelta(prevGradient Vector, nodes []Vector) (deltas, error) {
	deltaList := make([]weight, len(nodes))
	for i := len(nodes) - 1; i >= 0; i-- {
		// Backpropagate current layer, accumulate delta and update gradient for previous layer
		outputGradient, d, err := nt.backPropagate(prevGradient, nodes[i], nt.synaptics[i])
		if err != nil {
			return nil, err
		}
		deltaList[i] = d
		prevGradient = Vector(outputGradient)
	}

	return deltas(deltaList), nil
}

// backPropagate computes the gradient for the previous layer and the weight updates (deltas)
// for the current synaptic layer. It applies the derivative of the activation function
// and uses the optimizer to determine the magnitude of weight updates.
//
// Returns the new gradient, the calculated weight deltas, or an error if the process fails.
func (nt *Network) backPropagate(prevGradient Vector, node Vector, sy synaptic) (Vector, weight, error) {
	var weightDelta [][]float64
	outputGradient := make(Vector, len(node))

	for i := range node {
		var tErrLocalSum float64
		var weightDeltaLocal []float64
		for j := range prevGradient {
			tErrLocalSum += (prevGradient[j] * sy.weight.weight[i][j])
			weightDeltaLocal = append(weightDeltaLocal, nt.props.Optimizer.CalculateDelta(node[i]*prevGradient[j]))
		}
		outputGradient[i] = tErrLocalSum * sy.activation.Derivate(node[i])
		weightDelta = append(weightDelta, weightDeltaLocal)
	}

	var biasDelta []float64
	for j := range prevGradient {
		biasDelta = append(biasDelta, nt.props.Optimizer.CalculateDelta(prevGradient[j]))
	}

	dSet := weight{
		weight: weightDelta,
		bias:   biasDelta,
	}

	return outputGradient, dSet, nil
}

// updateWeight applies the provided deltas to the synaptic weights and biases of the network.
//
// Each delta contains the computed changes for a specific layer's weight matrix and bias vector,
// which are added to the corresponding parameters in the network's synaptics.
//
// Parameters:
//   - d: a slice of weight deltas, one for each layer in the network
func (nt *Network) updateWeight(d deltas) {
	if len(d) == 0 {
		return
	}

	// Parallel update if network is large enough
	if len(d[0].bias) >= asyncProcessThreshold {
		var wg sync.WaitGroup
		for i := range d {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				nt.applyDelta(i, d[i])
			}(i)
		}
		wg.Wait()
		return
	}

	// Sequential fallback
	for i := range d {
		nt.applyDelta(i, d[i])
	}
}

// applyDelta applies a single delta to the specified layer
func (nt *Network) applyDelta(i int, delta weight) {
	layer := &nt.synaptics[i].weight
	for j := range delta.weight {
		for k := range delta.weight[j] {
			layer.weight[j][k] += delta.weight[j][k]
		}
	}
	for j := range delta.bias {
		layer.bias[j] += delta.bias[j]
	}
}

// Save serializes the current network configuration, including architecture, weights,
// biases, activations, loss, and optimizer settings into a profile format, and writes it
// to a file at the given path.
//
// The resulting file can later be used for restoring the model's state.
//
// Returns an error if any step in the marshaling or writing process fails.
func (nt *Network) Save(path string) (err error) {
	var sy []model.Synaptic
	for _, sySource := range nt.synaptics {
		ajs, err := json.Marshal(sySource.activation)
		if err != nil {
			return fmt.Errorf("got error while marshalling activation: %v", err)
		}
		sy = append(sy, model.Synaptic{
			SourceSize: sySource.sourceSize,
			TargetSize: sySource.targetSize,
			Weight: model.Weight{
				Weight: sySource.weight.weight,
				Bias:   sySource.weight.bias,
			},
			Activation: model.Attr{
				Name:  sySource.activation.CallMe(),
				Props: string(ajs),
			},
		})
	}

	ljs, err := json.Marshal(nt.props.Loss)
	if err != nil {
		return fmt.Errorf("got error while marshalling loss: %v", err)
	}

	ojs, err := json.Marshal(nt.props.Optimizer)
	if err != nil {
		return fmt.Errorf("got error while marshalling optimizer: %v", err)
	}

	r := model.Network{
		InputSize:  nt.inputSize,
		OutputSize: nt.outputSize,
		Synaptics:  sy,
		Props: model.Props{
			Loss: model.Attr{
				Name:  nt.props.Loss.CallMe(),
				Props: string(ljs),
			},
			Optimizer: model.Attr{
				Name:  nt.props.Optimizer.CallMe(),
				Props: string(ojs),
			},
			ErrLimit: nt.props.ErrLimit,
			MaxEpoch: nt.props.MaxEpoch,
			Patience: nt.props.Patience,
		},
	}

	b, err := json.Marshal(r)
	if err != nil {
		return fmt.Errorf("got error while marshalling model: %v", err)
	}

	f, err := os.Create(path + "/rolade.profile")
	if err != nil {
		return fmt.Errorf("got error while creating model file: %v", err)
	}
	defer f.Close()

	_, err = f.Write(b)
	if err != nil {
		return fmt.Errorf("got error while writing model to file: %v", err)
	}
	return nil
}
