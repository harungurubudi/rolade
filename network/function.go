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

func NewNetwork(inputSize int, outputSize int, activation activation.IActivation) (nt *Network, err error) {
	var initW []weightset
	w, err := generateWeight(inputSize, outputSize, activation)
	if err != nil {
		return nt, fmt.Errorf("Got error while add layer: %v", err)
	}

	initW = append(initW, w)

	nt = &Network{
		inputSize: inputSize,
		outputSize: outputSize,
		props: Props{
			Loss:       &loss.RMSE{},
			Optimizer:  optimizer.NewSGD(),
			ErrLimit:  	0.001,
			MaxEpoch: 	1000,
		},
		weights: initW,
	}
	
	return nt, nil
}

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

	var w []weightset
	for _, wit := range pr.Weights {
		a, err := activation.Generate(&wit.Activation)
		if err != nil {
			return nil, fmt.Errorf("Got error while load model: %v", err)
		}

		w = append(w, weightset{
			sourceSize: wit.SourceSize,
			targetSize: wit.TargetSize,
			weight: wit.Weight,
			bias: wit.Bias,
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
		weights: w,
	}, nil
}

func generateWeight(sourceSize int, targetSize int, activation activation.IActivation) (w weightset, err error) {
	var weight [][]float64 
	for i := 0; i < sourceSize; i++ {
		var tmp []float64
		for j := 0; j < targetSize; j++ {
			tmp = append(tmp, getRandomFloat(-0.5, 0.5))
		}
		weight = append(weight, tmp)
	}

	var bias []float64
	for j := 0; j < targetSize; j++ {
		bias = append(bias, getRandomFloat(-0.5, 0.5))
	}
	
	result := weightset{
		sourceSize: sourceSize,
		targetSize: targetSize,
		weight: weight,
		bias: bias,
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
