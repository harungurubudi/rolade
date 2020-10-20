package network

import (
	"fmt"
	"math/rand"
)

func generateWeight(sourceSize int, targetSize int) weightset {
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
	}
	return result
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