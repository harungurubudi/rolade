package network

import "errors"

var (
	ErrSamplesLengthMismatch = errors.New("features and targets must have the same length")
)

type Vector []float64

type Sample struct {
	Feature Vector
	Target  Vector
}

type Samples []Sample

func NewSamples(features []Vector, targets []Vector) (result Samples, err error) {
	if len(features) != len(targets) {
		return result, ErrSamplesLengthMismatch
	}

	for index, feature := range features {
		result = append(result, Sample{
			Feature: feature,
			Target:  targets[index],
		})
	}

	return result, err
}

// Len returns the number of samples in the dataset.
//
// This is commonly used to determine batch sizes or iterate over the dataset.
func (l Samples) Len() int {
	return len(l)
}

func (l Samples) Split(batchSize int) (batches []Samples) {
	if batchSize <= 0 {
		return nil
	}

	for i := 0; i < l.Len(); i += batchSize {
		end := i + batchSize
		batches = append(batches, l[i:end])
	}
	return batches
}
