package loss

import (
	"math"
)

type RMSE struct{}

func (l *RMSE) Calculate(errs []float64) (result float64) {
	var sum float64 
	for _, err := range errs {
		sum += math.Pow(err, 2)
	}

	return math.Sqrt(
		sum / float64(len(errs)),
	)
}

func (l *RMSE) CallMe() string {
	return "RMSE"
}