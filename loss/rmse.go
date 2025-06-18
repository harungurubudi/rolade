package loss

import (
	"math"
)

// RMSE (Root Mean Squared Error) is a loss function that measures the
// square root of the average of the squared differences between predicted
// values and actual targets. It is commonly used in regression problems
// and penalizes larger errors more than smaller ones, making it sensitive
// to outliers.
type RMSE struct{}

func (l *RMSE) Calculate(errs []float64) (result float64) {
	var sum float64

	// If errs is empty, it returns 0 top prevent division by zero
	if len(errs) == 0 {
		return 0
	}

	for _, err := range errs {
		sum += (err * err)
	}

	return math.Sqrt(
		sum / float64(len(errs)),
	)
}

func (l *RMSE) CallMe() string {
	return "rmse"
}
