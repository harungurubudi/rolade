package loss

import (
	"testing"
	"math"
)

func getExpectedErrors() (errors []float64) {
	return []float64{
		0.01, 
		0.02, 
		0.34, 
		0.72,
	}
}

func TestCalculate(t *testing.T) {
	expected := 0.3983

	errors := getExpectedErrors()
	loss := RMSE{}
	got := loss.Calculate(errors)

	if math.Abs(got - expected) > 0.0001 {
		t.Errorf("Expected %f, get %f", expected, got)
	}
}
