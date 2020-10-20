package loss

type ILoss interface {
	Calculate(deltas []float64) (result float64)
	CallMe() string
}