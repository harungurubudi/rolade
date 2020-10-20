package optimizer

type IOptimizer interface {
	CalculateDelta(grad float64) (float64)
}