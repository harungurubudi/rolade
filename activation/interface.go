package activation

type IActivation interface {
	Activate(val float64) (result float64)
	Derivate(val float64) (result float64)
}