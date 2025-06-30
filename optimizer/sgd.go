package optimizer

// SGD (Stochastic Gradient Descent) is a basic optimization algorithm
// that updates model parameters by moving in the direction of the negative gradient
// of the loss function. It supports momentum and optional Nesterov acceleration.
//
// Fields:
//   - Alpha: Learning rate, controls the step size in each iteration.
//   - M: Momentum factor, helps accelerate gradients in the right direction.
//   - IsNesterov: If true, applies Nesterov accelerated gradient.
//   - V: Internal velocity term used for momentum calculation.
type SGD struct {
	Alpha      float64
	Momentum   float64
	IsNesterov bool
	Velocity   float64
}

func NewSGD() *SGD {
	o := &SGD{}
	o.initialize()
	return o
}

func NewSGDWithLearningRate(alpha float64) *SGD {
	sgd := NewSGD()
	sgd.Alpha = alpha
	return sgd
}

func (o *SGD) CalculateDelta(grad float64) float64 {
	var delta float64
	if o.Momentum == 0 {
		delta = o.Alpha * grad
	} else {
		o.Velocity = o.Momentum*o.Velocity + o.Alpha*grad
		if o.IsNesterov {
			delta = o.Velocity
		} else {
			delta = o.Momentum*o.Velocity + o.Alpha*grad
		}
	}

	return delta
}

func (o *SGD) initialize() {
	if o.Alpha == 0 {
		o.Alpha = float64(0.01)
	}
}

func (o *SGD) CallMe() string {
	return "sgd"
}
