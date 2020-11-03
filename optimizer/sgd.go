package optimizer

type SGD struct {
	Alpha float64 
	M float64
	IsNesterov bool
	V float64
}

func NewSGD() *SGD {
	o := &SGD{}
	o.initialize()
	return o
}

func (o *SGD) CalculateDelta(grad float64) (float64) {
	var delta float64
	if o.M == 0 {
		delta = o.Alpha * grad 			 
	} else {
		o.V = o.M * o.V + o.Alpha * grad
		if o.IsNesterov {
			delta = o.V
		} else {
			delta = o.M * o.V + o.Alpha * grad
		}
	}

	return delta
}

func (o *SGD) initialize() {
	if o.Alpha == 0 {
		o.Alpha = float64(0.001)
	}
}

func (o *SGD) CallMe() string {
	return "sgd"
}