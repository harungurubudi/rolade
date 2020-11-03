package profile

type (
	Attr struct {
		Name string `json:"name"`
		Props string `json:"props"`
	}

	Props struct {
		Loss       Attr `json:"loss"`
		Optimizer  Attr `json:"optimizer"`
		ErrLimit   float64 `json:"err_limit"`
		MaxEpoch   int `json:"max_epoch"`
	}
	
	Weightset struct {
		SourceSize int `json:"source_size"`
		TargetSize int `json:"target_size"`
		Weight [][]float64 `json:"weight"`
		Bias []float64 `json:"bias"`
		Activation Attr `json:"activation"`
	}
	
	Network struct {
		InputSize int `json:"input_size"`
		OutputSize int `json:"output_size"`
		Props Props `json:"props"`
		Weights []Weightset  `json:"weights"`
	}
)