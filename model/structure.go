package model

type (
	Attr struct {
		Name  string `json:"name"`
		Props string `json:"props"`
	}

	Props struct {
		Loss      Attr    `json:"loss"`
		Optimizer Attr    `json:"optimizer"`
		ErrLimit  float64 `json:"err_limit"`
		MaxEpoch  int     `json:"max_epoch"`
	}

	Weight struct {
		W [][]float64 `json:"w"`
		B []float64   `json:"b"`
	}

	Synaptic struct {
		SourceSize int    `json:"source_size"`
		TargetSize int    `json:"target_size"`
		Weight     Weight `json:"weight"`
		Activation Attr   `json:"activation"`
	}

	Network struct {
		InputSize  int        `json:"input_size"`
		OutputSize int        `json:"output_size"`
		Props      Props      `json:"props"`
		Synaptics  []Synaptic `json:"synaptics"`
	}
)
