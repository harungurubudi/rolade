package activation

func Generate(name string, props interface{}) (activation IActivation, err error) {
	switch name{
	case "sigmoid":
		activation, ok := props.(Sigmoid)
		if !ok {
			return nil, fmt.Errorf("Got error while reconstructing activation function")
		}
	case "relu":
		activation, ok := props.(Relu)
		if !ok {
			return nil, fmt.Errorf("Got error while reconstructing activation function")
		}
	case "tanh":
		activation, ok := props.(Tanh)
		if !ok {
			return nil, fmt.Errorf("Got error while reconstructing activation function")
		}
	default:
		return nil, fmt.Errorf("Activation function not found")
	}

	return activation, nil
}