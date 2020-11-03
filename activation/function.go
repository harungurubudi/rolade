package activation

import(
	"fmt"
	"encoding/json"

	"github.com/harungurubudi/rolade/profile"
)

func Generate(attr *profile.Attr) (activation IActivation, err error) {
	switch attr.Name{
	case "sigmoid":
		var a Sigmoid
		err := json.Unmarshal([]byte(attr.Props), &a)
		if err != nil {
			return generateFailed(err)
		}
		activation = &a
	case "relu":
		var a Relu
		err := json.Unmarshal([]byte(attr.Props), &a)
		if err != nil {
			return generateFailed(err)
		}
		activation = &a
	case "tanh":
		var a Tanh
		err := json.Unmarshal([]byte(attr.Props), &a)
		if err != nil {
			return generateFailed(err)
		}
		activation = &a
	default:
		return nil, fmt.Errorf("Activation function not found")
	}

	return activation, nil
}

func generateFailed(err error) (IActivation, error) {
	return nil, fmt.Errorf("Got error while generate activation function: %v", err)
}