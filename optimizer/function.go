package optimizer

import(
	"fmt"
	"encoding/json"

	"github.com/harungurubudi/rolade/profile"
)

func Generate(attr *profile.Attr) (optimizer IOptimizer, err error) {
	switch attr.Name{
	case "sgd":
		var o SGD
		err := json.Unmarshal([]byte(attr.Props), &o)
		if err != nil {
			return generateFailed(err)
		}
		optimizer = &o
	default:
		return nil, fmt.Errorf("Optimizer function not found")
	}

	return optimizer, nil
}

func generateFailed(err error) (IOptimizer, error) {
	return nil, fmt.Errorf("Got error while generate optimizer function: %v", err)
}