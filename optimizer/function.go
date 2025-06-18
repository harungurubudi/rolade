package optimizer

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/harungurubudi/rolade/profile"
)

// registry maps optimizer names to their corresponding constructor functions.
// Each function takes a JSON-encoded string of optimizer properties
// and returns an IOptimizer instance or an error.
var registry = map[string]func(string) (IOptimizer, error){
	"sgd": func(props string) (IOptimizer, error) {
		var o SGD
		err := json.Unmarshal([]byte(props), &o)
		if err != nil {
			return nil, fmt.Errorf("got error while generating optimizer function: %v", err)
		}
		return &o, nil
	},
}

// Generate creates an IOptimizer instance based on the given profile.Attr,
// which includes the optimizer name and its serialized properties.
// It returns an error if the optimizer name is not supported
// or if instantiation fails.
//
// Example:
//
//	attr := &profile.Attr{Name: "sgd", Props: "{\"LearningRate\":0.01}"}
//	optimizer, err := Generate(attr)
func Generate(attr *profile.Attr) (IOptimizer, error) {
	if gen, ok := registry[strings.ToLower(attr.Name)]; ok {
		return gen(attr.Props)
	}
	return nil, fmt.Errorf("unsupported optimizer: %s", attr.Name)
}
