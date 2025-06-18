package activation

import (
	"fmt"
	"strings"

	"github.com/harungurubudi/rolade/profile"
)

// registry maps activation function names to their constructor functions.
// Each constructor takes a serialized JSON string of activation-specific
// properties and returns an IActivation implementation. It is used to
// reconstruct activation functions from profile metadata.
var registry = map[string]func(string) (IActivation, error){
	"relu":    func(_ string) (IActivation, error) { return &ReLU{}, nil },
	"sigmoid": func(_ string) (IActivation, error) { return &Sigmoid{}, nil },
	"tanh":    func(_ string) (IActivation, error) { return &Tanh{}, nil },
}

// Load creates an activation function instance from a serialized profile attribute.
// It looks up the activation type by name and initializes it using the provided Props field,
// which may contain JSON-encoded parameters required by some activation functions.
//
// Returns an error if the activation type is not supported or if deserialization fails.

func Load(attr *profile.Attr) (IActivation, error) {
	if gen, ok := registry[strings.ToLower(attr.Name)]; ok {
		return gen(attr.Props)
	}
	return nil, fmt.Errorf("unsupported activation: %s", attr.Name)
}
