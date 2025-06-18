package loss

import (
	"fmt"
	"strings"

	"github.com/harungurubudi/rolade/model"
)

// registry maps loss function names to their corresponding constructor functions.
// Each entry defines how to instantiate a loss function from its serialized configuration.
var registry = map[string]func(string) (ILoss, error){
	"rmse": func(_ string) (ILoss, error) { return &RMSE{}, nil },
}

// Load returns an ILoss implementation based on the given profile attribute.
// It looks up the function name in the registry and invokes its constructor.
// If the loss function is not supported, it returns an error.
//
// This is typically used when restoring a model from a saved profile.
func Load(attr *model.Attr) (ILoss, error) {
	if gen, ok := registry[strings.ToLower(attr.Name)]; ok {
		return gen(attr.Props)
	}
	return nil, fmt.Errorf("unsupported loss function: %s", attr.Name)
}
