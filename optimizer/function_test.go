package optimizer

import (
	"reflect"
	"testing"

	"github.com/harungurubudi/rolade/profile"
)

func TestGenerateSGD(t *testing.T) {
	attr := &profile.Attr{
		Name:  "sgd",
		Props: `{"Alpha": 0.001, "Momentum": 0, "IsNesterov": false, "Velocity": 0}`,
	}

	optimizer, err := Generate(attr)
	if err != nil {
		t.Errorf("Error test optimizer generator : %v", err)
	}

	expectedType := "SGD"
	resultType := reflect.TypeOf(optimizer).Elem().Name()
	if resultType != expectedType {
		t.Errorf("Error test optimizer generator : Expected %s, got %s", expectedType, resultType)
	}
}
