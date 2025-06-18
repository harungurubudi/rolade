package activation

import (
	"reflect"
	"testing"

	"github.com/harungurubudi/rolade/profile"
)

func TestLoadSigmoid(t *testing.T) {
	attr := &profile.Attr{
		Name:  "sigmoid",
		Props: "{}",
	}

	activation, err := Load(attr)
	if err != nil {
		t.Errorf("Error test activation generator : %v", err)
	}

	expectedType := "Sigmoid"
	resultType := reflect.TypeOf(activation).Elem().Name()
	if resultType != expectedType {
		t.Errorf("Error test activation generator : Expected %s, got %s", expectedType, resultType)
	}
}

func TestLoadReLU(t *testing.T) {
	attr := &profile.Attr{
		Name:  "relu",
		Props: "{}",
	}

	activation, err := Load(attr)
	if err != nil {
		t.Errorf("Error test activation generator : %v", err)
	}

	expectedType := "ReLU"
	resultType := reflect.TypeOf(activation).Elem().Name()
	if resultType != expectedType {
		t.Errorf("Error test activation generator : Expected %s, got %s", expectedType, resultType)
	}
}

func TestLoadTanh(t *testing.T) {
	attr := &profile.Attr{
		Name:  "tanh",
		Props: "{}",
	}

	activation, err := Load(attr)
	if err != nil {
		t.Errorf("Error test activation generator : %v", err)
	}

	expectedType := "Tanh"
	resultType := reflect.TypeOf(activation).Elem().Name()
	if resultType != expectedType {
		t.Errorf("Error test activation generator : Expected %s, got %s", expectedType, resultType)
	}
}
