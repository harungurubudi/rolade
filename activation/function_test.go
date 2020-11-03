package activation

import (
	"reflect"
	"testing"

	"github.com/harungurubudi/rolade/profile"
)

func TestGenerateSigmoid(t *testing.T) {
	attr := &profile.Attr{
		Name: "sigmoid",
		Props: "{}",
	}

	activation, err := Generate(attr)
	if err != nil { 
		t.Errorf("Error test activation generator : %v", err)
	}

	expectedType := "Sigmoid"
	resultType := reflect.TypeOf(activation).Elem().Name()
	if resultType != expectedType {
		t.Errorf("Error test activation generator : Expected %s, got %s", expectedType, resultType)
	}
}

func TestGenerateRelu(t *testing.T) {
	attr := &profile.Attr{
		Name: "relu",
		Props: "{}",
	}

	activation, err := Generate(attr)
	if err != nil { 
		t.Errorf("Error test activation generator : %v", err)
	}

	expectedType := "Relu"
	resultType := reflect.TypeOf(activation).Elem().Name()
	if resultType != expectedType {
		t.Errorf("Error test activation generator : Expected %s, got %s", expectedType, resultType)
	}
}

func TestGenerateTanh(t *testing.T) {
	attr := &profile.Attr{
		Name: "tanh",
		Props: "{}",
	}

	activation, err := Generate(attr)
	if err != nil { 
		t.Errorf("Error test activation generator : %v", err)
	}

	expectedType := "Tanh"
	resultType := reflect.TypeOf(activation).Elem().Name()
	if resultType != expectedType {
		t.Errorf("Error test activation generator : Expected %s, got %s", expectedType, resultType)
	}
}

