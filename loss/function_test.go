package loss

import (
	"reflect"
	"testing"

	"github.com/harungurubudi/rolade/profile"
)

func TestGenerateRMSE(t *testing.T) {
	attr := &profile.Attr{
		Name: "rmse",
		Props: "{}",
	}

	loss, err := Generate(attr)
	if err != nil { 
		t.Errorf("Error test loss generator : %v", err)
	}

	expectedType := "RMSE"
	resultType := reflect.TypeOf(loss).Elem().Name()
	if resultType != expectedType {
		t.Errorf("Error test loss generator : Expected %s, got %s", expectedType, resultType)
	}
}

