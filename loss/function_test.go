package loss

import (
	"reflect"
	"testing"

	"github.com/harungurubudi/rolade/model"
)

func TestLoadRMSE(t *testing.T) {
	attr := &model.Attr{
		Name:  "rmse",
		Props: "{}",
	}

	loss, err := Load(attr)
	if err != nil {
		t.Errorf("Error test loss generator : %v", err)
	}

	expectedType := "RMSE"
	resultType := reflect.TypeOf(loss).Elem().Name()
	if resultType != expectedType {
		t.Errorf("Error test loss generator : Expected %s, got %s", expectedType, resultType)
	}
}
