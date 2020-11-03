package loss

import(
	"fmt"
	"encoding/json"

	"github.com/harungurubudi/rolade/profile"
)

func Generate(attr *profile.Attr) (loss ILoss, err error) {
	switch attr.Name{
	case "rmse":
		var l RMSE
		err := json.Unmarshal([]byte(attr.Props), &l)
		if err != nil {
			return generateFailed(err)
		}
		loss = &l
	default:
		return nil, fmt.Errorf("Loss function not found")
	}

	return loss, nil
}

func generateFailed(err error) (ILoss, error) {
	return nil, fmt.Errorf("Got error while generate loss function: %v", err)
}