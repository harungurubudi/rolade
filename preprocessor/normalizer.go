package preprocessor

import (
	"fmt"
	"github.com/harungurubudi/rolade/network"
)

func Normalize(features []network.DataArray) (result []network.DataArray, err error) {
	if len(features) == 0 {
		return result, fmt.Errorf("Feature's length is zero")
	}

	fragments := make([]network.DataArray, len(features[0]))

	// Shard to fragments
	for _, item := range features {
		if len(item) != len(fragments) {
			return result, fmt.Errorf("Feature's size is not constant")
		}
	
		for j, _ := range fragments {
			fragments[j] = append(fragments[j], item[j])
		}
	}

	// Normalize
	for i, fragment := range fragments {
		fragments[i], err = doNormalize(fragment)
		if err != nil {
			return result, err
		}
	}

	// Reconstruct
	for i, _ := range fragments[0] {
		var tmp network.DataArray
		for _, fragment := range fragments {
			tmp = append(tmp, fragment[i])
		}
		result = append(result, tmp)
	}

	return result, err
}

func doNormalize(pf network.DataArray) (result network.DataArray, err error) {
	var lowest, highest float64
	for i, item := range pf {
		if i == 0 {
			lowest = item
			highest = item
		} else {
			if (item < lowest) {
				lowest = item
			}

			if (item > highest) {
				highest = item
			}
		}
	}

	divider := highest - lowest 
	if divider == 0 {
		return result, fmt.Errorf("Highest is 0, it will trigger division by zero")
	}

	for _, item := range pf {
		result = append(result, (item - lowest) / divider)
	}

	return result, nil
}