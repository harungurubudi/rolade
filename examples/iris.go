package main

import (
	"fmt"
	"log"
	"strings"
	"io"
	"io/ioutil"
	"encoding/csv"
	"strconv"

	"github.com/harungurubudi/rolade/network"
	"github.com/harungurubudi/rolade/optimizer"
	"github.com/harungurubudi/rolade/activation"
)

func main() {
	nt, err := network.NewNetwork(4, 2, &activation.Tanh{})
	if err != nil {
		log.Fatal(err)
	}

	err = nt.AddLayer(4, &activation.Tanh{})
	if err != nil {
		log.Fatal(err)
	}
	
	err = nt.AddLayer(3, &activation.Tanh{})
	if err != nil {
		log.Fatal(err)
	}

	nt.SetProps(network.Props{
		Optimizer: &optimizer.SGD{
			Alpha: 0.001,
		},
		ErrLimit: 0.005,
		MaxEpoch: 100000,
	})

	features, targets, err := getData("./examples/dataset/IRIS_train.csv")
	if err != nil {
		log.Fatal(err)
	}

	err = nt.Train(features, targets)
	if err != nil {
		log.Fatal(err)
	}

	err = nt.Save(".")
	if err != nil {
		log.Fatal(err)
	}

	// Ok, training is done. Test the model
	testModel()
}


func testModel() {
	model, err := network.Load(".")
	if err != nil {
		log.Fatal(err)
	}

	features, targets, err := getData("./examples/dataset/IRIS_test.csv")
	if err != nil {
		log.Fatal(err)
	}

	var success int

	for i, item := range features {
		_, res, err := model.Test(item)
		if err != nil {
			log.Fatal(err)
		}

		expected := bindLabel([]int{
			int(targets[i][0]),
			int(targets[i][1]),
		})

		got := bindLabel([]int{
			res[0],
			res[1],
		})

		log.Printf("Expected %s, but got %s", expected, got)

		if expected == got {
			success++
		}
		
	}

	percentage := (float64(success) / float64(len(features))) * 100

	log.Printf("Got %d of %d, got %f percent successful", success, len(features), percentage)
}

func bindLabel(target []int) string {
	if compareLabel(target, []int{0, 0}) {
		return "Iris-setosa"
	} else if compareLabel(target, []int{0, 1}) {
		return "Iris-versicolor"
	} else if compareLabel(target, []int{1, 0}) {
		return "Iris-virginica"
	} else {
		return "Undefined"
	}
}

func compareLabel(input, target []int) (result bool) {
	if input[0] == target[0] && input[1] == target[1] {
		result = true
	}

	return result
}

func getData(filepath string) (features []network.DataArray, targets []network.DataArray, err error) {
	dataset, err := ioutil.ReadFile(filepath)
    if err != nil {
		return features, targets, fmt.Errorf("Got error while loading dataset: %v", err)
	}

	pfs := make([]network.DataArray, 4)

	r := csv.NewReader(strings.NewReader(string(dataset)))
	for {
		item, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return features, targets, err
		}

		for i := range pfs {
			iFloat, err := strconv.ParseFloat(item[i], 64)
			if err == nil {
				pfs[i] = append(pfs[i], iFloat)
			}
		}

		switch item[4]{
		case "Iris-setosa":
			targets = append(targets, network.DataArray{0, 0})
		case "Iris-versicolor":
			targets = append(targets, network.DataArray{0, 1})
		case "Iris-virginica":
			targets = append(targets, network.DataArray{1, 0})
		}
	}

	for i, item := range pfs {
		pfs[i], err = normalize(item) 
		if err != nil {
			return features, targets, err
		}
	}

	for i := range pfs[0] {
		features = append(features, network.DataArray{
			pfs[0][i],
			pfs[1][i],
			pfs[2][i],
			pfs[3][i],
		})
	}

	return features, targets, err
}

func normalize(pf network.DataArray) (result network.DataArray, err error) {
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