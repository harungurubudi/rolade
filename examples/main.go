package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"strconv"
	"strings"

	"github.com/harungurubudi/rolade/activation"
	"github.com/harungurubudi/rolade/network"
	"github.com/harungurubudi/rolade/optimizer"
	"github.com/harungurubudi/rolade/preprocessor"
)

func main() {
	nt, err := network.NewNetwork(4, 2, &activation.Tanh{})
	if err != nil {
		log.Fatal(err)
	}

	err = nt.AddLayer(4, &activation.Sigmoid{})
	if err != nil {
		log.Fatal(err)
	}

	err = nt.AddLayer(2, &activation.Sigmoid{})
	if err != nil {
		log.Fatal(err)
	}

	nt.SetProps(network.Props{
		Optimizer: &optimizer.SGD{
			Alpha: 1,
		},
		ErrLimit: 0.002,
		MaxEpoch: 20000,
		Patience: 10000,
	})

	features, targets, err := getData("./examples/dataset/IRIS_train.csv")
	if err != nil {
		log.Fatal(err)
	}

	samples, _ := network.NewSamples(features, targets)

	err = nt.Train(samples)
	if err != nil {
		log.Fatal(err)
	}

	err = nt.Save(".")
	if err != nil {
		log.Fatal(err)
	}

	// Ok, training is done. Test the model
	testModel(nt)
}

func testModel(model *network.Network) {
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

		log.Printf("%d. Expected %s, but got %s", (i + 1), expected, got)

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

func getData(filepath string) (features []network.Vector, targets []network.Vector, err error) {
	dataset, err := ioutil.ReadFile(filepath)
	if err != nil {
		return features, targets, fmt.Errorf("Got error while loading dataset: %v", err)
	}

	var featureData []network.Vector
	r := csv.NewReader(strings.NewReader(string(dataset)))
	for {
		item, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return features, targets, err
		}

		firstItem, err := strconv.ParseFloat(item[0], 64)
		if err == nil {
			featureData = append(featureData, network.Vector{
				firstItem,
				getFloat(item[1]),
				getFloat(item[2]),
				getFloat(item[3]),
			})
		}

		switch item[4] {
		case "Iris-setosa":
			targets = append(targets, network.Vector{0, 0})
		case "Iris-versicolor":
			targets = append(targets, network.Vector{0, 1})
		case "Iris-virginica":
			targets = append(targets, network.Vector{1, 0})
		}
	}

	features, err = preprocessor.Normalize(featureData)
	if err != nil {
		return features, targets, fmt.Errorf("Got error while normalize features: %v", err)
	}

	return features, targets, err
}

func getFloat(input string) (result float64) {
	result, err := strconv.ParseFloat(input, 64)
	if err != nil {
		result = 0
	}

	return result
}
