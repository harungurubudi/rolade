package main 

import (
	"fmt"
	"log"

	"github.com/harungurubudi/rolade/network"
	"github.com/harungurubudi/rolade/optimizer"
	"github.com/harungurubudi/rolade/activation"
)

func main() {
	nt, err := network.NewNetwork(4, 2, &activation.Sigmoid{})
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
			Alpha: 0.01,
		},
		ErrLimit: 0.005,
		MaxEpoch: 10000,
	})

	features, targets := getTrainingData()
	err = nt.Train(features, targets)
	if err != nil {
		log.Fatal(err)
	}

	err = nt.Save(".")
	if err != nil {
		log.Fatal(err)
	}

	model, err := network.Load(".")
	if err != nil {
		log.Fatal(err)
	}
	
	test(model, network.DataArray{0, 0, 0, 1}) // 0, 0
	test(model, network.DataArray{0, 1, 1, 1}) // 1, 0
	test(model, network.DataArray{1, 1, 0, 1}) // 0, 1
	test(model, network.DataArray{1, 0, 1, 1}) // 0, 0
}

func test(nt *network.Network, input network.DataArray) {
	real, res, err := nt.Test(input)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("%f -> %d\n", real, res)
}

func getTrainingData() (features []network.DataArray, targets []network.DataArray) {
	features = []network.DataArray{
		network.DataArray{0, 0, 0, 1},
		network.DataArray{0, 0, 1, 1},
		network.DataArray{0, 1, 1, 1},
		network.DataArray{1, 1, 1, 1},
		network.DataArray{1, 0, 0, 1},
		network.DataArray{1, 0, 1, 1},
		network.DataArray{1, 1, 0, 1},
		
	}

	targets = []network.DataArray{
		network.DataArray{0, 0},
		network.DataArray{0, 1},
		network.DataArray{1, 0},
		network.DataArray{1, 1},
		network.DataArray{1, 0},
		network.DataArray{0, 0},
		network.DataArray{0, 1},
	}

	return features, targets
}