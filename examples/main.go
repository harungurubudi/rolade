package main 

import (
	"fmt"
	"log"

	"github.com/harungurubudi/rolade/network"
	"github.com/harungurubudi/rolade/optimizer"
)

func main() {
	nt := network.NewNetwork(4, 2)
	nt.AddHiddenLayer(4)
	nt.AddHiddenLayer(3)

	nt.SetProps(network.Props{
		Optimizer: &optimizer.SGD{
			Alpha: 0.01,
		},
		ErrLimit: 0.005,
		MaxEpoch: 10000,
	})

	features, targets := getTrainingData()
	err := nt.Train(features, targets)
	if err != nil {
		log.Fatal(err)
	}
	
	test(nt, network.DataArray{0, 0, 0, 1}) // 0, 0
	test(nt, network.DataArray{0, 1, 1, 1}) // 1, 0
	test(nt, network.DataArray{1, 1, 0, 1}) // 0, 1
	test(nt, network.DataArray{1, 0, 1, 1}) // 0, 0
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