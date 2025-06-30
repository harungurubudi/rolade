package network_test

import (
	"testing"

	"github.com/harungurubudi/rolade/activation"
	"github.com/harungurubudi/rolade/loss"
	"github.com/harungurubudi/rolade/network"
	"github.com/harungurubudi/rolade/optimizer"
)

// generateXORData returns input-output samples for XOR logic.
func generateXORData() (features []network.Vector, targets []network.Vector) {
	features = []network.Vector{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	targets = []network.Vector{
		{0},
		{1},
		{1},
		{0},
	}
	return
}

func TestNetworkTrainXOR(t *testing.T) {
	// Arrange
	inputSize := 2
	outputSize := 1

	net, err := network.NewNetwork(inputSize, outputSize, &activation.ReLU{})
	if err != nil {
		t.Fatalf("AddLayer error: %v", err)
	}

	// Simple architecture: 1 hidden layer with 4 neurons
	err = net.AddLayer(4, &activation.Sigmoid{})
	if err != nil {
		t.Fatalf("AddLayer error: %v", err)
	}

	net.SetProps(network.Props{
		Loss:      &loss.RMSE{},
		Optimizer: optimizer.NewSGDWithLearningRate(1),
		MaxEpoch:  10000,
		ErrLimit:  0.001,
		Patience:  2000,
	})

	features, targets := generateXORData()
	samples, err := network.NewSamples(features, targets)
	if err != nil {
		t.Fatalf("Failed to create samples: %v", err)
	}

	// Act
	err = net.Train(samples)
	if err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	// Assert: model should approximate XOR behavior
	correct := 0
	for i, feature := range features {
		_, conclusion, err := net.Test(feature)
		if err != nil {
			t.Errorf("Error during testing sample %d: %v", i, err)
		}
		if len(conclusion) != 1 {
			t.Errorf("Expected single output, got %d", len(conclusion))
		}
		if conclusion[0] == int(targets[i][0]) {
			correct++
		}
	}

	accuracy := float64(correct) / float64(len(features))
	t.Logf("XOR test accuracy: %.2f", accuracy)

	if accuracy < 0.75 {
		t.Errorf("Expected accuracy >= 0.75, got %.2f", accuracy)
	}
}
