// network.go
package neuron

type Network struct {
	Layers []*Layer `json:"layers"`
}

func NewNetwork(layers []*Layer) *Network {
	return &Network{Layers: layers}
}

func (n *Network) Forward(inputs []float64, currentTime int) []float64 {
	for _, layer := range n.Layers {
		inputs = layer.Forward(inputs, currentTime)
	}
	return inputs
}

func (n *Network) UpdateWeights(currentTime int, learningRate float64) {
	for _, layer := range n.Layers {
		layer.UpdateWeights(currentTime, learningRate)
	}
}

// For classification tasks
func (n *Network) ComputeLoss(target []float64) float64 {
	loss := 0.0
	lastLayer := n.Layers[len(n.Layers)-1]
	for i, neuron := range lastLayer.Neurons {
		diff := target[i] - neuron.SurrogateGradient()
		loss += diff * diff
	}
	return loss
}
