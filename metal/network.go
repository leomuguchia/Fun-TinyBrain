package neuron

// Spiking neural network
type Network struct {
	Layers []*Layer `json:"layers"`
	Time   int      `json:"time"`
}

func NewNetwork(layers []*Layer) *Network {
	return &Network{Layers: layers, Time: 0}
}

func (n *Network) Forward(input []float64, currentTime int, learningRate float64) []int {
	for _, layer := range n.Layers {
		input = floatSlice(layer.Forward(input, currentTime, learningRate))
	}
	return intSlice(input)
}

func floatSlice(inputs []int) []float64 {
	result := make([]float64, len(inputs))
	for i, v := range inputs {
		result[i] = float64(v)
	}
	return result
}

func intSlice(inputs []float64) []int {
	result := make([]int, len(inputs))
	for i, v := range inputs {
		if v >= 1 {
			result[i] = 1
		} else {
			result[i] = 0
		}
	}
	return result
}
