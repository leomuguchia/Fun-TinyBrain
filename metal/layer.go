package neuron

type Layer struct {
	Neurons []SpikingNeuron `json:"neurons"`
}

func NewLayer(neurons []SpikingNeuron) *Layer {
	return &Layer{Neurons: neurons}
}

func (l *Layer) Forward(inputs []float64, currentTime int) []int {
	spikes := make([]int, len(l.Neurons))
	for i := range l.Neurons {
		spikes[i] = l.Neurons[i].Forward(inputs, currentTime)
	}
	return spikes
}
