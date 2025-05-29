package neuron

type Layer struct {
	neurons []SpikingNeuron
}

func NewLayer(neurons []SpikingNeuron) *Layer {
	return &Layer{neurons: neurons}
}

func (l *Layer) Forward(inputs []float64) []int {
	spikes := make([]int, len(l.neurons))
	for i := range l.neurons {
		spikes[i] = l.neurons[i].Forward(inputs)
	}
	return spikes
}
