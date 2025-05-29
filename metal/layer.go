package neuron

import "sync"

type Layer struct {
	Neurons []SpikingNeuron `json:"neurons"`
}

func NewLayer(neurons []SpikingNeuron) *Layer {
	return &Layer{Neurons: neurons}
}

func (l *Layer) Forward(inputs []float64, currentTime int, learningRate float64) []int {
	spikes := make([]int, len(l.Neurons))
	var wg sync.WaitGroup
	for i := range l.Neurons {
		wg.Add(1)
		go func(i int) {
			spikes[i] = l.Neurons[i].Forward(inputs, currentTime, learningRate)
			wg.Done()
		}(i)
	}
	wg.Wait()
	return spikes
}
