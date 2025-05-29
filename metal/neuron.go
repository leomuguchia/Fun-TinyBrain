package neuron

type Connection struct {
	Weight        float64
	LastPreSpike  int
	LastPostSpike int
}

type SpikingNeuron struct {
	MembranePotential float64
	Threshold         float64
	Decay             float64
	Bias              float64
	Connections       []Connection
	RefractoryPeriod  int
	RefractoryTimer   int
	LastSpikeTime     int
}

func (n *SpikingNeuron) Forward(inputs []float64, currentTime int) int {
	if len(inputs) != len(n.Connections) {
		panic("inputs and connections length mismatch")
	}

	if n.RefractoryTimer > 0 {
		n.RefractoryTimer--
		return 0
	}

	n.MembranePotential *= n.Decay

	for i := 0; i < len(inputs); i++ {
		n.MembranePotential += inputs[i] * n.Connections[i].Weight
		if inputs[i] > 0 {
			n.Connections[i].LastPreSpike = currentTime
		}
	}
	n.MembranePotential += n.Bias

	if n.MembranePotential < 0 {
		n.MembranePotential = 0
	}

	if n.MembranePotential >= n.Threshold {
		n.MembranePotential = 0
		n.RefractoryTimer = n.RefractoryPeriod
		n.LastSpikeTime = currentTime

		// STDP update (post-spike)
		const tau = 5
		for i := range n.Connections {
			dt := currentTime - n.Connections[i].LastPreSpike
			if dt >= 0 && dt <= tau {
				n.Connections[i].Weight += 0.05 // LTP
			}
			n.Connections[i].LastPostSpike = currentTime
		}

		return 1
	}

	// STDP depression (LTD)
	const tau = 5
	for i := range n.Connections {
		dt := currentTime - n.Connections[i].LastPostSpike
		if inputs[i] > 0 && dt >= 0 && dt <= tau {
			n.Connections[i].Weight -= 0.03
		}
	}

	return 0
}
