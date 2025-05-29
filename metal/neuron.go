package neuron

type Connection struct {
	Weight        float64 `json:"weight"`
	LastPreSpike  int     `json:"lastPreSpike"`
	LastPostSpike int     `json:"lastPostSpike"`
}

type SpikingNeuron struct {
	MembranePotential float64      `json:"membranePotential"`
	Threshold         float64      `json:"threshold"`
	Decay             float64      `json:"decay"`
	Bias              float64      `json:"bias"`
	Connections       []Connection `json:"connections"`
	RefractoryPeriod  int          `json:"refractoryPeriod"`
	RefractoryTimer   int          `json:"refractoryTimer"`
	LastSpikeTime     int          `json:"lastSpikeTime"`
}

func (n *SpikingNeuron) Forward(inputs []float64, currentTime int) int {
	if len(inputs) != len(n.Connections) {
		panic("inputs and connections length mismatch")
	}

	if n.RefractoryTimer > 0 {
		n.RefractoryTimer--
		return 0
	}

	// Decay potential
	n.MembranePotential *= n.Decay

	// Add input contributions
	for i := 0; i < len(inputs); i++ {
		n.MembranePotential += inputs[i] * n.Connections[i].Weight
		if inputs[i] > 0 {
			n.Connections[i].LastPreSpike = currentTime
		}
	}

	// Add bias
	n.MembranePotential += n.Bias

	// Clip to zero if below
	if n.MembranePotential < 0 {
		n.MembranePotential = 0
	}

	// Check for firing
	if n.MembranePotential >= n.Threshold {
		n.MembranePotential = 0
		n.RefractoryTimer = n.RefractoryPeriod
		n.LastSpikeTime = currentTime

		// --- STDP Long-Term Potentiation (LTP) ---
		const tau = 5
		const LTP = 0.2
		for i := range n.Connections {
			dt := currentTime - n.Connections[i].LastPreSpike
			if dt >= 0 && dt <= tau {
				n.Connections[i].Weight += LTP
			}
			n.Connections[i].LastPostSpike = currentTime
		}

		return 1
	}

	// --- STDP Long-Term Depression (LTD) ---
	const tau = 5
	const LTD = 0.1
	for i := range n.Connections {
		dt := currentTime - n.Connections[i].LastPostSpike
		if inputs[i] > 0 && dt >= 0 && dt <= tau {
			n.Connections[i].Weight -= LTD
		}
	}

	return 0
}
