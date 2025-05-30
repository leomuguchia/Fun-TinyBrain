// neuron.go
package neuron

import (
	"math"
	"math/rand/v2"
)

// STDP parameters
const (
	Aplus    = 0.1  // LTP strength
	Aminus   = 0.12 // LTD strength
	TauPlus  = 20.0 // LTP time constant
	TauMinus = 20.0 // LTD time constant
)

type Connection struct {
	Weight        float64 `json:"weight"`
	LastPreSpike  int     `json:"lastPreSpike"`  // Timestep of last presynaptic spike
	LastPostSpike int     `json:"lastPostSpike"` // Timestep of last postsynaptic spike
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
	MinWeight         float64      `json:"minWeight"`
	MaxWeight         float64      `json:"maxWeight"`
	Fired             bool         `json:"fired"`
	Inhibition        float64      `json:"inhibition"` // Lateral inhibition
}

func (n *SpikingNeuron) Forward(inputs []float64, currentTime int) {
	if len(inputs) != len(n.Connections) {
		panic("input/connection size mismatch")
	}

	// Apply inhibition
	n.MembranePotential -= n.Inhibition
	n.Inhibition *= 0.8 // Decay inhibition

	// Refractory period handling
	if n.RefractoryTimer > 0 {
		n.Fired = false
		n.RefractoryTimer--
		n.MembranePotential *= n.Decay
		return
	}

	// Decay membrane potential
	n.MembranePotential *= n.Decay

	// Integrate inputs
	for i, input := range inputs {
		if input > 0 { // Input spike
			n.MembranePotential += n.Connections[i].Weight
			n.Connections[i].LastPreSpike = currentTime
		}
	}
	n.MembranePotential += n.Bias

	// Add noise
	n.MembranePotential += (rand.Float64() - 0.5) * 0.1

	// Check for spike
	if n.MembranePotential >= n.Threshold {
		n.spike(currentTime)
	}
}

func (n *SpikingNeuron) spike(currentTime int) {
	n.MembranePotential = 0
	n.RefractoryTimer = n.RefractoryPeriod
	n.LastSpikeTime = currentTime
	n.Fired = true
}

func (n *SpikingNeuron) UpdateWeights(currentTime int, learningRate float64) {
	for i := range n.Connections {
		conn := &n.Connections[i]

		// Calculate time differences
		Δt := float64(conn.LastPostSpike - conn.LastPreSpike)

		// Apply STDP rule
		if Δt > 0 {
			// LTP: Pre before post
			conn.Weight += Aplus * math.Exp(-Δt/TauPlus) * learningRate
		} else if Δt < 0 {
			// LTD: Post before pre
			conn.Weight -= Aminus * math.Exp(Δt/TauMinus) * learningRate
		}

		// Clamp weights
		conn.Weight = clamp(conn.Weight, n.MinWeight, n.MaxWeight)

		// Reset spike times
		if n.Fired {
			conn.LastPostSpike = currentTime
		}
	}
}

// Surrogate gradient for supervised learning
func (n *SpikingNeuron) SurrogateGradient() float64 {
	// Differentiable approximation of spike function
	x := n.MembranePotential - n.Threshold
	return math.Exp(-x * x) // Gaussian approximation
}

func clamp(x, min, max float64) float64 {
	if x < min {
		return min
	}
	if x > max {
		return max
	}
	return x
}
