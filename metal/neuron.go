package neuron

import (
	"math"
	"math/rand/v2"
)

type Connection struct {
	Weight        float64 `json:"weight"`
	LastPreSpike  int     `json:"lastPreSpike"`  // Last time pre-synaptic neuron fired
	LastPostSpike int     `json:"lastPostSpike"` // Last time this neuron fired
}

type SpikingNeuron struct {
	MembranePotential float64      `json:"membranePotential"`
	Threshold         float64      `json:"threshold"`         // Base spike threshold
	AdaptiveThreshold float64      `json:"adaptiveThreshold"` // Dynamic threshold adjustment
	Decay             float64      `json:"decay"`             // Membrane potential decay rate
	Bias              float64      `json:"bias"`              // Constant input bias
	Connections       []Connection `json:"connections"`       // Input connections
	RefractoryPeriod  int          `json:"refractoryPeriod"`  // Steps before next allowed spike
	RefractoryTimer   int          `json:"refractoryTimer"`   // Current refractory countdown
	LastSpikeTime     int          `json:"lastSpikeTime"`     // Last spike timestep
	MinWeight         float64      `json:"minWeight"`         // Minimum connection weight
	MaxWeight         float64      `json:"maxWeight"`         // Maximum connection weight
	Fired             bool         `json:"fired"`
	MinBias           float64      `json:"minBias"`
	MaxBias           float64      `json:"maxBias"`
}

func NewSpikingNeuron(
	numInputs int,
	threshold, decay, bias float64,
	refractoryPeriod int,
) *SpikingNeuron {
	connections := make([]Connection, numInputs)
	for i := range connections {
		connections[i] = Connection{
			Weight:       0.5,  // Initialize to mid-range
			LastPreSpike: -100, // Initialize to distant past
		}
	}

	return &SpikingNeuron{
		Threshold:         threshold,
		AdaptiveThreshold: 0,
		Decay:             decay,
		Bias:              bias,
		Connections:       connections,
		RefractoryPeriod:  refractoryPeriod,
		MinWeight:         -1.5,
		MaxWeight:         1.5,
		MinBias:           -1.0,
		MaxBias:           1.0,
	}
}

func (n *SpikingNeuron) Forward(inputs []float64, currentTime int, learningRate float64) int {
	if len(inputs) != len(n.Connections) {
		panic("input/connection size mismatch")
	}

	// Refractory period handling
	if n.RefractoryTimer > 0 {
		n.Fired = false
		n.RefractoryTimer--
		n.MembranePotential *= n.Decay // Still decay during refractory
		return 0
	}

	// Decay and integrate inputs
	n.MembranePotential *= n.Decay
	weightedSum := 0.0
	for i, input := range inputs {
		weightedSum += input * n.Connections[i].Weight
		n.Connections[i].LastPreSpike = currentTime
	}
	n.MembranePotential += weightedSum + n.Bias

	// Add small noise to break symmetry
	n.MembranePotential += (rand.Float64() - 0.5) * 0.05

	// Calculate effective threshold with adaptive component
	effectiveThreshold := n.Threshold + n.AdaptiveThreshold

	// Check for spike
	if n.MembranePotential >= effectiveThreshold {
		n.MembranePotential = 0
		n.RefractoryTimer = n.RefractoryPeriod
		n.LastSpikeTime = currentTime
		n.Fired = true

		// STDP with diminishing returns
		for i := range n.Connections {
			if inputs[i] > 0 {
				// Scale learning by current weight (prevent saturation)
				scale := 1.0 - math.Abs(n.Connections[i].Weight)/n.MaxWeight
				n.Connections[i].Weight = clamp(
					n.Connections[i].Weight+learningRate*0.5*scale,
					n.MinWeight,
					n.MaxWeight,
				)
			}
			n.Connections[i].LastPostSpike = currentTime
		}

		// Adjust adaptive threshold (makes firing harder after each spike)
		n.AdaptiveThreshold += 0.1

		// Adjust bias to make firing slightly harder next time
		n.Bias = clamp(n.Bias-learningRate*0.1, n.MinBias, n.MaxBias)
		return 1
	}

	// Gradually relax adaptive threshold
	n.AdaptiveThreshold *= 0.9

	// LTD - depress all active connections when we don't fire
	for i, input := range inputs {
		if input > 0 {
			// Gentler depression that weakens over time
			n.Connections[i].Weight = clamp(
				n.Connections[i].Weight-learningRate*0.1*(1-math.Abs(n.Connections[i].Weight)/n.MaxWeight),
				n.MinWeight,
				n.MaxWeight,
			)
		}
	}

	// Adjust bias to make firing slightly easier next time
	n.Bias = clamp(n.Bias+learningRate*0.05, n.MinBias, n.MaxBias)
	return 0
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
