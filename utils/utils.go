package utils

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	neuron "tinybrain/metal"

	"github.com/go-echarts/go-echarts/v2/charts"
	"github.com/go-echarts/go-echarts/v2/opts"
	"github.com/muesli/termenv"
)

// ClassificationResult holds evaluation metrics
type ClassificationResult struct {
	PatternDifferentiation bool
	ConsistencyA           float64
	ConsistencyB           float64
	SeparationScore        float64
}

// CheckClassification evaluates if the network distinguishes between patterns
func CheckClassification(net *neuron.Network, patternA, patternB []float64, trials int) ClassificationResult {
	var diffSum, outputASum, outputBSum float64
	outputAHistory := make([][]int, trials)
	outputBHistory := make([][]int, trials)

	for i := 0; i < trials; i++ {
		outputA := net.Forward(patternA, i)
		outputB := net.Forward(patternB, i)
		outputAHistory[i] = outputA
		outputBHistory[i] = outputB

		// Calculate pattern difference
		for j := range outputA {
			diffSum += math.Abs(float64(outputA[j] - outputB[j]))
			outputASum += float64(outputA[j])
			outputBSum += float64(outputB[j])
		}
	}

	avgDiff := diffSum / float64(trials*len(patternA))
	consistencyA := calculateConsistency(outputAHistory)
	consistencyB := calculateConsistency(outputBHistory)
	separationScore := avgDiff * (consistencyA + consistencyB) / 2

	result := ClassificationResult{
		PatternDifferentiation: avgDiff > 0.5,
		ConsistencyA:           consistencyA,
		ConsistencyB:           consistencyB,
		SeparationScore:        separationScore,
	}

	fmt.Printf("\nClassification Results:\n")
	fmt.Printf("✅ Pattern Differentiation: %v\n", result.PatternDifferentiation)
	fmt.Printf("📊 Separation Score: %.2f\n", result.SeparationScore)
	fmt.Printf("🔵 Pattern A Consistency: %.2f\n", consistencyA)
	fmt.Printf("🔴 Pattern B Consistency: %.2f\n", consistencyB)

	return result
}

func calculateConsistency(outputs [][]int) float64 {
	if len(outputs) == 0 {
		return 0
	}

	var sum float64
	base := outputs[0]
	for i := 1; i < len(outputs); i++ {
		for j := range base {
			if outputs[i][j] == base[j] {
				sum++
			}
		}
	}
	return sum / float64((len(outputs)-1)*len(base))
}

// VisualizeSpikeRaster creates an interactive spike raster plot
func VisualizeSpikeRaster(spikes [][]int, patterns []string, patternSwitchInterval int) error {
	if err := os.MkdirAll("visualization", 0755); err != nil {
		return fmt.Errorf("failed to create visualization directory: %w", err)
	}

	line := charts.NewLine()
	line.SetGlobalOptions(
		charts.WithTitleOpts(opts.Title{
			Title:    "Network Spike Raster",
			Subtitle: "Each line represents a neuron's activity over time",
		}),
		charts.WithTooltipOpts(opts.Tooltip{Show: true, Trigger: "axis"}),
		charts.WithXAxisOpts(opts.XAxis{Name: "Time Step"}),
		charts.WithYAxisOpts(opts.YAxis{Name: "Neuron Index"}),
		charts.WithDataZoomOpts(opts.DataZoom{
			Type:  "slider",
			Start: 0,
			End:   100,
		}),
	)

	// Add pattern background
	line.SetXAxis(makeTimeAxis(len(spikes)))
	line.AddSeries("Pattern A", generatePatternSeries(patterns, patternSwitchInterval, "A")).
		SetSeriesOptions(charts.WithAreaStyleOpts(opts.AreaStyle{
			Opacity: 0.1,
			Color:   "#4682B4",
		}))

	// Add each neuron's spike train
	for neuronIdx := 0; neuronIdx < len(spikes[0]); neuronIdx++ {
		data := make([]opts.LineData, 0, len(spikes))
		for t := 0; t < len(spikes); t++ {
			if spikes[t][neuronIdx] == 1 {
				data = append(data, opts.LineData{Value: neuronIdx})
			} else {
				data = append(data, opts.LineData{Value: nil})
			}
		}
		line.AddSeries(fmt.Sprintf("Neuron %d", neuronIdx), data).
			SetSeriesOptions(charts.WithLineStyleOpts(opts.LineStyle{Width: 1}))
	}

	// Save to HTML file
	f, err := os.Create(filepath.Join("visualization", "spike_raster.html"))
	if err != nil {
		return fmt.Errorf("failed to create spike raster file: %w", err)
	}
	defer f.Close()

	return line.Render(f)
}

func makeTimeAxis(length int) []string {
	axis := make([]string, length)
	for i := range axis {
		axis[i] = fmt.Sprintf("%d", i)
	}
	return axis
}

func generatePatternSeries(patterns []string, interval int, highlight string) []opts.LineData {
	series := make([]opts.LineData, 0)
	for t := 0; ; t++ {
		patternIdx := t / interval % len(patterns)
		if patterns[patternIdx] == highlight {
			series = append(series, opts.LineData{Value: 0})
		} else {
			series = append(series, opts.LineData{Value: nil})
		}
		if len(series) >= 1000 { // Limit for performance
			break
		}
	}
	return series
}

// PrintPatternClassification prints a colored terminal output
func PrintPatternClassification(outputs [][]int, patterns []string, patternSwitchInterval int) error {
	p := termenv.ColorProfile()
	colors := map[string]termenv.Color{
		"A":     p.Color("#4682B4"), // SteelBlue
		"B":     p.Color("#DC143C"), // Crimson
		"spike": p.Color("#FFD700"), // Gold
	}

	for t, output := range outputs {
		if t >= 100 { // Limit output for terminal
			break
		}

		patternIdx := t / patternSwitchInterval % len(patterns)
		currentPattern := patterns[patternIdx]
		color := colors[currentPattern]

		// Print time and pattern label
		fmt.Printf(termenv.String("t=%03d [%s] ").Foreground(color).String(), t, currentPattern)

		// Print spikes
		for _, spike := range output {
			if spike == 1 {
				fmt.Print(termenv.String("■").Foreground(colors["spike"]).String())
			} else {
				fmt.Print(" ")
			}
		}

		// Print potential classification marker
		if t%patternSwitchInterval == patternSwitchInterval/2 {
			fmt.Print(termenv.String(" ⇨").Foreground(color).String())
		}

		fmt.Println()
	}
	return nil
}

// GenerateWeightHeatmap creates weight visualizations for all layers
func GenerateWeightHeatmap(net *neuron.Network) error {
	if err := os.MkdirAll("visualization/weights", 0755); err != nil {
		return fmt.Errorf("failed to create weights directory: %w", err)
	}

	for layerIdx, layer := range net.Layers {
		heatmap := charts.NewHeatMap()
		heatmap.SetGlobalOptions(
			charts.WithTitleOpts(opts.Title{
				Title:    fmt.Sprintf("Layer %d Weight Matrix", layerIdx),
				Subtitle: "Input neurons vs Layer neurons",
			}),
			charts.WithVisualMapOpts(opts.VisualMap{
				Calculable: true,
				Min:        -1,
				Max:        1,
				InRange:    &opts.VisualMapInRange{Color: []string{"#0000FF", "#FFFFFF", "#FF0000"}},
			}),
		)

		// Prepare heatmap data
		var data []opts.HeatMapData
		for neuronIdx, neuron := range layer.Neurons {
			for connIdx, conn := range neuron.Connections {
				data = append(data, opts.HeatMapData{
					Value: [3]interface{}{connIdx, neuronIdx, conn.Weight},
				})
			}
		}

		heatmap.AddSeries("weights", data)

		// Save to file
		f, err := os.Create(fmt.Sprintf("visualization/weights/layer_%d.html", layerIdx))
		if err != nil {
			return fmt.Errorf("failed to create heatmap file: %w", err)
		}
		defer f.Close()

		if err := heatmap.Render(f); err != nil {
			return fmt.Errorf("failed to render heatmap: %w", err)
		}
	}

	return nil
}
