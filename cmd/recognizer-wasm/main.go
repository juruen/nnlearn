//go:build js && wasm

package main

import (
	"fmt"
	"syscall/js"

	"nnlearn/internal/nn"
	"nnlearn/internal/types"

	"gonum.org/v1/gonum/mat"
)

var (
	model *nn.FeedForward
	funcs []js.Func
)

func main() {
	api := map[string]any{
		"loadModel":    js.FuncOf(loadModel),
		"predictDigit": js.FuncOf(predictDigit),
	}

	for _, name := range []string{"loadModel", "predictDigit"} {
		funcs = append(funcs, api[name].(js.Func))
	}

	js.Global().Set("nnlearn", js.ValueOf(api))
	select {}
}

func loadModel(_ js.Value, args []js.Value) any {
	if len(args) != 1 || args[0].Type() != js.TypeString {
		return errorResult("loadModel expects the model JSON as a string")
	}

	loaded, err := nn.LoadFeedForwardBytes([]byte(args[0].String()))
	if err != nil {
		return errorResult(err.Error())
	}

	model = loaded
	return map[string]any{
		"ok":           true,
		"inputLength":  loaded.InputLength(),
		"outputLength": loaded.OutputLength(),
	}
}

func predictDigit(_ js.Value, args []js.Value) any {
	if model == nil {
		return errorResult("model not loaded")
	}

	if len(args) != 1 {
		return errorResult("predictDigit expects an array of input pixels")
	}

	input, err := float64SliceFromJS(args[0], model.InputLength())
	if err != nil {
		return errorResult(err.Error())
	}

	output, err := model.Predict(mat.NewVecDense(len(input), input))
	if err != nil {
		return errorResult(err.Error())
	}

	digit, confidence, scores := argmax(output)
	return map[string]any{
		"ok":         true,
		"digit":      digit,
		"confidence": confidence,
		"scores":     scores,
	}
}

func float64SliceFromJS(value js.Value, expectedLen int) ([]float64, error) {
	if value.Type() != js.TypeObject {
		return nil, fmt.Errorf("input must be an array-like object")
	}

	length := value.Length()
	if length != expectedLen {
		return nil, fmt.Errorf("expected %d inputs, got %d", expectedLen, length)
	}

	input := make([]float64, length)
	for i := range length {
		item := value.Index(i)
		if item.Type() != js.TypeNumber {
			return nil, fmt.Errorf("input[%d] must be a number", i)
		}
		input[i] = item.Float()
	}

	return input, nil
}

func argmax(v types.Vector) (digit int, confidence float64, scores []any) {
	maxIdx := 0
	maxVal := v.AtVec(0)
	scores = make([]any, v.Len())

	for i := range v.Len() {
		val := v.AtVec(i)
		scores[i] = val
		if val > maxVal {
			maxVal = val
			maxIdx = i
		}
	}

	return maxIdx, maxVal, scores
}

func errorResult(message string) map[string]any {
	return map[string]any{
		"ok":    false,
		"error": message,
	}
}
