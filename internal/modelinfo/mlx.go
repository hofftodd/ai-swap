package modelinfo

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// mlxConfig is the part of an MLX model's config.json we use. Multimodal
// models nest the language model's settings under text_config.
type mlxConfig struct {
	ModelType   string         `json:"model_type"`
	MaxPos      int            `json:"max_position_embeddings"`
	Experts     int            `json:"num_experts"`
	RoutedExp   int            `json:"n_routed_experts"`
	LocalExp    int            `json:"num_local_experts"`
	ExpertsUsed int            `json:"num_experts_per_tok"`
	Quant       map[string]any `json:"quantization"`
	TextConfig  *mlxConfig     `json:"text_config"`
}

func (c *mlxConfig) experts() int { return max(c.Experts, c.RoutedExp, c.LocalExp) }

type stTensor struct {
	Dtype string  `json:"dtype"`
	Shape []int64 `json:"shape"`
}

// safetensorsHeader reads a .safetensors file's tensor table.
func safetensorsHeader(path string) (map[string]stTensor, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var n uint64
	if err := binary.Read(f, binary.LittleEndian, &n); err != nil {
		return nil, err
	}
	if n > 100<<20 {
		return nil, errors.New("implausible safetensors header")
	}
	b := make([]byte, n)
	if _, err := io.ReadFull(f, b); err != nil {
		return nil, err
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(b, &raw); err != nil {
		return nil, err
	}
	out := make(map[string]stTensor, len(raw))
	for k, v := range raw {
		if k == "__metadata__" {
			continue
		}
		var t stTensor
		if json.Unmarshal(v, &t) == nil {
			out[k] = t
		}
	}
	return out, nil
}

func prod(s []int64) int64 {
	n := int64(1)
	for _, d := range s {
		n *= d
	}
	return n
}

// quantBits returns the bits a tensor's weight was quantized to: a per-layer
// override in the quantization map, else the default.
func quantBits(q map[string]any, layer string, def int) int {
	if o, ok := q[layer].(map[string]any); ok {
		if b, ok := o["bits"].(float64); ok {
			return int(b)
		}
	}
	return def
}

func readMLX(dir string) (*Info, error) {
	b, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var cfg mlxConfig
	if err := json.Unmarshal(b, &cfg); err != nil {
		return nil, fmt.Errorf("config.json: %w", err)
	}
	text := &cfg
	if cfg.TextConfig != nil {
		text = cfg.TextConfig
		if text.ExpertsUsed == 0 {
			text.ExpertsUsed = cfg.ExpertsUsed
		}
	}
	info := &Info{Format: "mlx", Arch: cfg.ModelType, CtxTrain: max(text.MaxPos, cfg.MaxPos)}
	info.Experts, info.ExpertsUsed = max(text.experts(), cfg.experts()), text.ExpertsUsed

	def := 0
	if v, ok := cfg.Quant["bits"].(float64); ok {
		def = int(v)
	}
	files, _ := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	sort.Strings(files)
	if len(files) == 0 {
		return nil, errors.New("no .safetensors files")
	}
	var expertParams, embedParams int64
	bitsSeen := map[int]bool{}
	for _, f := range files {
		if st, err := os.Stat(f); err == nil {
			info.SizeBytes += st.Size()
		}
		ts, err := safetensorsHeader(f)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", filepath.Base(f), err)
		}
		for name, t := range ts {
			if strings.HasSuffix(name, ".scales") || strings.HasSuffix(name, ".biases") {
				continue
			}
			n := prod(t.Shape)
			layer := strings.TrimSuffix(name, ".weight")
			if _, quantized := ts[layer+".scales"]; quantized && def > 0 {
				bits := quantBits(cfg.Quant, layer, def)
				bitsSeen[bits] = true
				// Packed into uint32: each element holds 32/bits weights.
				if t.Dtype == "U32" && bits > 0 {
					n = n * 32 / int64(bits)
				}
			}
			info.Params += n
			if strings.Contains(name, "embed") {
				embedParams += n
			}
			if info.Experts > 0 && len(t.Shape) >= 3 && t.Shape[0] == int64(info.Experts) &&
				(strings.Contains(name, "switch_mlp") || strings.Contains(name, "experts")) {
				expertParams += n
			}
		}
	}
	if info.Experts > 0 && info.ExpertsUsed > 0 && expertParams > 0 {
		info.ActiveParams = info.Params - embedParams - expertParams + expertParams*int64(info.ExpertsUsed)/int64(info.Experts)
	}
	if def > 0 {
		info.Quant = fmt.Sprintf("%d-bit", def)
		if g, ok := cfg.Quant["group_size"].(float64); ok {
			info.Quant += fmt.Sprintf(" (group %d)", int(g))
		}
		var others []string
		for b := range bitsSeen {
			if b != def {
				others = append(others, fmt.Sprintf("%d-bit", b))
			}
		}
		sort.Strings(others)
		if len(others) > 0 {
			info.Quant += ", some layers " + strings.Join(others, "/")
		}
	} else {
		info.Quant = "unquantized"
	}
	return info, nil
}
