// Package modelinfo reads what a model is from its files on disk: parameter
// count (total and, for mixture-of-experts models, active per token),
// quantization, weight size, architecture and trained context. It reads only
// headers (GGUF metadata and tensor tables, safetensors headers and MLX's
// config.json), never the weights, and caches results by path and mtime.
package modelinfo

import (
	"errors"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
)

// Info is what the files say about a model. Zero values mean unknown.
type Info struct {
	Format       string  `json:"format,omitempty"`        // "gguf" or "mlx"
	Arch         string  `json:"arch,omitempty"`          // e.g. "qwen3moe"
	Name         string  `json:"name,omitempty"`          // general.name, if any
	Params       int64   `json:"params,omitempty"`        // total parameters
	ActiveParams int64   `json:"active_params,omitempty"` // per token for MoE models, embedding tables excluded
	Experts      int     `json:"experts,omitempty"`       // routed experts
	ExpertsUsed  int     `json:"experts_used,omitempty"`  // experts per token
	Quant        string  `json:"quant,omitempty"`         // e.g. "Q4_K_M", "4-bit (group 64), 8-bit parts"
	SizeBytes    int64   `json:"size_bytes,omitempty"`    // weights on disk
	BitsPerW     float64 `json:"bits_per_weight,omitempty"`
	CtxTrain     int     `json:"ctx_train,omitempty"`
	SizeLabel    string  `json:"size_label,omitempty"` // general.size_label, e.g. "35B-A3B"
}

type entry struct {
	mod  time.Time
	info *Info
	err  error
}

var (
	mu    sync.Mutex
	cache = map[string]entry{}
)

// Read describes the model at path: a .gguf file (the first shard of a split
// model) or an MLX model directory. Results are cached until the file changes.
func Read(path string) (*Info, error) {
	st, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	mu.Lock()
	if e, ok := cache[path]; ok && e.mod.Equal(st.ModTime()) {
		mu.Unlock()
		return e.info, e.err
	}
	mu.Unlock()

	var info *Info
	switch {
	case st.IsDir():
		info, err = readMLX(path)
	case strings.HasSuffix(strings.ToLower(path), ".gguf"):
		info, err = readGGUF(path)
	default:
		err = errors.New("not a .gguf file or model directory")
	}
	if info != nil {
		info.finish()
	}
	mu.Lock()
	cache[path] = entry{mod: st.ModTime(), info: info, err: err}
	mu.Unlock()
	return info, err
}

func (i *Info) finish() {
	if i.Params > 0 && i.SizeBytes > 0 {
		i.BitsPerW = float64(i.SizeBytes*8) / float64(i.Params)
		i.BitsPerW = float64(int(i.BitsPerW*100+0.5)) / 100
	}
	if i.ActiveParams >= i.Params {
		i.ActiveParams = 0
	}
}

// shardRe matches split model files: name-00001-of-00003.gguf.
var shardRe = regexp.MustCompile(`^(.*)-(\d{5})-of-(\d{5})\.gguf$`)

// shards returns every file of a split GGUF model, in order, or just path.
func shards(path string) []string {
	m := shardRe.FindStringSubmatch(filepath.Base(path))
	if m == nil {
		return []string{path}
	}
	all, _ := filepath.Glob(filepath.Join(filepath.Dir(path), m[1]+"-*-of-"+m[3]+".gguf"))
	if len(all) == 0 {
		return []string{path}
	}
	sort.Strings(all)
	return all
}
