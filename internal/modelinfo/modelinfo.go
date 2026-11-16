// Package modelinfo reads what a model is from its files on disk: parameter
// count (total and, for mixture-of-experts models, active per token),
// quantization, weight size, architecture and trained context. It reads only
// headers (GGUF metadata and tensor tables, safetensors headers and MLX's
// config.json), never the weights, and caches results by path and mtime.
package modelinfo

import (
	"bufio"
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
	Lab          string  `json:"lab,omitempty"`        // who made the model, e.g. "Qwen (Alibaba)"
	URL          string  `json:"url,omitempty"`        // the original model card, e.g. on Hugging Face
	BaseModel    string  `json:"base_model,omitempty"` // its Hugging Face id, e.g. "Qwen/Qwen3.8-27B"
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
		info.origin(path, st.IsDir())
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

// labNames gives Hugging Face organisations a readable name; others show as they are.
var labNames = map[string]string{
	"qwen": "Qwen (Alibaba)", "google": "Google", "nvidia": "NVIDIA", "deepseek-ai": "DeepSeek",
	"deepseek": "DeepSeek", "meta-llama": "Meta", "mistralai": "Mistral AI", "microsoft": "Microsoft",
	"openai": "OpenAI", "thinkingmachines": "Thinking Machines Lab", "zai-org": "Z.ai",
	"moonshotai": "Moonshot AI", "ibm-granite": "IBM", "allenai": "Ai2", "nanbeige": "Nanbeige",
	"minimaxai": "MiniMax", "baidu": "Baidu", "tencent": "Tencent", "xiaomimimo": "Xiaomi",
	"ricursiveai": "Ricursive",
}

// LabName is the readable name for an organisation (a Hugging Face handle or a GGUF
// general.organization value).
func LabName(org string) string {
	org = strings.TrimSpace(org)
	if n, ok := labNames[strings.ToLower(org)]; ok {
		return n
	}
	return org
}

// hfRepo is the "org/name" of a Hugging Face model URL, or "".
func hfRepo(u string) string {
	const pre = "https://huggingface.co/"
	if !strings.HasPrefix(u, pre) {
		return ""
	}
	parts := strings.Split(strings.Trim(strings.TrimPrefix(u, pre), "/"), "/")
	if len(parts) < 2 || parts[0] == "" || parts[1] == "" {
		return ""
	}
	return parts[0] + "/" + parts[1]
}

// origin fills Lab, URL and BaseModel from the README next to the model: Hugging Face model
// cards start with YAML front matter whose base_model names the original ("Qwen/Qwen3.8-27B").
// GGUF metadata, read first, wins. For a GGUF the README counts only when it is the sole
// model in its folder, so one README isn't credited to several models.
func (i *Info) origin(path string, isDir bool) {
	dir := path
	if !isDir {
		dir = filepath.Dir(path)
		ggufs, _ := filepath.Glob(filepath.Join(dir, "*.gguf"))
		n := 0
		for _, g := range ggufs {
			b := strings.ToLower(filepath.Base(g))
			if !strings.Contains(b, "mmproj") && !shardRe.MatchString(filepath.Base(g)) || strings.Contains(b, "-00001-of-") {
				n++
			}
		}
		if n != 1 {
			return
		}
	}
	base := readmeBaseModel(filepath.Join(dir, "README.md"))
	if base == "" {
		return
	}
	if i.BaseModel == "" {
		i.BaseModel = base
	}
	if i.Lab == "" {
		i.Lab = LabName(strings.SplitN(base, "/", 2)[0])
	}
	if i.URL == "" {
		i.URL = "https://huggingface.co/" + base
	}
}

// readmeBaseModel reads base_model from a model card's YAML front matter (a string, or the
// first item of a list), or "".
func readmeBaseModel(path string) string {
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	inFront, inList := false, false
	for n := 0; sc.Scan() && n < 80; n++ {
		line := strings.TrimRight(sc.Text(), " \t")
		if line == "---" {
			if inFront {
				return ""
			}
			inFront = true
			continue
		}
		if !inFront {
			return "" // no front matter
		}
		if inList {
			if v, ok := strings.CutPrefix(strings.TrimSpace(line), "- "); ok {
				return cleanRepo(v)
			}
			inList = false
		}
		if v, ok := strings.CutPrefix(line, "base_model:"); ok {
			if v = strings.TrimSpace(v); v == "" {
				inList = true
				continue
			}
			return cleanRepo(strings.Trim(v, "[]"))
		}
	}
	return ""
}

func cleanRepo(v string) string {
	v = strings.Trim(strings.TrimSpace(strings.SplitN(v, ",", 2)[0]), `"'`)
	if strings.Count(v, "/") != 1 || strings.ContainsAny(v, " <>") {
		return ""
	}
	return v
}
