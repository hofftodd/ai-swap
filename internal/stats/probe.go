package stats

import (
	"bufio"
	"context"
	"encoding/json"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Backend is what a running engine says about itself. Every field is
// optional: engines expose different things, and absent means unknown.
type Backend struct {
	Params      int64              `json:"params,omitempty"`       // parameter count
	SizeBytes   int64              `json:"size_bytes,omitempty"`   // weights on disk/in memory
	Quant       string             `json:"quant,omitempty"`        // e.g. "Q4_K - Medium"
	CtxTrain    int                `json:"ctx_train,omitempty"`    // context the model was trained for
	Ctx         int                `json:"ctx,omitempty"`          // context the server was started with (total)
	CtxPerSlot  int                `json:"ctx_per_slot,omitempty"` // context available to one request
	Vocab       int                `json:"vocab,omitempty"`
	EmbedDim    int                `json:"embed_dim,omitempty"`
	Build       string             `json:"build,omitempty"`
	Slots       int                `json:"slots,omitempty"`
	SlotsBusy   *int               `json:"slots_busy,omitempty"`
	Speculative bool               `json:"speculative,omitempty"`
	Modalities  []string           `json:"modalities,omitempty"`
	Tools       *bool              `json:"tools,omitempty"`
	ModelPath   string             `json:"model_path,omitempty"`
	Metrics     map[string]float64 `json:"metrics,omitempty"` // Prometheus gauges/counters, prefix stripped
}

var client = &http.Client{Timeout: 1500 * time.Millisecond}

func getJSON(ctx context.Context, base, path string, v any) bool {
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, base+path, nil)
	resp, err := client.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK && json.NewDecoder(resp.Body).Decode(v) == nil
}

// ProbeBackend asks a running engine on 127.0.0.1:port about itself. It
// tries llama.cpp's endpoints (/props, /slots, /metrics) and the OpenAI
// /v1/models listing, which other engines may enrich with metadata. The
// probes run in parallel; anything that fails is simply left out.
func ProbeBackend(ctx context.Context, engine string, port int) *Backend {
	base := "http://127.0.0.1:" + strconv.Itoa(port)
	b := &Backend{}
	var mu sync.Mutex
	var wg sync.WaitGroup
	run := func(f func()) { wg.Add(1); go func() { defer wg.Done(); f() }() }

	run(func() { // OpenAI listing; llama.cpp puts GGUF metadata in "meta"
		var r struct {
			Data []struct {
				Meta struct {
					NParams   int64  `json:"n_params"`
					Size      int64  `json:"size"`
					NCtxTrain int    `json:"n_ctx_train"`
					NCtx      int    `json:"n_ctx"`
					NVocab    int    `json:"n_vocab"`
					NEmbd     int    `json:"n_embd"`
					Ftype     string `json:"ftype"`
				} `json:"meta"`
				Architecture struct {
					Input []string `json:"input_modalities"`
				} `json:"architecture"`
			} `json:"data"`
		}
		if !getJSON(ctx, base, "/v1/models", &r) || len(r.Data) == 0 {
			return
		}
		d := r.Data[0]
		mu.Lock()
		defer mu.Unlock()
		b.Params, b.SizeBytes, b.CtxTrain = d.Meta.NParams, d.Meta.Size, d.Meta.NCtxTrain
		b.Vocab, b.EmbedDim = d.Meta.NVocab, d.Meta.NEmbd
		if b.Quant == "" {
			b.Quant = d.Meta.Ftype
		}
		if b.Ctx == 0 {
			b.Ctx = d.Meta.NCtx
		}
		if len(b.Modalities) == 0 && len(d.Architecture.Input) > 0 {
			b.Modalities = d.Architecture.Input
		}
	})

	if engine == "llamacpp" {
		run(func() {
			var p struct {
				Defaults struct {
					NCtx int `json:"n_ctx"`
				} `json:"default_generation_settings"`
				TotalSlots int             `json:"total_slots"`
				Ftype      string          `json:"model_ftype"`
				Path       string          `json:"model_path"`
				Build      string          `json:"build_info"`
				Modalities map[string]bool `json:"modalities"`
				Caps       struct {
					Tools *bool `json:"supports_tools"`
				} `json:"chat_template_caps"`
			}
			if !getJSON(ctx, base, "/props", &p) {
				return
			}
			mu.Lock()
			defer mu.Unlock()
			b.Slots, b.Build, b.ModelPath, b.Tools = p.TotalSlots, p.Build, p.Path, p.Caps.Tools
			if p.Ftype != "" {
				b.Quant = p.Ftype
			}
			if p.Defaults.NCtx > 0 {
				b.CtxPerSlot = p.Defaults.NCtx
			}
			mods := []string{"text"}
			for k, on := range p.Modalities {
				if on {
					mods = append(mods, k)
				}
			}
			sort.Strings(mods[1:])
			b.Modalities = mods
		})
		run(func() {
			var slots []struct {
				NCtx        int  `json:"n_ctx"`
				Processing  bool `json:"is_processing"`
				Speculative bool `json:"speculative"`
			}
			if !getJSON(ctx, base, "/slots", &slots) {
				return
			}
			busy := 0
			for _, s := range slots {
				if s.Processing {
					busy++
				}
			}
			mu.Lock()
			defer mu.Unlock()
			b.SlotsBusy = &busy
			if len(slots) > 0 {
				b.Speculative = slots[0].Speculative
				if b.CtxPerSlot == 0 {
					b.CtxPerSlot = slots[0].NCtx
				}
			}
		})
		run(func() {
			m := metrics(ctx, base+"/metrics")
			mu.Lock()
			b.Metrics = m
			mu.Unlock()
		})
	}
	wg.Wait()
	return b
}

// metrics reads a Prometheus text endpoint into name -> value, dropping the
// "engine:" prefix llama.cpp uses ("llamacpp:kv_cache_usage_ratio").
func metrics(ctx context.Context, url string) map[string]float64 {
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	resp, err := client.Do(req)
	if err != nil {
		return nil
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil
	}
	out := map[string]float64{}
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		line := sc.Text()
		if line == "" || line[0] == '#' {
			continue
		}
		name, val, ok := strings.Cut(line, " ")
		if !ok || strings.Contains(name, "{") {
			continue
		}
		f, err := strconv.ParseFloat(strings.TrimSpace(val), 64)
		if err != nil {
			continue
		}
		if _, after, found := strings.Cut(name, ":"); found {
			name = after
		}
		out[name] = f
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// RSSBytes returns the resident memory of a process, or 0 if unknown.
// On Apple Silicon this includes Metal buffers in unified memory; on Linux
// with a discrete GPU, model weights in VRAM are not counted here (see GPUs).
func RSSBytes(pid int) int64 {
	if pid <= 0 {
		return 0
	}
	if runtime.GOOS == "linux" {
		f, err := os.Open("/proc/" + strconv.Itoa(pid) + "/status")
		if err != nil {
			return 0
		}
		defer f.Close()
		sc := bufio.NewScanner(f)
		for sc.Scan() {
			if v, ok := strings.CutPrefix(sc.Text(), "VmRSS:"); ok {
				kb, _ := strconv.ParseInt(strings.Fields(v)[0], 10, 64)
				return kb << 10
			}
		}
		return 0
	}
	out, err := exec.Command("ps", "-o", "rss=", "-p", strconv.Itoa(pid)).Output()
	if err != nil {
		return 0
	}
	kb, _ := strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64)
	return kb << 10
}

// GPU is one graphics device's memory and load, where the OS exposes it.
type GPU struct {
	Name           string `json:"name"`
	VRAMUsedBytes  int64  `json:"vram_used_bytes"`
	VRAMTotalBytes int64  `json:"vram_total_bytes"`
	GTTUsedBytes   int64  `json:"gtt_used_bytes,omitempty"` // system RAM mapped for the GPU
	GTTTotalBytes  int64  `json:"gtt_total_bytes,omitempty"`
	BusyPercent    *int   `json:"busy_percent,omitempty"`
}

// GPUs reports AMD GPUs via the amdgpu driver's sysfs files (Linux). Other
// platforms return nil for now; on Apple Silicon, memory is unified and the
// process RSS already covers the model.
func GPUs() []GPU {
	if runtime.GOOS != "linux" {
		return nil
	}
	cards, _ := filepath.Glob("/sys/class/drm/card[0-9]*/device/mem_info_vram_total")
	var out []GPU
	for _, f := range cards {
		dev := filepath.Dir(f)
		g := GPU{
			Name:           filepath.Base(filepath.Dir(dev)),
			VRAMUsedBytes:  readInt(dev, "mem_info_vram_used"),
			VRAMTotalBytes: readInt(dev, "mem_info_vram_total"),
			GTTUsedBytes:   readInt(dev, "mem_info_gtt_used"),
			GTTTotalBytes:  readInt(dev, "mem_info_gtt_total"),
		}
		if g.VRAMTotalBytes == 0 {
			continue
		}
		if b, err := os.ReadFile(filepath.Join(dev, "gpu_busy_percent")); err == nil {
			if v, err := strconv.Atoi(strings.TrimSpace(string(b))); err == nil {
				g.BusyPercent = &v
			}
		}
		out = append(out, g)
	}
	// Biggest card first: usually the one models run on.
	sort.Slice(out, func(i, j int) bool { return out[i].VRAMTotalBytes > out[j].VRAMTotalBytes })
	return out
}

func readInt(dir, name string) int64 {
	b, err := os.ReadFile(filepath.Join(dir, name))
	if err != nil {
		return 0
	}
	v, _ := strconv.ParseInt(strings.TrimSpace(string(b)), 10, 64)
	return v
}
