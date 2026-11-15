package modelinfo

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type ggufKV struct {
	key string
	typ uint32
	val any
}

type ggufT struct {
	name string
	dims []uint64
	typ  uint32
}

// writeGGUF writes a header-only GGUF (no tensor data; readers never look).
func writeGGUF(t *testing.T, path string, kvs []ggufKV, ts []ggufT) {
	t.Helper()
	var b bytes.Buffer
	w := func(v any) { _ = binary.Write(&b, binary.LittleEndian, v) }
	str := func(s string) { w(uint64(len(s))); b.WriteString(s) }
	b.WriteString("GGUF")
	w(uint32(3))
	w(uint64(len(ts)))
	w(uint64(len(kvs)))
	for _, kv := range kvs {
		str(kv.key)
		w(kv.typ)
		switch v := kv.val.(type) {
		case string:
			str(v)
		case uint32:
			w(v)
		case []string: // array of strings, skipped by the reader
			w(uint32(ggufString))
			w(uint64(len(v)))
			for _, s := range v {
				str(s)
			}
		}
	}
	for _, x := range ts {
		str(x.name)
		w(uint32(len(x.dims)))
		for _, d := range x.dims {
			w(d)
		}
		w(x.typ)
		w(uint64(0))
	}
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestGGUFMoE(t *testing.T) {
	path := filepath.Join(t.TempDir(), "m-Q4_K_M.gguf")
	writeGGUF(t, path, []ggufKV{
		{"general.architecture", ggufString, "qwen3moe"},
		{"general.name", ggufString, "Test MoE"},
		{"general.file_type", ggufU32, uint32(15)}, // Q4_K_M
		{"qwen3moe.context_length", ggufU32, uint32(262144)},
		{"qwen3moe.expert_count", ggufU32, uint32(8)},
		{"qwen3moe.expert_used_count", ggufU32, uint32(2)},
		{"tokenizer.ggml.tokens", ggufArray, []string{"a", "b"}},
	}, []ggufT{
		{"token_embd.weight", []uint64{100, 10}, 12},           // 1,000 (embedding)
		{"blk.0.attn_q.weight", []uint64{100, 100}, 12},        // 10,000
		{"blk.0.ffn_up_exps.weight", []uint64{100, 10, 8}, 12}, // 8,000 (experts)
		{"output_norm.weight", []uint64{100}, 0},               // 100
	})
	info, err := Read(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.Params != 19100 || info.Arch != "qwen3moe" || info.Name != "Test MoE" || info.CtxTrain != 262144 {
		t.Fatalf("basics: %+v", info)
	}
	if info.Quant != "Q4_K_M" {
		t.Fatalf("quant: %q", info.Quant)
	}
	// 19,100 - 1,000 embedding - 8,000 experts + 2,000 (2 of 8 experts).
	if info.ActiveParams != 12100 || info.Experts != 8 || info.ExpertsUsed != 2 {
		t.Fatalf("active: %+v", info)
	}
}

func TestGGUFMisleadingFileType(t *testing.T) {
	path := filepath.Join(t.TempDir(), "mix.gguf")
	writeGGUF(t, path, []ggufKV{
		{"general.architecture", ggufString, "llama"},
		{"general.file_type", ggufU32, uint32(19)}, // claims IQ2_XXS
	}, []ggufT{
		{"blk.0.ffn_up.weight", []uint64{1000, 100}, 12}, // Q4_K, 100,000
		{"blk.0.attn_q.weight", []uint64{100, 100}, 8},   // Q8_0, 10,000
	})
	info, err := Read(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(info.Quant, "mostly Q4_K (90%)") || !strings.Contains(info.Quant, "Q8_0") {
		t.Fatalf("quant: %q", info.Quant)
	}
}

func TestGGUFShards(t *testing.T) {
	dir := t.TempDir()
	kv := []ggufKV{{"general.architecture", ggufString, "llama"}, {"general.file_type", ggufU32, uint32(7)}}
	writeGGUF(t, filepath.Join(dir, "big-00001-of-00002.gguf"), kv, []ggufT{{"a.weight", []uint64{10, 10}, 8}})
	writeGGUF(t, filepath.Join(dir, "big-00002-of-00002.gguf"), nil, []ggufT{{"b.weight", []uint64{10, 20}, 8}})
	info, err := Read(filepath.Join(dir, "big-00001-of-00002.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	if info.Params != 300 || info.Quant != "Q8_0" {
		t.Fatalf("shards: %+v", info)
	}
}

func writeSafetensors(t *testing.T, path string, tensors map[string]stTensor) {
	t.Helper()
	h, _ := json.Marshal(tensors)
	var b bytes.Buffer
	_ = binary.Write(&b, binary.LittleEndian, uint64(len(h)))
	b.Write(h)
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestMLXQuantizedMoE(t *testing.T) {
	dir := t.TempDir()
	cfg := `{"model_type":"flint","max_position_embeddings":262144,"num_experts":4,"num_experts_per_tok":1,
	  "quantization":{"bits":4,"group_size":64,"model.layers.0.self_attn.q_proj":{"bits":8,"group_size":64}}}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}
	writeSafetensors(t, filepath.Join(dir, "model.safetensors"), map[string]stTensor{
		// 4-bit: 8 weights per uint32. 100x(10*8) = 8,000 embedding weights.
		"model.embed_tokens.weight": {"U32", []int64{100, 10}},
		"model.embed_tokens.scales": {"BF16", []int64{100, 2}},
		// 8-bit override: 4 weights per uint32. 100x(25*4) = 10,000.
		"model.layers.0.self_attn.q_proj.weight": {"U32", []int64{100, 25}},
		"model.layers.0.self_attn.q_proj.scales": {"BF16", []int64{100, 2}},
		// Stacked experts, 4-bit: 4x10x(10*8) = 3,200.
		"model.layers.0.mlp.switch_mlp.up_proj.weight": {"U32", []int64{4, 10, 10}},
		"model.layers.0.mlp.switch_mlp.up_proj.scales": {"BF16", []int64{4, 10, 2}},
		"model.norm.weight":                            {"BF16", []int64{100}},
	})
	info, err := Read(dir)
	if err != nil {
		t.Fatal(err)
	}
	if info.Params != 21300 || info.Format != "mlx" || info.CtxTrain != 262144 {
		t.Fatalf("basics: %+v", info)
	}
	if info.Quant != "4-bit (group 64), some layers 8-bit" {
		t.Fatalf("quant: %q", info.Quant)
	}
	// 21,300 - 8,000 embedding - 3,200 experts + 800 (1 of 4).
	if info.ActiveParams != 10900 {
		t.Fatalf("active: %d", info.ActiveParams)
	}
	if info.SizeBytes == 0 || info.BitsPerW == 0 {
		t.Fatalf("size: %+v", info)
	}
}

func TestReadErrors(t *testing.T) {
	if _, err := Read(filepath.Join(t.TempDir(), "missing.gguf")); err == nil {
		t.Fatal("missing file")
	}
	p := filepath.Join(t.TempDir(), "x.gguf")
	_ = os.WriteFile(p, []byte("nope"), 0o644)
	if _, err := Read(p); err == nil {
		t.Fatal("bad magic")
	}
	if _, err := Read(t.TempDir()); err == nil {
		t.Fatal("dir without config.json")
	}
}
