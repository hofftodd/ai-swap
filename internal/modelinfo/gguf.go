package modelinfo

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
)

// GGUF value types.
const (
	ggufU8 = iota
	ggufI8
	ggufU16
	ggufI16
	ggufU32
	ggufI32
	ggufF32
	ggufBool
	ggufString
	ggufArray
	ggufU64
	ggufI64
	ggufF64
)

// fileTypes names general.file_type (llama.cpp's LLAMA_FTYPE_*).
var fileTypes = map[uint64]string{
	0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
	10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M",
	16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S",
	22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M",
	28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0",
	37: "TQ2_0", 38: "MXFP4_MOE",
}

type ggufReader struct {
	r   *bufio.Reader
	err error
}

func (g *ggufReader) read(v any) {
	if g.err == nil {
		g.err = binary.Read(g.r, binary.LittleEndian, v)
	}
}

func (g *ggufReader) u32() uint32 { var v uint32; g.read(&v); return v }
func (g *ggufReader) u64() uint64 { var v uint64; g.read(&v); return v }

func (g *ggufReader) str() string {
	n := g.u64()
	if g.err != nil {
		return ""
	}
	if n > 1<<20 {
		g.err = fmt.Errorf("string of %d bytes", n)
		return ""
	}
	b := make([]byte, n)
	_, g.err = io.ReadFull(g.r, b)
	return string(b)
}

// value reads one value of type t; numbers come back as int64/uint64/float64.
// Arrays are skipped (nil) except their length, which is all we need.
func (g *ggufReader) value(t uint32) any {
	switch t {
	case ggufU8, ggufI8, ggufBool:
		var v uint8
		g.read(&v)
		return uint64(v)
	case ggufU16, ggufI16:
		var v uint16
		g.read(&v)
		return uint64(v)
	case ggufU32:
		return uint64(g.u32())
	case ggufI32:
		var v int32
		g.read(&v)
		return int64(v)
	case ggufF32:
		var v float32
		g.read(&v)
		return float64(v)
	case ggufU64:
		return g.u64()
	case ggufI64:
		var v int64
		g.read(&v)
		return v
	case ggufF64:
		var v float64
		g.read(&v)
		return v
	case ggufString:
		return g.str()
	case ggufArray:
		et, n := g.u32(), g.u64()
		if et == ggufArray || n > 1<<26 { // vocabularies are ~10^5-10^6 entries
			g.err = fmt.Errorf("unsupported GGUF array (type %d, %d entries)", et, n)
			return nil
		}
		for i := uint64(0); i < n && g.err == nil; i++ {
			g.value(et)
		}
		return nil
	}
	g.err = fmt.Errorf("unknown GGUF value type %d", t)
	return nil
}

func asInt(v any) int64 {
	switch x := v.(type) {
	case uint64:
		return int64(x)
	case int64:
		return x
	case float64:
		return int64(x)
	}
	return 0
}

// ggmlTypes names tensor types (ggml_type), for the per-tensor quantization mix.
var ggmlTypes = map[uint32]string{
	0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
	10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS",
	17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S",
	23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
	30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4",
}

type tensor struct {
	name string
	dims []uint64
	typ  uint32
}

// elems is the tensor's element count, or 0 for an implausible shape.
func (t tensor) elems() int64 {
	n := int64(1)
	for _, d := range t.dims {
		if d == 0 || d > 1<<40 || n > (1<<62)/int64(d) {
			return 0
		}
		n *= int64(d)
	}
	return n
}

// readGGUFFile returns a file's metadata and tensor table.
func readGGUFFile(path string) (map[string]any, []tensor, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	g := &ggufReader{r: bufio.NewReaderSize(f, 1<<20)}
	var magic [4]byte
	g.read(&magic)
	if g.err == nil && string(magic[:]) != "GGUF" {
		return nil, nil, errors.New("not a GGUF file")
	}
	if v := g.u32(); g.err == nil && v < 2 {
		return nil, nil, fmt.Errorf("GGUF version %d is too old", v)
	}
	nt, nkv := g.u64(), g.u64()
	if nt > 1<<20 || nkv > 1<<20 {
		return nil, nil, errors.New("implausible GGUF header")
	}
	kv := map[string]any{}
	for i := uint64(0); i < nkv && g.err == nil; i++ {
		k := g.str()
		kv[k] = g.value(g.u32())
	}
	ts := make([]tensor, 0, nt)
	for i := uint64(0); i < nt && g.err == nil; i++ {
		t := tensor{name: g.str()}
		nd := g.u32()
		if nd > 8 {
			return nil, nil, fmt.Errorf("tensor %s has %d dims", t.name, nd)
		}
		for j := uint32(0); j < nd; j++ {
			t.dims = append(t.dims, g.u64())
		}
		t.typ = g.u32()
		g.u64() // offset
		ts = append(ts, t)
	}
	return kv, ts, g.err
}

func readGGUF(path string) (*Info, error) {
	info := &Info{Format: "gguf"}
	var kv map[string]any
	var experts, expertParams, embedParams int64
	byType := map[string]int64{}
	for i, p := range shards(path) {
		m, ts, err := readGGUFFile(p)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", p, err)
		}
		if i == 0 {
			kv = m
		}
		if st, err := os.Stat(p); err == nil {
			info.SizeBytes += st.Size()
		}
		for _, t := range ts {
			n := t.elems()
			info.Params += n
			if len(t.dims) >= 2 {
				byType[ggmlTypes[t.typ]] += n
			}
			// Routed expert weights are stacked: ffn_{gate,up,down}_exps.
			if strings.Contains(t.name, "_exps") {
				expertParams += n
			}
			if strings.Contains(t.name, "token_embd") || strings.Contains(t.name, "per_layer_token") {
				embedParams += n
			}
		}
	}
	arch, _ := kv["general.architecture"].(string)
	info.Arch = arch
	info.Name, _ = kv["general.name"].(string)
	info.SizeLabel, _ = kv["general.size_label"].(string)
	ftName := ""
	if ft, ok := kv["general.file_type"]; ok {
		ftName = fileTypes[uint64(asInt(ft))]
	}
	info.Quant = quantLabel(ftName, byType)
	info.CtxTrain = int(asInt(kv[arch+".context_length"]))
	experts = asInt(kv[arch+".expert_count"])
	used := asInt(kv[arch+".expert_used_count"])
	info.Experts, info.ExpertsUsed = int(experts), int(used)
	if experts > 0 && used > 0 && expertParams > 0 {
		info.ActiveParams = info.Params - embedParams - expertParams + expertParams*used/experts
	}
	return info, nil
}

// quantLabel names a GGUF's quantization. general.file_type is right for files
// quantized in one pass; hand-mixed files often carry a misleading one, so
// when the tensors are mostly another type it describes the mix instead.
func quantLabel(fileType string, byType map[string]int64) string {
	var total int64
	for _, n := range byType {
		total += n
	}
	if total == 0 {
		return fileType
	}
	type share struct {
		name string
		n    int64
	}
	var shares []share
	for k, n := range byType {
		shares = append(shares, share{k, n})
	}
	sort.Slice(shares, func(i, j int) bool { return shares[i].n > shares[j].n })
	top := shares[0]
	family := strings.TrimSuffix(strings.TrimSuffix(strings.TrimSuffix(fileType, "_S"), "_M"), "_L")
	if fileType != "" && (top.name == fileType || top.name == family) {
		return fileType
	}
	label := fmt.Sprintf("mostly %s (%d%%)", top.name, top.n*100/total)
	var rest []string
	for _, s := range shares[1:] {
		if s.n*100/total >= 3 {
			rest = append(rest, s.name)
		}
	}
	if len(rest) > 0 {
		label += ", " + strings.Join(rest, ", ")
	}
	return label
}
