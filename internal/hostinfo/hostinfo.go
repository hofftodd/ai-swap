// Package hostinfo describes the machine ai-swap runs on: OS, CPU, memory, GPUs, power and
// temperatures, as far as the OS exposes them to an ordinary user (no root, no extra
// tools beyond what ships with the OS). Fields the machine can't report are left empty.
package hostinfo

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Info is one snapshot. Static facts (models, totals, versions) are read once; live
// readings (usage, load, power, temperatures) at most every few seconds.
type Info struct {
	Hostname      string    `json:"hostname"`
	OS            string    `json:"os"`               // e.g. "Ubuntu 26.04.1 LTS", "macOS 27.0"
	Kernel        string    `json:"kernel,omitempty"` // e.g. "6.17.0-14-generic", "Darwin 27.0.0"
	Arch          string    `json:"arch"`
	Model         string    `json:"model,omitempty"` // the computer, e.g. "Mac Studio (Mac16,9)"
	CPU           CPU       `json:"cpu"`
	MemTotalBytes int64     `json:"mem_total_bytes"`
	MemAvailBytes int64     `json:"mem_available_bytes"`
	SwapTotal     int64     `json:"swap_total_bytes,omitempty"`
	SwapUsed      int64     `json:"swap_used_bytes,omitempty"`
	UnifiedMemory bool      `json:"unified_memory,omitempty"` // GPU shares system RAM (Apple Silicon)
	Load          []float64 `json:"load,omitempty"`           // 1, 5, 15 minute load averages
	UptimeSeconds int64     `json:"uptime_seconds,omitempty"`
	GPUs          []GPU     `json:"gpus,omitempty"`
	Power         Power     `json:"power"`
}

type CPU struct {
	Name    string   `json:"name"`
	Cores   int      `json:"cores,omitempty"`             // physical
	Threads int      `json:"threads"`                     // logical
	PCores  int      `json:"performance_cores,omitempty"` // Apple Silicon
	ECores  int      `json:"efficiency_cores,omitempty"`
	MaxMHz  int      `json:"max_mhz,omitempty"`
	TempC   *float64 `json:"temp_c,omitempty"`
	Watts   *float64 `json:"watts,omitempty"` // package power (Linux RAPL, when readable)
	BusyPct *float64 `json:"busy_percent,omitempty"`
}

type GPU struct {
	Name          string   `json:"name"`
	Vendor        string   `json:"vendor,omitempty"`
	Driver        string   `json:"driver,omitempty"`
	Cores         int      `json:"cores,omitempty"` // Apple Silicon GPU cores
	VRAMTotal     int64    `json:"vram_total_bytes,omitempty"`
	VRAMUsed      int64    `json:"vram_used_bytes,omitempty"`
	BusyPercent   *int     `json:"busy_percent,omitempty"`
	Watts         *float64 `json:"watts,omitempty"`
	WattsCap      *float64 `json:"watts_cap,omitempty"`
	TempC         *float64 `json:"temp_c,omitempty"`
	JunctionTempC *float64 `json:"junction_temp_c,omitempty"`
	ClockMHz      *int     `json:"clock_mhz,omitempty"`
	FanRPM        *int     `json:"fan_rpm,omitempty"`

	dev string // sysfs device directory (Linux), for live readings
}

type Power struct {
	Source         string `json:"source,omitempty"` // "ac" or "battery"
	BatteryPercent *int   `json:"battery_percent,omitempty"`
	// Note says why power draw isn't shown, when it isn't (e.g. macOS needs root for it).
	Note string `json:"note,omitempty"`
}

var (
	mu       sync.Mutex
	static   *Info
	cached   Info
	cachedAt time.Time
	raplPrev struct {
		uj int64
		at time.Time
	}
	cpuPrev   struct{ busy, total int64 }
	liveEvery = 4 * time.Second
)

// Get returns the current snapshot (cheap: live readings are refreshed every few seconds).
func Get() Info {
	mu.Lock()
	defer mu.Unlock()
	if static == nil {
		s := readStatic()
		static = &s
	}
	if time.Since(cachedAt) < liveEvery && !cachedAt.IsZero() {
		return cached
	}
	info := *static
	info.CPU = static.CPU
	info.GPUs = append([]GPU(nil), static.GPUs...)
	readLive(&info)
	cached, cachedAt = info, time.Now()
	return info
}

func readStatic() Info {
	host, _ := os.Hostname()
	info := Info{Hostname: host, Arch: runtime.GOARCH, CPU: CPU{Threads: runtime.NumCPU()}}
	switch runtime.GOOS {
	case "linux":
		linuxStatic(&info)
	case "darwin":
		darwinStatic(&info)
	default:
		info.OS = runtime.GOOS
	}
	return info
}

func readLive(info *Info) {
	switch runtime.GOOS {
	case "linux":
		linuxLive(info)
	case "darwin":
		darwinLive(info)
	}
}

// ---- Linux -----------------------------------------------------------------------------

func linuxStatic(info *Info) {
	info.OS = osRelease()
	info.Kernel = strings.TrimSpace(readFile("/proc/sys/kernel/osrelease"))
	vendor := strings.TrimSpace(readFile("/sys/class/dmi/id/sys_vendor"))
	product := strings.TrimSpace(readFile("/sys/class/dmi/id/product_name"))
	if product != "" && !strings.EqualFold(product, "To Be Filled By O.E.M.") {
		info.Model = strings.TrimSpace(vendor + " " + product)
	}
	// CPU: model name, physical cores (unique physical id + core id), max clock.
	cores := map[string]bool{}
	var phys, core string
	sc := bufio.NewScanner(strings.NewReader(readFile("/proc/cpuinfo")))
	for sc.Scan() {
		k, v, ok := strings.Cut(sc.Text(), ":")
		if !ok {
			continue
		}
		k, v = strings.TrimSpace(k), strings.TrimSpace(v)
		switch k {
		case "model name":
			if info.CPU.Name == "" {
				info.CPU.Name = v
			}
		case "physical id":
			phys = v
		case "core id":
			core = v
			cores[phys+"/"+core] = true
		}
	}
	info.CPU.Cores = len(cores)
	if khz := readInt64("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"); khz > 0 {
		info.CPU.MaxMHz = int(khz / 1000)
	}
	info.GPUs = linuxGPUs()
}

func linuxLive(info *Info) {
	mem := meminfo()
	info.MemTotalBytes = mem["MemTotal"]
	info.MemAvailBytes = mem["MemAvailable"]
	info.SwapTotal = mem["SwapTotal"]
	info.SwapUsed = mem["SwapTotal"] - mem["SwapFree"]
	if f := strings.Fields(readFile("/proc/loadavg")); len(f) >= 3 {
		info.Load = parseFloats(f[:3])
	}
	if f := strings.Fields(readFile("/proc/uptime")); len(f) > 0 {
		if v, err := strconv.ParseFloat(f[0], 64); err == nil {
			info.UptimeSeconds = int64(v)
		}
	}
	info.CPU.BusyPct = cpuBusy()
	info.CPU.TempC = cpuTemp()
	info.CPU.Watts = raplWatts()
	for i := range info.GPUs {
		linuxGPULive(&info.GPUs[i])
	}
	info.Power = linuxPower()
	if info.CPU.Watts == nil && hasRAPL() {
		info.Power.Note = "CPU power needs read access to /sys/class/powercap (root on recent kernels)"
	}
}

// linuxGPUs lists display devices from /sys/class/drm, named via lspci when it's there.
func linuxGPUs() []GPU {
	cards, _ := filepath.Glob("/sys/class/drm/card[0-9]*")
	seen := map[string]bool{}
	var out []GPU
	for _, c := range cards {
		if strings.Contains(filepath.Base(c), "-") { // connectors, e.g. card1-DP-1
			continue
		}
		dev, err := filepath.EvalSymlinks(filepath.Join(c, "device"))
		if err != nil || seen[dev] {
			continue
		}
		seen[dev] = true
		g := GPU{Vendor: vendorName(strings.TrimSpace(readFile(filepath.Join(dev, "vendor"))))}
		if drv, err := filepath.EvalSymlinks(filepath.Join(dev, "driver")); err == nil {
			g.Driver = filepath.Base(drv)
		}
		g.Name = pciName(filepath.Base(dev))
		if g.Name == "" {
			g.Name = strings.TrimSpace(g.Vendor + " GPU")
		}
		g.VRAMTotal = readInt64(filepath.Join(dev, "mem_info_vram_total"))
		g.dev = dev
		out = append(out, g)
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].VRAMTotal > out[j].VRAMTotal })
	return out
}

func linuxGPULive(g *GPU) {
	dev := g.dev
	if dev == "" {
		return
	}
	if v := readInt64(filepath.Join(dev, "mem_info_vram_used")); v > 0 {
		g.VRAMUsed = v
	}
	if s := strings.TrimSpace(readFile(filepath.Join(dev, "gpu_busy_percent"))); s != "" {
		if v, err := strconv.Atoi(s); err == nil {
			g.BusyPercent = &v
		}
	}
	hw, _ := filepath.Glob(filepath.Join(dev, "hwmon", "hwmon*"))
	for _, h := range hw {
		// amdgpu: power1_average or power1_input in microwatts, power1_cap; temp1 edge,
		// temp2 junction (millidegrees); freq1_input (Hz); fan1_input (RPM).
		if v := readInt64(filepath.Join(h, "power1_average")); v > 0 {
			g.Watts = ptr(float64(v) / 1e6)
		} else if v := readInt64(filepath.Join(h, "power1_input")); v > 0 {
			g.Watts = ptr(float64(v) / 1e6)
		}
		if v := readInt64(filepath.Join(h, "power1_cap")); v > 0 {
			g.WattsCap = ptr(float64(v) / 1e6)
		}
		if v := readInt64(filepath.Join(h, "temp1_input")); v > 0 {
			g.TempC = ptr(float64(v) / 1000)
		}
		if strings.TrimSpace(readFile(filepath.Join(h, "temp2_label"))) == "junction" {
			if v := readInt64(filepath.Join(h, "temp2_input")); v > 0 {
				g.JunctionTempC = ptr(float64(v) / 1000)
			}
		}
		if v := readInt64(filepath.Join(h, "freq1_input")); v > 0 {
			mhz := int(v / 1e6)
			g.ClockMHz = &mhz
		}
		if v := readInt64(filepath.Join(h, "fan1_input")); v > 0 {
			rpm := int(v)
			g.FanRPM = &rpm
		}
	}
}

func pciName(slot string) string {
	if _, err := exec.LookPath("lspci"); err != nil {
		return ""
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "lspci", "-s", slot, "-mm").Output()
	if err != nil {
		return ""
	}
	// Quoted fields: slot "class" "vendor" "device" ...
	f := regexp.MustCompile(`"([^"]*)"`).FindAllStringSubmatch(string(out), -1)
	if len(f) < 3 {
		return ""
	}
	device := f[2][1]
	if m := regexp.MustCompile(`\[([^\]]+)\]`).FindStringSubmatch(device); m != nil {
		device = m[1] // "Navi 48 [Radeon AI PRO R9700]" -> "Radeon AI PRO R9700"
	}
	return device
}

func vendorName(id string) string {
	switch strings.ToLower(id) {
	case "0x1002":
		return "AMD"
	case "0x10de":
		return "NVIDIA"
	case "0x8086":
		return "Intel"
	}
	return ""
}

func meminfo() map[string]int64 {
	out := map[string]int64{}
	sc := bufio.NewScanner(strings.NewReader(readFile("/proc/meminfo")))
	for sc.Scan() {
		k, v, ok := strings.Cut(sc.Text(), ":")
		if !ok {
			continue
		}
		f := strings.Fields(v)
		if len(f) == 0 {
			continue
		}
		n, _ := strconv.ParseInt(f[0], 10, 64)
		if len(f) > 1 && f[1] == "kB" {
			n <<= 10
		}
		out[k] = n
	}
	return out
}

func cpuBusy() *float64 {
	line, _, _ := strings.Cut(readFile("/proc/stat"), "\n")
	f := strings.Fields(line)
	if len(f) < 5 || f[0] != "cpu" {
		return nil
	}
	var total, idle int64
	for i, s := range f[1:] {
		v, _ := strconv.ParseInt(s, 10, 64)
		total += v
		if i == 3 || i == 4 { // idle, iowait
			idle += v
		}
	}
	busy := total - idle
	prev := cpuPrev
	cpuPrev.busy, cpuPrev.total = busy, total
	if prev.total == 0 || total <= prev.total {
		return nil
	}
	return ptr(100 * float64(busy-prev.busy) / float64(total-prev.total))
}

func cpuTemp() *float64 {
	hw, _ := filepath.Glob("/sys/class/hwmon/hwmon*")
	for _, h := range hw {
		switch strings.TrimSpace(readFile(filepath.Join(h, "name"))) {
		case "k10temp", "coretemp", "zenpower", "cpu_thermal":
			if v := readInt64(filepath.Join(h, "temp1_input")); v > 0 {
				return ptr(float64(v) / 1000)
			}
		}
	}
	return nil
}

func hasRAPL() bool {
	_, err := os.Stat("/sys/class/powercap/intel-rapl:0/energy_uj")
	return err == nil
}

// raplWatts: CPU package power from the energy counter's change since the last reading
// (Intel and AMD both expose it through intel-rapl; often root-only).
func raplWatts() *float64 {
	b, err := os.ReadFile("/sys/class/powercap/intel-rapl:0/energy_uj")
	if err != nil {
		return nil
	}
	uj, err := strconv.ParseInt(strings.TrimSpace(string(b)), 10, 64)
	if err != nil {
		return nil
	}
	now := time.Now()
	prev := raplPrev
	raplPrev.uj, raplPrev.at = uj, now
	if prev.at.IsZero() || uj < prev.uj {
		return nil
	}
	dt := now.Sub(prev.at).Seconds()
	if dt <= 0 {
		return nil
	}
	return ptr(float64(uj-prev.uj) / 1e6 / dt)
}

func linuxPower() Power {
	var p Power
	supplies, _ := filepath.Glob("/sys/class/power_supply/*")
	for _, s := range supplies {
		switch strings.TrimSpace(readFile(filepath.Join(s, "type"))) {
		case "Mains":
			if strings.TrimSpace(readFile(filepath.Join(s, "online"))) == "1" {
				p.Source = "ac"
			} else if p.Source == "" {
				p.Source = "battery"
			}
		case "Battery":
			if v := readInt64(filepath.Join(s, "capacity")); v > 0 {
				n := int(v)
				p.BatteryPercent = &n
			}
		}
	}
	return p
}

func osRelease() string {
	vals := map[string]string{}
	sc := bufio.NewScanner(strings.NewReader(readFile("/etc/os-release")))
	for sc.Scan() {
		k, v, ok := strings.Cut(sc.Text(), "=")
		if ok {
			vals[k] = strings.Trim(v, `"`)
		}
	}
	if v := vals["PRETTY_NAME"]; v != "" {
		return v
	}
	return "Linux"
}

// ---- macOS -----------------------------------------------------------------------------

func darwinStatic(info *Info) {
	name := run("sw_vers", "-productName")
	ver := run("sw_vers", "-productVersion")
	info.OS = strings.TrimSpace(name + " " + ver)
	if k := run("uname", "-sr"); k != "" {
		info.Kernel = k
	}
	info.CPU.Name = sysctl("machdep.cpu.brand_string")
	info.CPU.Cores = atoi(sysctl("hw.physicalcpu"))
	info.CPU.Threads = atoi(sysctl("hw.logicalcpu"))
	info.CPU.PCores = atoi(sysctl("hw.perflevel0.physicalcpu"))
	info.CPU.ECores = atoi(sysctl("hw.perflevel1.physicalcpu"))
	info.MemTotalBytes = int64(atoi64(sysctl("hw.memsize")))
	if m := sysctl("hw.model"); m != "" {
		info.Model = m
	}
	info.UnifiedMemory = runtime.GOARCH == "arm64"
	// GPU (and the computer's marketing name) from system_profiler: slow, so once.
	var sp struct {
		Displays []struct {
			Model  string `json:"sppci_model"`
			Cores  string `json:"sppci_cores"`
			Vendor string `json:"spdisplays_vendor"`
		} `json:"SPDisplaysDataType"`
		Hardware []struct {
			Name  string `json:"machine_name"`
			Model string `json:"machine_model"`
		} `json:"SPHardwareDataType"`
	}
	if out := run("system_profiler", "-json", "SPDisplaysDataType", "SPHardwareDataType"); out != "" {
		_ = json.Unmarshal([]byte(out), &sp)
	}
	for _, d := range sp.Displays {
		g := GPU{Name: d.Model, Cores: atoi(d.Cores), Vendor: strings.TrimPrefix(d.Vendor, "sppci_vendor_")}
		if info.UnifiedMemory {
			g.VRAMTotal = info.MemTotalBytes // unified: the GPU can use system memory
		}
		info.GPUs = append(info.GPUs, g)
	}
	if len(sp.Hardware) > 0 && sp.Hardware[0].Name != "" {
		info.Model = strings.TrimSpace(sp.Hardware[0].Name + " (" + firstNonEmpty(sp.Hardware[0].Model, info.Model) + ")")
	}
}

func darwinLive(info *Info) {
	info.Load = parseFloats(strings.Fields(strings.Trim(sysctl("vm.loadavg"), "{ }")))
	if bt := sysctl("kern.boottime"); bt != "" {
		if m := regexp.MustCompile(`sec = (\d+)`).FindStringSubmatch(bt); m != nil {
			if s, err := strconv.ParseInt(m[1], 10, 64); err == nil {
				info.UptimeSeconds = time.Now().Unix() - s
			}
		}
	}
	// Available memory: free + inactive + speculative + purgeable pages (what vm_stat shows).
	if out := run("vm_stat"); out != "" {
		page := int64(16384)
		if m := regexp.MustCompile(`page size of (\d+) bytes`).FindStringSubmatch(out); m != nil {
			page = atoi64(m[1])
		}
		pages := func(label string) int64 {
			m := regexp.MustCompile(regexp.QuoteMeta(label) + `:\s+(\d+)`).FindStringSubmatch(out)
			if m == nil {
				return 0
			}
			return atoi64(m[1])
		}
		info.MemAvailBytes = page * (pages("Pages free") + pages("Pages inactive") +
			pages("Pages speculative") + pages("Pages purgeable"))
	}
	if sw := sysctl("vm.swapusage"); sw != "" {
		if m := regexp.MustCompile(`total = ([\d.]+)M\s+used = ([\d.]+)M`).FindStringSubmatch(sw); m != nil {
			t, _ := strconv.ParseFloat(m[1], 64)
			u, _ := strconv.ParseFloat(m[2], 64)
			info.SwapTotal, info.SwapUsed = int64(t*(1<<20)), int64(u*(1<<20))
		}
	}
	info.Power = Power{Note: "power draw on macOS needs root (powermetrics)"}
	if out := run("pmset", "-g", "batt"); out != "" {
		if strings.Contains(out, "'AC Power'") {
			info.Power.Source = "ac"
		} else if strings.Contains(out, "'Battery Power'") {
			info.Power.Source = "battery"
		}
		if m := regexp.MustCompile(`(\d+)%`).FindStringSubmatch(out); m != nil {
			n := atoi(m[1])
			info.Power.BatteryPercent = &n
		}
	}
}

// ---- helpers ---------------------------------------------------------------------------

func readFile(path string) string {
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	return string(b)
}

func readInt64(path string) int64 {
	v, _ := strconv.ParseInt(strings.TrimSpace(readFile(path)), 10, 64)
	return v
}

func run(name string, args ...string) string {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, name, args...).Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(bytes.TrimSpace(out)))
}

func sysctl(key string) string { return run("sysctl", "-n", key) }

func atoi(s string) int {
	v, _ := strconv.Atoi(strings.TrimSpace(s))
	return v
}

func atoi64(s string) int64 {
	v, _ := strconv.ParseInt(strings.TrimSpace(s), 10, 64)
	return v
}

func parseFloats(f []string) []float64 {
	var out []float64
	for _, s := range f {
		if v, err := strconv.ParseFloat(s, 64); err == nil {
			out = append(out, v)
		}
	}
	return out
}

func firstNonEmpty(s ...string) string {
	for _, x := range s {
		if x != "" {
			return x
		}
	}
	return ""
}

func ptr[T any](v T) *T { return &v }
