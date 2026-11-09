package hostinfo

import (
	"encoding/json"
	"testing"
	"time"
)

func TestGetDescribesThisMachine(t *testing.T) {
	first := Get()
	if first.Hostname == "" || first.OS == "" || first.Arch == "" || first.CPU.Threads == 0 {
		t.Fatalf("missing basics: %+v", first)
	}
	if first.MemTotalBytes == 0 {
		t.Fatalf("no memory total: %+v", first)
	}
	// Live readings are cached briefly: an immediate second call is the same snapshot.
	if again := Get(); again.UptimeSeconds != first.UptimeSeconds {
		t.Fatalf("expected the cached snapshot")
	}
	cachedAt = time.Time{} // force a fresh read (rate-based values need two readings)
	b, _ := json.MarshalIndent(Get(), "", "  ")
	t.Logf("%s", b)
}
