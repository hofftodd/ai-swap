package process

import (
	"context"
	"net"
	"os/exec"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/hofftodd/ai-swap/internal/engine"
)

func freePort(t *testing.T) int {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	return ln.Addr().(*net.TCPAddr).Port
}

func TestStartHealthyStop(t *testing.T) {
	py, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 not available")
	}
	port := freePort(t)
	logs := NewRingBuffer(4096)
	// Wrap in sh so Stop has to reach a child process via the process group.
	p, err := Start(&engine.Launch{
		Argv:       []string{"sh", "-c", py + " -m http.server --bind 127.0.0.1 " + strconv.Itoa(port)},
		Dir:        t.TempDir(),
		Port:       port,
		HealthPath: "/",
	}, logs)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.WaitHealthy(context.Background(), 20*time.Second); err != nil {
		t.Fatalf("not healthy: %v\n%s", err, logs.Bytes())
	}
	p.Stop(5 * time.Second)
	select {
	case <-p.Done():
	default:
		t.Fatal("not stopped")
	}
	// The grandchild must be gone too: the port should be free again.
	deadline := time.Now().Add(5 * time.Second)
	for {
		ln, err := net.Listen("tcp", "127.0.0.1:"+strconv.Itoa(port))
		if err == nil {
			ln.Close()
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("child still holds the port after Stop")
		}
		time.Sleep(50 * time.Millisecond)
	}
}

func TestExitBeforeHealthy(t *testing.T) {
	p, err := Start(&engine.Launch{Argv: []string{"sh", "-c", "echo bad flag; exit 3"}, Port: freePort(t), HealthPath: "/"}, NewRingBuffer(1024))
	if err != nil {
		t.Fatal(err)
	}
	err = p.WaitHealthy(context.Background(), 10*time.Second)
	if err == nil || !strings.Contains(err.Error(), "exited") {
		t.Fatalf("want exit error, got %v", err)
	}
}

func TestRingBuffer(t *testing.T) {
	r := NewRingBuffer(5)
	r.Write([]byte("abc"))
	r.Write([]byte("defg"))
	if got := string(r.Bytes()); got != "cdefg" {
		t.Fatalf("got %q", got)
	}
}
