package server

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

func get(t *testing.T, url string, hdr ...string) (*http.Response, string) {
	t.Helper()
	req, _ := http.NewRequest("GET", url, nil)
	for i := 0; i+1 < len(hdr); i += 2 {
		req.Header.Set(hdr[i], hdr[i+1])
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return resp, string(b)
}

func TestClientConfigs(t *testing.T) {
	linux := newNode(t, linuxCfg)
	mac := newNode(t, macCfg, linux.url)
	base := mac.url + "/v1"

	resp, body := get(t, mac.url+"/api/clients/pi")
	if resp.StatusCode != 200 || !strings.Contains(resp.Header.Get("Content-Disposition"), `filename="models.json"`) {
		t.Fatalf("pi: %d %v %s", resp.StatusCode, resp.Header, body)
	}
	var pi struct {
		Providers map[string]struct {
			BaseURL string `json:"baseUrl"`
			Compat  map[string]any
			Models  []struct{ ID, Name string }
		}
	}
	if err := json.Unmarshal([]byte(body), &pi); err != nil {
		t.Fatal(err)
	}
	q, c := pi.Providers["ai-swap-qwen"], pi.Providers["ai-swap-coder"]
	if q.BaseURL != base || q.Models[0].ID != "qwen" || q.Compat["thinkingFormat"] != "qwen-chat-template" {
		t.Fatalf("pi qwen provider: %+v", q)
	}
	if c.BaseURL != base || c.Models[0].Name != "coder (peer0)" { // the peer's model, via this node
		t.Fatalf("pi coder provider: %+v", c)
	}

	_, body = get(t, mac.url+"/api/clients/omp")
	var omp map[string]map[string]map[string]any
	if err := yaml.Unmarshal([]byte(body), &omp); err != nil || omp["providers"]["ai-swap-coder"]["auth"] != "none" {
		t.Fatalf("omp: %v %s", err, body)
	}

	_, body = get(t, mac.url+"/api/clients/opencode?default=coder")
	var oc map[string]any
	if err := json.Unmarshal([]byte(body), &oc); err != nil || oc["model"] != "ai-swap-coder/coder" {
		t.Fatalf("opencode: %v %s", err, body)
	}

	// The endpoint is the address the browser used.
	resp, body = get(t, mac.url+"/api/clients/claude", "X-Forwarded-Host", "studio.local:8080")
	if resp.StatusCode != 200 || !strings.Contains(body, `ANTHROPIC_BASE_URL="http://studio.local:8080"`) ||
		!strings.Contains(body, `AI_SWAP_MODEL:-qwen}`) {
		t.Fatalf("claude: %d %s", resp.StatusCode, body)
	}

	for url, want := range map[string]int{
		"/api/clients/vim":                                       404,
		"/api/clients/claude?default=nope":                       400,
		"/api/clients/claude?endpoint=http://x$(touch%20/tmp/p)": 400,
		"/api/clients/pi?endpoint=http://h:1/v1":                 400,
	} {
		if resp, body := get(t, mac.url+url); resp.StatusCode != want {
			t.Errorf("%s: %d, want %d (%s)", url, resp.StatusCode, want, body)
		}
	}
}
