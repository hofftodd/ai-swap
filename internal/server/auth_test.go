package server

import (
	"io"
	"net/http"
	"strings"
	"testing"
)

func keyed(cfg string) string {
	return strings.Replace(cfg, "node: {name: ", "node: {api_key: s3cret, name: ", 1)
}

func call(t *testing.T, method, url, body string, hdr map[string]string) (int, string) {
	t.Helper()
	req, _ := http.NewRequest(method, url, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	for k, v := range hdr {
		req.Header.Set(k, v)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, string(b)
}

func TestAPIKey(t *testing.T) {
	mac := newNode(t, keyed(macCfg))
	chat := `{"model":"qwen","messages":[]}`
	bearer := map[string]string{"Authorization": "Bearer s3cret"}

	if code, body := call(t, "POST", mac.url+"/v1/chat/completions", chat, nil); code != 401 || !strings.Contains(body, "API key") {
		t.Fatalf("no key: %d %s", code, body)
	}
	if code, body := call(t, "POST", mac.url+"/v1/messages", chat, nil); code != 401 || !strings.Contains(body, `"type":"error"`) {
		t.Fatalf("anthropic shape: %d %s", code, body)
	}
	if code, _ := call(t, "POST", mac.url+"/v1/chat/completions", chat, map[string]string{"Authorization": "Bearer wrong"}); code != 401 {
		t.Fatalf("wrong key: %d", code)
	}
	if code, body := call(t, "POST", mac.url+"/v1/chat/completions", chat, bearer); code != 200 {
		t.Fatalf("bearer: %d %s", code, body)
	}
	if code, _ := call(t, "POST", mac.url+"/v1/messages", chat, map[string]string{"x-api-key": "s3cret"}); code != 200 {
		t.Fatalf("x-api-key: %d", code)
	}
	for _, p := range []string{"/api/status", "/api/leases", "/running", "/v1/models"} {
		if code, _ := call(t, "GET", mac.url+p, "", nil); code != 401 {
			t.Fatalf("%s without key: %d", p, code)
		}
	}
	if code, _ := call(t, "GET", mac.url+"/health", "", nil); code != 200 {
		t.Fatalf("health should stay open: %d", code)
	}

	// Client configs point at $AI_SWAP_API_KEY and never contain the key.
	code, body := call(t, "GET", mac.url+"/api/clients/pi", "", bearer)
	if code != 200 || !strings.Contains(body, "$AI_SWAP_API_KEY") || strings.Contains(body, "s3cret") {
		t.Fatalf("pi config: %d %s", code, body)
	}
	if _, body := call(t, "GET", mac.url+"/api/clients/claude", "", bearer); !strings.Contains(body, "${AI_SWAP_API_KEY:-no-key-required}") {
		t.Fatalf("claude script: %s", body)
	}
}

func TestAPIKeyBetweenPeers(t *testing.T) {
	linux := newNode(t, keyed(linuxCfg))
	// newNode waits until the peer is online, which needs the key on polls.
	mac := newNodeWithKey(t, keyed(macCfg), "s3cret", linux.url)
	code, body := call(t, "POST", mac.url+"/v1/chat/completions", `{"model":"coder","messages":[]}`,
		map[string]string{"x-api-key": "s3cret"})
	if code != 200 || !strings.Contains(body, "coder-backend") {
		t.Fatalf("forward with key: %d %s", code, body)
	}
}
