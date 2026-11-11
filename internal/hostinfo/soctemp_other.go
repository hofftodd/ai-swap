//go:build !darwin || !cgo

package hostinfo

// socTemp needs cgo on macOS (IOKit); elsewhere CPU temperature comes from hwmon.
func socTemp() *float64 { return nil }
