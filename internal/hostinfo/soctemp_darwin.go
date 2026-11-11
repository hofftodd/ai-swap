//go:build darwin && cgo

package hostinfo

/*
#cgo LDFLAGS: -framework IOKit -framework CoreFoundation
#include <CoreFoundation/CoreFoundation.h>
#include <string.h>

// IOKit's HID event system (private API, the one powermetrics-free tools use): Apple
// Silicon's temperature sensors are HID services on usage page 0xff00, usage 5, and can
// be read without root.
typedef struct __IOHIDEvent *IOHIDEventRef;
typedef struct __IOHIDServiceClient *IOHIDServiceClientRef;
typedef struct __IOHIDEventSystemClient *IOHIDEventSystemClientRef;
IOHIDEventSystemClientRef IOHIDEventSystemClientCreate(CFAllocatorRef);
int IOHIDEventSystemClientSetMatching(IOHIDEventSystemClientRef, CFDictionaryRef);
CFArrayRef IOHIDEventSystemClientCopyServices(IOHIDEventSystemClientRef);
IOHIDEventRef IOHIDServiceClientCopyEvent(IOHIDServiceClientRef, int64_t, int32_t, int64_t);
CFTypeRef IOHIDServiceClientCopyProperty(IOHIDServiceClientRef, CFStringRef);
double IOHIDEventGetFloatValue(IOHIDEventRef, int32_t);

#define TEMPERATURE_EVENT 15

// The hottest die sensor ("PMU tdie<n>"), or -1 if there is none.
static double hottest_die(void) {
	int page = 0xff00, usage = 5;
	CFNumberRef p = CFNumberCreate(NULL, kCFNumberIntType, &page);
	CFNumberRef u = CFNumberCreate(NULL, kCFNumberIntType, &usage);
	const void *keys[] = {CFSTR("PrimaryUsagePage"), CFSTR("PrimaryUsage")};
	const void *vals[] = {p, u};
	CFDictionaryRef match = CFDictionaryCreate(NULL, keys, vals, 2,
		&kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
	CFRelease(p);
	CFRelease(u);
	double best = -1;
	IOHIDEventSystemClientRef client = IOHIDEventSystemClientCreate(kCFAllocatorDefault);
	if (client) {
		IOHIDEventSystemClientSetMatching(client, match);
		CFArrayRef services = IOHIDEventSystemClientCopyServices(client);
		if (services) {
			for (CFIndex i = 0; i < CFArrayGetCount(services); i++) {
				IOHIDServiceClientRef sc = (IOHIDServiceClientRef)CFArrayGetValueAtIndex(services, i);
				CFStringRef name = IOHIDServiceClientCopyProperty(sc, CFSTR("Product"));
				char buf[128] = "";
				if (name) {
					CFStringGetCString(name, buf, sizeof buf, kCFStringEncodingUTF8);
					CFRelease(name);
				}
				if (strncmp(buf, "PMU tdie", 8) != 0) continue;
				IOHIDEventRef ev = IOHIDServiceClientCopyEvent(sc, TEMPERATURE_EVENT, 0, 0);
				if (!ev) continue;
				double t = IOHIDEventGetFloatValue(ev, TEMPERATURE_EVENT << 16);
				CFRelease(ev);
				if (t > 0 && t < 150 && t > best) best = t;
			}
			CFRelease(services);
		}
		CFRelease(client);
	}
	CFRelease(match);
	return best;
}
*/
import "C"

// socTemp: Apple Silicon's hottest die sensor. CPU and GPU share the die, and the
// sensors readable without root aren't labelled as one or the other.
func socTemp() *float64 {
	if t := float64(C.hottest_die()); t > 0 {
		return ptr(t)
	}
	return nil
}
