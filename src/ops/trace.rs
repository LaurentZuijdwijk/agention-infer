//! Kernel-level tracing for the GPU-resident forward pass.
//!
//! Set `GGUF_TRACE_KERNEL=1` to enable per-kernel timing eprintln's on the
//! resident dispatch paths. Tracing is enabled once at startup (checked via
//! `std::env::var` on every call otherwise — a full process-environment walk
//! per matmul) — cached once, same pattern as `trace_cpu::enabled()`.
//!
//! For structured output (JSON), use `GGUF_TRACE_KERNEL=json` — this will
//! collect per-kernel timings and output them as JSON when `dump_trace()`
//! is called (e.g. at the end of a forward pass).
//!
//! The output includes:
//!   - kernel: name of the kernel
//!   - time_ms: elapsed time in milliseconds
//!   - effective_bandwidth_gb_s: estimated bandwidth utilization (GB/s)
//!     if tensor size is provided, else null
//!
//! To include effective bandwidth, call `Timer::new_with_size()` with the
//! tensor size in bytes, or use the `trace_with_size()` convenience function.

use std::sync::{OnceLock, Mutex};

static TRACING_ENABLED: OnceLock<bool> = OnceLock::new();
static TRACING_FORMAT: OnceLock<&'static str> = OnceLock::new();

/// Returns true if kernel-level tracing is enabled.
pub fn enabled() -> bool {
    *TRACING_ENABLED.get_or_init(|| {
        let result = std::env::var("GGUF_TRACE_KERNEL").is_ok();
        eprintln!("[debug] enabled(): {}", result);
        result
    })
}

/// Returns the tracing format: "text" or "json".
pub fn format() -> &'static str {
    TRACING_FORMAT.get_or_init(|| {
        match std::env::var("GGUF_TRACE_KERNEL").as_deref() {
            Ok("json") => "json",
            _ => "text",
        }
    })
}

/// A single kernel trace entry.
#[derive(Debug, Clone)]
pub struct TraceEntry {
    pub kernel: &'static str,
    pub time_ms: f64,
    pub tensor_bytes: Option<u64>,
}

/// Global trace buffer — collected when `GGUF_TRACE_KERNEL=json`.
static TRACE_BUFFER: OnceLock<Mutex<Vec<TraceEntry>>> = OnceLock::new();

/// Collect a trace entry (called from Timer::drop). Only active for json format.
pub fn collect_trace(kernel: &'static str, time_ms: f64, tensor_bytes: Option<u64>) {
    if format() != "json" {
        return;
    }
    let buf = TRACE_BUFFER.get_or_init(|| Mutex::new(Vec::new()));
    buf.lock().unwrap().push(TraceEntry {
        kernel,
        time_ms,
        tensor_bytes,
    });
    // Debug: print first few entries
    if format() == "json" && buf.lock().unwrap().len() <= 3 {
        eprintln!("[debug] collect_trace: {} {}ms", kernel, time_ms);
    }
}

/// Dump the collected trace entries as JSON.
/// Call this at the end of a benchmark run.
///
/// If `GGUF_TRACE_KERNEL=1` (text mode), this also prints eprintln's.
pub fn dump_trace() {
    eprintln!("[debug] dump_trace: called");
    let buf = TRACE_BUFFER.get_or_init(|| Mutex::new(Vec::new()));
    let entries = buf.lock().unwrap();
    if entries.is_empty() {
        eprintln!("[debug] dump_trace: no entries");
        return;
    }
    // Print summary: total kernels and total time
    let total_time: f64 = entries.iter().map(|e| e.time_ms).sum();
    let total_bytes: u64 = entries.iter().filter_map(|e| e.tensor_bytes).sum();
    let total_gb = total_bytes as f64 / 1e9;
    let total_gb_s = total_gb / (total_time / 1e3);
    let kernel_count = entries.len();

    if format() == "json" {
        // JSON output
        eprintln!(
            "{}",
            serde_json::json!({
                "summary": {
                    "kernel_count": kernel_count,
                    "total_time_ms": total_time,
                    "total_bytes": total_bytes,
                    "total_gb_s": (total_gb_s * 100.0).round() / 100.0,
                },
                "kernels": entries.iter().map(|e| {
                    serde_json::json!({
                        "kernel": e.kernel,
                        "time_ms": (e.time_ms * 100.0).round() / 100.0,
                        "effective_bandwidth_gb_s": e.tensor_bytes.map(|b| {
                            (b as f64 / 1e9 / (e.time_ms / 1e3) * 100.0).round() / 100.0
                        }),
                    })
                }).collect::<Vec<_>>(),
            })
        );
    } else {
        // Text output (existing behavior)
        eprintln!(
            "\n=== Kernel Trace Summary ==="
        );
        eprintln!(
            "  {} kernels, {} ms total, {} GB total bandwidth ({} GB/s)",
            kernel_count, total_time, total_gb, total_gb_s
        );
        for e in entries.iter() {
            let bw = e.tensor_bytes.map(|b| {
                let gb = b as f64 / 1e9;
                let s = e.time_ms / 1e3;
                (gb / s * 100.0).round() / 100.0
            });
            if let Some(bw) = bw {
                eprintln!(
                    "  {}  {}ms  {} GB/s",
                    e.kernel, e.time_ms, bw
                );
            } else {
                eprintln!(
                    "  {}  {}ms",
                    e.kernel, e.time_ms
                );
            }
        }
    }
}

/// A simple high-resolution timer for GPU kernels.
///
/// Wraps `std::time::Instant` and prints the elapsed time when dropped,
/// but only if tracing is enabled.
pub struct Timer {
    start: std::time::Instant,
    name: &'static str,
    tensor_bytes: Option<u64>,
}

impl Timer {
    pub fn new(name: &'static str) -> Self {
        if enabled() {
            eprintln!("[debug] Timer::new: {}", name);
            Self {
                start: std::time::Instant::now(),
                name,
                tensor_bytes: None,
            }
        } else {
            Self {
                start: std::time::Instant::now(),
                name: "",
                tensor_bytes: None,
            }
        }
    }

    /// Create a timer with tensor size info for bandwidth calculation.
    /// The size should be in bytes.
    pub fn new_with_size(name: &'static str, tensor_bytes: u64) -> Self {
        if enabled() {
            Self {
                start: std::time::Instant::now(),
                name,
                tensor_bytes: Some(tensor_bytes),
            }
        } else {
            Self {
                start: std::time::Instant::now(),
                name: "",
                tensor_bytes: None,
            }
        }
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        if self.name.is_empty() {
            return;
        }
        let elapsed = self.start.elapsed().as_secs_f64() * 1e3;
        if format() == "json" {
            eprintln!("[debug] Timer::drop: {} {}ms", self.name, elapsed);
            collect_trace(self.name, elapsed, self.tensor_bytes);
        } else {
            let bw = self.tensor_bytes.map(|b| {
                let gb = b as f64 / 1e9;
                let s = elapsed / 1e3;
                (gb / s * 100.0).round() / 100.0
            });
            if let Some(bw) = bw {
                eprintln!(
                    "  [kernel] {} {}ms  {} GB/s",
                    self.name, elapsed, bw
                );
            } else {
                eprintln!(
                    "  [kernel] {} {}ms",
                    self.name, elapsed
                );
            }
        }
    }
}
