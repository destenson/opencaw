use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};

/// The three operational tiers from the design doc.
/// Degradation is per-component and can recover.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatingTier {
    /// All systems healthy: automatic probes + tool-based reads + eviction + consolidation
    FullRecall,
    /// Embedding service slow/down, or probe rate excessive:
    /// stubs still generated, but no automatic recall. Model uses tools manually.
    StubsAndToolsOnly,
    /// Summary cache cold, stub generation failing:
    /// prompt passed through unmodified.
    PassThrough,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Down,
}

/// Tracks health for a single component (embedding service, summary generator, etc.).
/// Auto-degrades when error rate or latency exceeds thresholds.
/// Requires N consecutive successes to recover (hysteresis prevents flapping).
#[derive(Debug, Clone)]
pub struct ComponentHealth {
    status: HealthStatus,
    error_count: usize,
    total_calls: usize,
    last_latency_ms: u64,
    /// Consecutive successes since last failure — used for recovery hysteresis
    consecutive_successes: usize,
    config: ComponentHealthConfig,
}

#[derive(Debug, Clone)]
pub struct ComponentHealthConfig {
    pub latency_threshold_ms: u64,
    pub error_rate_threshold: f32,
    /// How many consecutive successes needed to recover from Degraded → Healthy
    pub recovery_threshold: usize,
    /// Minimum calls before error rate is meaningful
    pub min_calls_for_rate: usize,
}

impl Default for ComponentHealthConfig {
    fn default() -> Self {
        Self {
            latency_threshold_ms: 2000,
            error_rate_threshold: 0.3,
            recovery_threshold: 5,
            min_calls_for_rate: 10,
        }
    }
}

impl ComponentHealth {
    pub fn with_config(config: ComponentHealthConfig) -> Self {
        Self {
            status: HealthStatus::Healthy,
            error_count: 0,
            total_calls: 0,
            last_latency_ms: 0,
            consecutive_successes: 0,
            config,
        }
    }

    pub fn status(&self) -> HealthStatus {
        self.status
    }

    pub fn error_rate(&self) -> f32 {
        if self.total_calls == 0 {
            return 0.0;
        }
        self.error_count as f32 / self.total_calls as f32
    }

    pub fn record_call(&mut self, latency_ms: u64, success: bool) {
        self.total_calls += 1;
        self.last_latency_ms = latency_ms;

        if !success {
            self.error_count += 1;
            self.consecutive_successes = 0;
        } else {
            self.consecutive_successes += 1;
        }

        self.reevaluate();
    }

    fn reevaluate(&mut self) {
        match self.status {
            HealthStatus::Healthy => {
                if self.should_degrade() {
                    self.status = HealthStatus::Degraded;
                }
            }
            HealthStatus::Degraded => {
                if self.should_mark_down() {
                    self.status = HealthStatus::Down;
                } else if self.consecutive_successes >= self.config.recovery_threshold {
                    self.status = HealthStatus::Healthy;
                    self.consecutive_successes = 0;
                }
            }
            HealthStatus::Down => {
                if self.consecutive_successes >= self.config.recovery_threshold {
                    self.status = HealthStatus::Degraded;
                    self.consecutive_successes = 0;
                }
            }
        }
    }

    fn should_degrade(&self) -> bool {
        let high_latency = self.last_latency_ms > self.config.latency_threshold_ms;
        let high_error_rate = self.total_calls >= self.config.min_calls_for_rate
            && self.error_rate() > self.config.error_rate_threshold;
        high_latency || high_error_rate
    }

    /// Down requires both high error rate AND recent failure (not just one slow call)
    fn should_mark_down(&self) -> bool {
        self.total_calls >= self.config.min_calls_for_rate
            && self.error_rate() > self.config.error_rate_threshold * 2.0
            && self.consecutive_successes == 0
    }
}

/// Tracks probe frequency per session. If the model emits probes too fast,
/// the system throttles to tools-only mode to prevent thrashing or gaming.
#[derive(Debug, Clone)]
pub struct ProbeRateLimiter {
    /// Timestamps (unix seconds) of recent probes within the window
    recent_probes: VecDeque<u64>,
    config: ProbeRateConfig,
}

#[derive(Debug, Clone)]
pub struct ProbeRateConfig {
    /// Maximum probes allowed within the window before throttling
    pub max_probes: usize,
    /// Window size in seconds
    pub window_secs: u64,
}

impl Default for ProbeRateConfig {
    fn default() -> Self {
        Self {
            max_probes: 10,
            window_secs: 60,
        }
    }
}

impl ProbeRateLimiter {
    pub fn with_config(config: ProbeRateConfig) -> Self {
        Self {
            recent_probes: VecDeque::new(),
            config,
        }
    }

    pub fn record_probe(&mut self) {
        let now = current_timestamp();
        self.recent_probes.push_back(now);
        self.expire_old(now);
    }

    pub fn record_probe_at(&mut self, timestamp: u64) {
        self.recent_probes.push_back(timestamp);
        self.expire_old(timestamp);
    }

    pub fn is_throttled(&self) -> bool {
        self.recent_probes.len() >= self.config.max_probes
    }

    pub fn recent_count(&self) -> usize {
        self.recent_probes.len()
    }

    fn expire_old(&mut self, now: u64) {
        let cutoff = now.saturating_sub(self.config.window_secs);
        while let Some(&oldest) = self.recent_probes.front() {
            if oldest < cutoff {
                self.recent_probes.pop_front();
            } else {
                break;
            }
        }
    }
}

/// Central monitor that combines per-component health into an operating tier.
/// Add to DynamicRecallOrchestrator to enable graceful degradation.
#[derive(Debug, Clone)]
pub struct DegradationMonitor {
    pub embedding_health: ComponentHealth,
    pub summary_health: ComponentHealth,
    pub probe_rate: ProbeRateLimiter,
}

/// Configuration for constructing a DegradationMonitor
#[derive(Debug, Clone)]
pub struct DegradationMonitorConfig {
    pub embedding: ComponentHealthConfig,
    pub summary: ComponentHealthConfig,
    pub probe_rate: ProbeRateConfig,
}

impl Default for DegradationMonitorConfig {
    fn default() -> Self {
        Self {
            embedding: ComponentHealthConfig {
                latency_threshold_ms: 2000,
                error_rate_threshold: 0.3,
                recovery_threshold: 5,
                min_calls_for_rate: 10,
            },
            summary: ComponentHealthConfig {
                latency_threshold_ms: 5000,
                error_rate_threshold: 0.3,
                recovery_threshold: 5,
                min_calls_for_rate: 10,
            },
            probe_rate: ProbeRateConfig::default(),
        }
    }
}

impl DegradationMonitor {
    pub fn with_config(config: DegradationMonitorConfig) -> Self {
        Self {
            embedding_health: ComponentHealth::with_config(config.embedding),
            summary_health: ComponentHealth::with_config(config.summary),
            probe_rate: ProbeRateLimiter::with_config(config.probe_rate),
        }
    }

    pub fn record_embedding_call(&mut self, latency_ms: u64, success: bool) {
        self.embedding_health.record_call(latency_ms, success);
    }

    pub fn record_summary_call(&mut self, latency_ms: u64, success: bool) {
        self.summary_health.record_call(latency_ms, success);
    }

    pub fn record_probe(&mut self) {
        self.probe_rate.record_probe();
    }

    pub fn record_probe_at(&mut self, timestamp: u64) {
        self.probe_rate.record_probe_at(timestamp);
    }

    /// Determine which tier the system should operate at based on all component health.
    pub fn effective_tier(&self) -> OperatingTier {
        // Summary service down → can't generate stubs → pass-through
        if self.summary_health.status() == HealthStatus::Down {
            return OperatingTier::PassThrough;
        }

        // Embedding service unhealthy or probes throttled → stubs+tools only
        if self.embedding_health.status() != HealthStatus::Healthy || self.probe_rate.is_throttled()
        {
            return OperatingTier::StubsAndToolsOnly;
        }

        OperatingTier::FullRecall
    }

    /// Whether automatic recall (probes + thinking-trace matching) should run.
    /// False when embedding service is degraded or probe rate is excessive.
    pub fn should_auto_recall(&self) -> bool {
        self.effective_tier() == OperatingTier::FullRecall
    }

    /// Whether stub generation should run.
    /// False only when the summary service is completely down.
    pub fn should_generate_stubs(&self) -> bool {
        self.effective_tier() != OperatingTier::PassThrough
    }
}

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn healthy_by_default() {
        let monitor = DegradationMonitor::with_config(DegradationMonitorConfig::default());
        assert_eq!(monitor.effective_tier(), OperatingTier::FullRecall);
        assert!(monitor.should_auto_recall());
        assert!(monitor.should_generate_stubs());
    }

    #[test]
    fn embedding_latency_degrades_to_stubs_only() {
        let mut monitor = DegradationMonitor::with_config(DegradationMonitorConfig::default());
        // One high-latency call triggers degradation
        monitor.record_embedding_call(5000, true);
        assert_eq!(monitor.effective_tier(), OperatingTier::StubsAndToolsOnly);
        assert!(!monitor.should_auto_recall());
        assert!(monitor.should_generate_stubs());
    }

    #[test]
    fn embedding_recovers_after_consecutive_successes() {
        let config = DegradationMonitorConfig {
            embedding: ComponentHealthConfig {
                recovery_threshold: 3,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut monitor = DegradationMonitor::with_config(config);

        // Degrade
        monitor.record_embedding_call(5000, true);
        assert_eq!(monitor.effective_tier(), OperatingTier::StubsAndToolsOnly);

        // Not enough successes yet
        monitor.record_embedding_call(100, true);
        monitor.record_embedding_call(100, true);
        assert_eq!(monitor.effective_tier(), OperatingTier::StubsAndToolsOnly);

        // Third success triggers recovery
        monitor.record_embedding_call(100, true);
        assert_eq!(monitor.effective_tier(), OperatingTier::FullRecall);
    }

    #[test]
    fn summary_down_falls_to_passthrough() {
        let config = DegradationMonitorConfig {
            summary: ComponentHealthConfig {
                min_calls_for_rate: 2,
                error_rate_threshold: 0.2,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut monitor = DegradationMonitor::with_config(config);

        // First: degrade summary
        monitor.record_summary_call(10000, true);
        assert!(monitor.should_generate_stubs()); // degraded but not down

        // Slam with failures to push to Down (need >2× error rate threshold with 0 consecutive successes)
        for _ in 0..10 {
            monitor.record_summary_call(100, false);
        }
        assert_eq!(monitor.effective_tier(), OperatingTier::PassThrough);
        assert!(!monitor.should_generate_stubs());
    }

    #[test]
    fn probe_rate_throttles_to_stubs_only() {
        let config = DegradationMonitorConfig {
            probe_rate: ProbeRateConfig {
                max_probes: 3,
                window_secs: 60,
            },
            ..Default::default()
        };
        let mut monitor = DegradationMonitor::with_config(config);

        let now = 1000;
        monitor.record_probe_at(now);
        monitor.record_probe_at(now + 1);
        assert_eq!(monitor.effective_tier(), OperatingTier::FullRecall);

        monitor.record_probe_at(now + 2);
        assert_eq!(monitor.effective_tier(), OperatingTier::StubsAndToolsOnly);
    }

    #[test]
    fn probe_rate_recovers_after_window_expires() {
        let config = DegradationMonitorConfig {
            probe_rate: ProbeRateConfig {
                max_probes: 2,
                window_secs: 10,
            },
            ..Default::default()
        };
        let mut monitor = DegradationMonitor::with_config(config);

        monitor.record_probe_at(100);
        monitor.record_probe_at(101);
        assert!(monitor.probe_rate.is_throttled());

        // Record a probe well past the window — old ones expire
        monitor.record_probe_at(200);
        assert!(!monitor.probe_rate.is_throttled());
    }

    #[test]
    fn error_rate_needs_minimum_calls() {
        let config = DegradationMonitorConfig {
            embedding: ComponentHealthConfig {
                min_calls_for_rate: 5,
                error_rate_threshold: 0.3,
                latency_threshold_ms: 10000, // high so latency doesn't trigger
                ..Default::default()
            },
            ..Default::default()
        };
        let mut monitor = DegradationMonitor::with_config(config);

        // 2 failures out of 3 calls — high rate but below min_calls
        monitor.record_embedding_call(100, false);
        monitor.record_embedding_call(100, false);
        monitor.record_embedding_call(100, true);
        assert_eq!(monitor.effective_tier(), OperatingTier::FullRecall);
    }
}
