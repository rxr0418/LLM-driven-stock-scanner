import { useState, useEffect, useCallback } from "react";
import axios from "axios";
import "./App.css";

const API = "http://localhost:8000/api";

// ─────────────────────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────────────────────

function RegimeBadge({ regime }) {
  const colors = {
    TRENDING: "badge-trending",
    VOLATILE: "badge-volatile",
    NEUTRAL:  "badge-neutral",
  };
  return (
    <span className={`badge ${colors[regime] || "badge-neutral"}`}>
      {regime}
    </span>
  );
}

function SignalBadge({ signal }) {
  const colors = {
    STRONG_BUY: "signal-strong-buy",
    BUY:        "signal-buy",
    NEUTRAL:    "signal-neutral",
    AVOID:      "signal-avoid",
  };
  return (
    <span className={`signal ${colors[signal] || "signal-neutral"}`}>
      {signal || "—"}
    </span>
  );
}

function ConfidenceBar({ value }) {
  const color =
    value >= 75 ? "#22c55e" :
    value >= 50 ? "#f59e0b" :
    "#ef4444";
  return (
    <div className="conf-bar-bg">
      <div
        className="conf-bar-fill"
        style={{ width: `${value}%`, background: color }}
      />
      <span className="conf-label">{value}%</span>
    </div>
  );
}

function StockCard({ item, side }) {
  const [expanded, setExpanded] = useState(false);
  const hasLlm = item.signal && item.signal !== undefined;

  return (
    <div
      className={`stock-card ${side === "long" ? "card-long" : "card-short"} ${expanded ? "expanded" : ""}`}
      onClick={() => setExpanded(!expanded)}
    >
      <div className="card-header">
        <div className="card-left">
          <span className="ticker">{item.ticker}</span>
          {hasLlm && <SignalBadge signal={item.signal} />}
        </div>
        <div className="card-right">
          <span className="score-label">score</span>
          <span className="score-value">{item.score?.toFixed(3) ?? item.factor_score?.toFixed(3)}</span>
        </div>
      </div>

      {hasLlm && (
        <div className="card-body">
          <ConfidenceBar value={item.confidence || 0} />
          {item.reason && (
            <p className="reason">{item.reason}</p>
          )}
          {expanded && item.risk_flag && item.risk_flag !== "none" && (
            <p className="risk-flag">⚠ {item.risk_flag}</p>
          )}
          {expanded && item.news_titles && item.news_titles.length > 0 && (
            <div className="news-list">
              <p className="news-heading">Recent news</p>
              {item.news_titles.map((t, i) => (
                <p key={i} className="news-item">· {t}</p>
              ))}
            </div>
          )}
        </div>
      )}

      {!hasLlm && (
        <p className="no-llm">Factor signal only</p>
      )}
    </div>
  );
}

function RegimePanel({ regime }) {
  if (!regime) return null;
  return (
    <div className="regime-panel">
      <div className="regime-row">
        <RegimeBadge regime={regime.regime} />
        <span className="regime-desc">{regime.description}</span>
      </div>
      <div className="regime-stats">
        <div className="stat">
          <span className="stat-label">VIX</span>
          <span className="stat-value">{regime.vix ?? "—"}</span>
        </div>
        <div className="stat">
          <span className="stat-label">Realized Vol</span>
          <span className="stat-value">
            {regime.realized_vol ? (regime.realized_vol * 100).toFixed(1) + "%" : "—"}
          </span>
        </div>
        <div className="stat">
          <span className="stat-label">Trend</span>
          <span className="stat-value">
            {regime.trend_strength ? (regime.trend_strength * 100).toFixed(0) + "%" : "—"}
          </span>
        </div>
        <div className="stat">
          <span className="stat-label">Factors</span>
          <span className="stat-value factors">{regime.recommended_factors?.join(", ")}</span>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Main App
// ─────────────────────────────────────────────────────────────

export default function App() {
  const [regime, setRegime]       = useState(null);
  const [scanData, setScanData]   = useState(null);
  const [loading, setLoading]     = useState(false);
  const [llmLoading, setLlmLoading] = useState(false);
  const [error, setError]         = useState(null);
  const [lastRun, setLastRun]     = useState(null);
  const [hasLlm, setHasLlm]       = useState(false);
  const [topN, setTopN]           = useState(5);

  // Load latest saved result on mount
  useEffect(() => {
    axios.get(`${API}/latest`)
      .then(r => {
        setScanData(r.data);
        setHasLlm(true);
        setLastRun(r.data.timestamp);
      })
      .catch(() => {}); // no saved results yet, that's fine

    axios.get(`${API}/regime`)
      .then(r => setRegime(r.data))
      .catch(() => {});
  }, []);

  const runFactorScan = useCallback(async () => {
    setLoading(true);
    setError(null);
    setHasLlm(false);
    try {
      const r = await axios.get(`${API}/scan?top_n=${topN}`);
      setScanData(r.data);
      setRegime({
        regime:              r.data.regime,
        description:         r.data.description,
        recommended_factors: r.data.factors_used,
        vix:                 r.data.vix,
        realized_vol:        r.data.realized_vol,
        trend_strength:      r.data.trend_strength,
      });
      setLastRun(r.data.timestamp);
    } catch (e) {
      setError(e.response?.data?.detail || "Scan failed");
    } finally {
      setLoading(false);
    }
  }, [topN]);

  const runFullScan = useCallback(async () => {
    setLlmLoading(true);
    setError(null);
    try {
      // First run factor scan for immediate feedback
      await runFactorScan();
      // Then run full LLM scan
      const r = await axios.post(`${API}/scan/full?top_n=${topN}&save=true`);
      setScanData(r.data);
      setHasLlm(true);
      setLastRun(r.data.timestamp);
      // Refresh regime
      const reg = await axios.get(`${API}/regime`);
      setRegime(reg.data);
    } catch (e) {
      setError(e.response?.data?.detail || "Full scan failed");
    } finally {
      setLlmLoading(false);
      setLoading(false);
    }
  }, [topN, runFactorScan]);

  const longList  = scanData?.long_watchlist  || scanData?.long_candidates  || [];
  const shortList = scanData?.short_watchlist || scanData?.short_candidates || [];

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-left">
          <h1 className="title">
            <span className="title-accent">◈</span> Stock Scanner
          </h1>
          <p className="subtitle">Regime-adaptive · LLM-filtered</p>
        </div>
        <div className="header-right">
          {lastRun && (
            <span className="last-run">Last run: {lastRun}</span>
          )}
        </div>
      </header>

      {/* Controls */}
      <div className="controls">
        <div className="control-group">
          <label className="control-label">Top N per side</label>
          <select
            className="control-select"
            value={topN}
            onChange={e => setTopN(Number(e.target.value))}
          >
            {[3, 5, 10, 15].map(n => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
        </div>

        <button
          className={`btn ${!hasLlm ? "btn-active" : "btn-secondary"}`}
          onClick={runFactorScan}
          disabled={loading || llmLoading}
        >
          {loading ? "Scanning…" : "⚡ Factor Scan"}
        </button>

        <button
          className={`btn ${hasLlm ? "btn-active" : "btn-secondary"}`}
          onClick={runFullScan}
          disabled={loading || llmLoading}
        >
          {llmLoading ? "Analyzing…" : "✦ Full Scan (LLM included)"}
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="error-banner">{error}</div>
      )}

      {/* Regime */}
      <RegimePanel regime={regime} />

      {/* Watchlist */}
      {scanData && (
        <div className="watchlist">
          {/* Long */}
          <div className="watchlist-col">
            <div className="col-header long-header">
              <span className="col-title">▲ Long Candidates</span>
              <span className="col-count">{longList.length}</span>
            </div>
            <div className="card-list">
              {longList.map(item => (
                <StockCard
                  key={item.ticker}
                  item={item}
                  side="long"
                />
              ))}
              {longList.length === 0 && (
                <p className="empty">No candidates</p>
              )}
            </div>
          </div>

          {/* Short */}
          <div className="watchlist-col">
            <div className="col-header short-header">
              <span className="col-title">▼ Short Candidates</span>
              <span className="col-count">{shortList.length}</span>
            </div>
            <div className="card-list">
              {shortList.map(item => (
                <StockCard
                  key={item.ticker}
                  item={item}
                  side="short"
                />
              ))}
              {shortList.length === 0 && (
                <p className="empty">No candidates</p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Empty state */}
      {!scanData && !loading && !llmLoading && (
        <div className="empty-state">
          <p className="empty-title">No scan data</p>
          <p className="empty-hint">Click "Factor Scan" for a quick scan, or "Full Scan + LLM" for complete analysis.</p>
        </div>
      )}

      {/* Loading overlay */}
      {(loading || llmLoading) && !scanData && (
        <div className="loading-state">
          <div className="spinner" />
          <p>{llmLoading ? "Running LLM analysis… (~30–60s)" : "Fetching data…"}</p>
        </div>
      )}
    </div>
  );
}
