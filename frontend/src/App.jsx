import { useState, useEffect, useCallback } from "react";
import axios from "axios";
import "./App.css";

const API = "https://llm-driven-stock-scanner-production.up.railway.app/api";

// ─────────────────────────────────────────────────────────────
// Translations
// ─────────────────────────────────────────────────────────────

const T = {
  en: {
    title:          "Stock Scanner",
    subtitle:       "Regime-adaptive · LLM-filtered",
    lastRun:        "Last run",
    topNLabel:      "Top N per side",
    factorScan:     "⚡ Factor Scan",
    fullScan:       "✦ Full Scan (LLM included)",
    scanning:       "Scanning…",
    analyzing:      "Analyzing…",
    longTitle:      "▲ Long Candidates",
    shortTitle:     "▼ Short Candidates",
    noCandidates:   "No candidates",
    noScanTitle:    "No scan data",
    noScanHint:     'Click "Factor Scan" for a quick scan, or "Full Scan (LLM included)" for complete analysis.',
    loadingLlm:     "Running LLM analysis… (~30–60s)",
    loadingData:    "Fetching data…",
    recentNews:     "Recent news",
    factorOnly:     "Factor signal only",
    vix:            "VIX",
    realizedVol:    "Realized Vol",
    trend:          "Trend",
    factors:        "Factors",
    scanFailed:     "Scan failed",
    fullScanFailed: "Full scan failed",
    regime: {
      TRENDING: "TRENDING",
      VOLATILE: "VOLATILE",
      NEUTRAL:  "NEUTRAL",
    },
    signal: {
      STRONG_BUY: "STRONG BUY",
      BUY:        "BUY",
      NEUTRAL:    "NEUTRAL",
      AVOID:      "AVOID",
    },
  },
  zh: {
    title:          "股票扫描器",
    subtitle:       "市场环境自适应 · LLM过滤",
    lastRun:        "上次运行",
    topNLabel:      "每侧显示数量",
    factorScan:     "⚡ 因子扫描",
    fullScan:       "✦ 完整扫描（含AI大模型分析）",
    scanning:       "扫描中…",
    analyzing:      "分析中…",
    longTitle:      "▲ 做多候选",
    shortTitle:     "▼ 做空候选",
    noCandidates:   "暂无候选",
    noScanTitle:    "暂无扫描数据",
    noScanHint:     '点击"因子扫描"快速扫描，或"完整扫描"获取LLM分析。',
    loadingLlm:     "AI分析中… (约30-60秒)",
    loadingData:    "数据加载中…",
    recentNews:     "相关新闻",
    factorOnly:     "仅因子信号",
    vix:            "恐慌指数",
    realizedVol:    "已实现波动率",
    trend:          "趋势强度",
    factors:        "使用因子",
    scanFailed:     "扫描失败",
    fullScanFailed: "完整扫描失败",
    regime: {
      TRENDING: "趋势市",
      VOLATILE: "震荡市",
      NEUTRAL:  "中性",
    },
    signal: {
      STRONG_BUY: "强烈买入",
      BUY:        "买入",
      NEUTRAL:    "中性",
      AVOID:      "回避",
    },
  },
};

// ─────────────────────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────────────────────

function LangToggle({ lang, setLang }) {
  return (
    <button
      className="lang-toggle"
      onClick={() => setLang(lang === "en" ? "zh" : "en")}
      title="Switch language / 切换语言"
    >
      {lang === "en" ? "中文" : "EN"}
    </button>
  );
}

function RegimeBadge({ regime, t }) {
  const colors = {
    TRENDING: "badge-trending",
    VOLATILE: "badge-volatile",
    NEUTRAL:  "badge-neutral",
  };
  return (
    <span className={`badge ${colors[regime] || "badge-neutral"}`}>
      {t.regime[regime] || regime}
    </span>
  );
}

function SignalBadge({ signal, t }) {
  const colors = {
    STRONG_BUY: "signal-strong-buy",
    BUY:        "signal-buy",
    NEUTRAL:    "signal-neutral",
    AVOID:      "signal-avoid",
  };
  return (
    <span className={`signal ${colors[signal] || "signal-neutral"}`}>
      {t.signal[signal] || signal || "—"}
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

function StockCard({ item, side, t }) {
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
          {hasLlm && <SignalBadge signal={item.signal} t={t} />}
        </div>
        <div className="card-right">
          <span className="score-label">score</span>
          <span className="score-value">
            {item.score?.toFixed(3) ?? item.factor_score?.toFixed(3)}
          </span>
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
              <p className="news-heading">{t.recentNews}</p>
              {item.news_titles.map((title, i) => (
                <p key={i} className="news-item">· {title}</p>
              ))}
            </div>
          )}
        </div>
      )}

      {!hasLlm && (
        <p className="no-llm">{t.factorOnly}</p>
      )}
    </div>
  );
}

function RegimePanel({ regime, t }) {
  if (!regime) return null;
  return (
    <div className="regime-panel">
      <div className="regime-row">
        <RegimeBadge regime={regime.regime} t={t} />
        <span className="regime-desc">{regime.description}</span>
      </div>
      <div className="regime-stats">
        <div className="stat">
          <span className="stat-label">{t.vix}</span>
          <span className="stat-value">{regime.vix ?? "—"}</span>
        </div>
        <div className="stat">
          <span className="stat-label">{t.realizedVol}</span>
          <span className="stat-value">
            {regime.realized_vol ? (regime.realized_vol * 100).toFixed(1) + "%" : "—"}
          </span>
        </div>
        <div className="stat">
          <span className="stat-label">{t.trend}</span>
          <span className="stat-value">
            {regime.trend_strength ? (regime.trend_strength * 100).toFixed(0) + "%" : "—"}
          </span>
        </div>
        <div className="stat">
          <span className="stat-label">{t.factors}</span>
          <span className="stat-value factors">
            {regime.recommended_factors?.join(", ")}
          </span>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Main App
// ─────────────────────────────────────────────────────────────

export default function App() {
  const [lang, setLang]             = useState("en");
  const [regime, setRegime]         = useState(null);
  const [scanData, setScanData]     = useState(null);
  const [loading, setLoading]       = useState(false);
  const [llmLoading, setLlmLoading] = useState(false);
  const [error, setError]           = useState(null);
  const [lastRun, setLastRun]       = useState(null);
  const [hasLlm, setHasLlm]         = useState(false);
  const [topN, setTopN]             = useState(5);

  const t = T[lang];

  useEffect(() => {
    axios.get(`${API}/latest`)
      .then(r => {
        setScanData(r.data);
        setHasLlm(true);
        setLastRun(r.data.timestamp);
      })
      .catch(() => {});

    axios.get(`${API}/regime`)
      .then(r => setRegime(r.data))
      .catch(() => {});
  }, []);

  const runFactorScan = useCallback(async () => {
    setLoading(true);
    setError(null);
    setHasLlm(false);
    try {
      const r = await axios.post(`${API}/scan/full?top_n=${topN}&save=true&lang=${lang}`);
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
      setError(e.response?.data?.detail || t.scanFailed);
    } finally {
      setLoading(false);
    }
  }, [topN, t]);

  const runFullScan = useCallback(async () => {
    setLlmLoading(true);
    setError(null);
    try {
      await runFactorScan();
      const r = await axios.post(`${API}/scan/full?top_n=${topN}&save=true`);
      setScanData(r.data);
      setHasLlm(true);
      setLastRun(r.data.timestamp);
      const reg = await axios.get(`${API}/regime`);
      setRegime(reg.data);
    } catch (e) {
      setError(e.response?.data?.detail || t.fullScanFailed);
    } finally {
      setLlmLoading(false);
      setLoading(false);
    }
  }, [topN, t, runFactorScan]);

  const longList  = scanData?.long_watchlist  || scanData?.long_candidates  || [];
  const shortList = scanData?.short_watchlist || scanData?.short_candidates || [];

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-left">
          <h1 className="title">
            <span className="title-accent">◈</span> {t.title}
          </h1>
          <p className="subtitle">{t.subtitle}</p>
        </div>
        <div className="header-right">
          {lastRun && (
            <span className="last-run">{t.lastRun}: {lastRun}</span>
          )}
          <LangToggle lang={lang} setLang={setLang} />
        </div>
      </header>

      {/* Controls */}
      <div className="controls">
        <div className="control-group">
          <label className="control-label">{t.topNLabel}</label>
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
          {loading ? t.scanning : t.factorScan}
        </button>

        <button
          className={`btn ${hasLlm ? "btn-active" : "btn-secondary"}`}
          onClick={runFullScan}
          disabled={loading || llmLoading}
        >
          {llmLoading ? t.analyzing : t.fullScan}
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="error-banner">{error}</div>
      )}

      {/* Regime */}
      <RegimePanel regime={regime} t={t} />

      {/* Watchlist */}
      {scanData && (
        <div className="watchlist">
          <div className="watchlist-col">
            <div className="col-header long-header">
              <span className="col-title">{t.longTitle}</span>
              <span className="col-count">{longList.length}</span>
            </div>
            <div className="card-list">
              {longList.map(item => (
                <StockCard key={item.ticker} item={item} side="long" t={t} />
              ))}
              {longList.length === 0 && (
                <p className="empty">{t.noCandidates}</p>
              )}
            </div>
          </div>

          <div className="watchlist-col">
            <div className="col-header short-header">
              <span className="col-title">{t.shortTitle}</span>
              <span className="col-count">{shortList.length}</span>
            </div>
            <div className="card-list">
              {shortList.map(item => (
                <StockCard key={item.ticker} item={item} side="short" t={t} />
              ))}
              {shortList.length === 0 && (
                <p className="empty">{t.noCandidates}</p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Empty state */}
      {!scanData && !loading && !llmLoading && (
        <div className="empty-state">
          <p className="empty-title">{t.noScanTitle}</p>
          <p className="empty-hint">{t.noScanHint}</p>
        </div>
      )}

      {/* Loading */}
      {(loading || llmLoading) && !scanData && (
        <div className="loading-state">
          <div className="spinner" />
          <p>{llmLoading ? t.loadingLlm : t.loadingData}</p>
        </div>
      )}
    </div>
  );
}
