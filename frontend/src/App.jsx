import { useState, useEffect, useCallback } from "react";
import axios from "axios";
import "./App.css";

const API  = "https://llm-driven-stock-scanner-production.up.railway.app/api";
const TABS = { SWING: "swing", PREMARKET: "premarket" };

// ─────────────────────────────────────────────────────────────
// Translations
// ─────────────────────────────────────────────────────────────

const T = {
  en: {
    title:           "Stock Scanner",
    subtitle:        "Regime-adaptive · LLM-filtered",
    lastRun:         "Last run",
    topNLabel:       "Top N per side",
    tabSwing:        "📊 Swing Trade",
    tabDayTrade:     "🌅 Day Trade",
    factorScan:      "⚡ Factor Scan",
    fullScan:        "✦ Full Scan (LLM included)",
    quickScan:       "⚡ Quick Scan",
    deepScan:        "✦ Deep Scan (LLM included)",
    scanning:        "Scanning…",
    analyzing:       "Analyzing…",
    longTitle:       "▲ Long Candidates",
    shortTitle:      "▼ Short Candidates",
    pmTitle:         "🌅 Premarket Movers",
    noCandidates:    "No candidates",
    noScanTitle:     "No scan data",
    noSwingHint:     'Click "Factor Scan" for a quick scan, or "Full Scan" for LLM analysis.',
    noPmHint:        "Best run between 4:00–9:30 AM ET. Click Quick Scan to check for movers.",
    loadingLlm:      "Running LLM analysis… (~30–60s)",
    loadingData:     "Fetching data…",
    recentNews:      "Recent news",
    factorOnly:      "Factor signal only",
    dataOnly:        "Data only — no LLM analysis",
    vix:             "VIX",
    realizedVol:     "Realized Vol",
    trend:           "Trend",
    factors:         "Factors",
    scanFailed:      "Scan failed",
    fullScanFailed:  "Full scan failed",
    change:          "Change",
    rvol:            "RVOL",
    volume:          "Volume",
    marketCap:       "Mkt Cap",
    float:           "Float",
    catalyst:        "Catalyst",
    risk:            "Risk",
    pmDisclaimer:    "Premarket data — best used 4:00–9:30 AM ET before market open.",
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
    pmSignal: {
      TRADE: "TRADE",
      WATCH: "WATCH",
      AVOID: "AVOID",
    },
  },
  zh: {
    title:           "股票扫描器",
    subtitle:        "市场环境自适应 · LLM过滤",
    lastRun:         "上次运行",
    topNLabel:       "每侧显示数量",
    tabSwing:        "📊 波段交易",
    tabDayTrade:     "🌅 日内交易",
    factorScan:      "⚡ 因子扫描",
    fullScan:        "✦ 完整扫描（含AI分析）",
    quickScan:       "⚡ 快速扫描",
    deepScan:        "✦ 深度扫描（含AI分析）",
    scanning:        "扫描中…",
    analyzing:       "分析中…",
    longTitle:       "▲ 做多候选",
    shortTitle:      "▼ 做空候选",
    pmTitle:         "🌅 盘前异动",
    noCandidates:    "暂无候选",
    noScanTitle:     "暂无扫描数据",
    noSwingHint:     '点击"因子扫描"快速扫描，或"完整扫描"获取AI分析。',
    noPmHint:        "最佳运行时间：美东时间4:00–9:30。点击快速扫描查看盘前异动。",
    loadingLlm:      "AI分析中… (约30-60秒)",
    loadingData:     "数据加载中…",
    recentNews:      "相关新闻",
    factorOnly:      "仅因子信号",
    dataOnly:        "仅数据，无AI分析",
    vix:             "恐慌指数",
    realizedVol:     "已实现波动率",
    trend:           "趋势强度",
    factors:         "使用因子",
    scanFailed:      "扫描失败",
    fullScanFailed:  "完整扫描失败",
    change:          "涨跌幅",
    rvol:            "相对量",
    volume:          "成交量",
    marketCap:       "市值",
    float:           "流通股",
    catalyst:        "催化剂",
    risk:            "风险",
    pmDisclaimer:    "盘前数据，建议在美东时间4:00–9:30使用。",
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
    pmSignal: {
      TRADE: "可交易",
      WATCH: "观察",
      AVOID: "回避",
    },
  },
};

// ─────────────────────────────────────────────────────────────
// Shared components
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

function TabBar({ activeTab, setActiveTab, t }) {
  return (
    <div className="tab-bar">
      <button
        className={`tab-btn ${activeTab === TABS.SWING ? "tab-active" : ""}`}
        onClick={() => setActiveTab(TABS.SWING)}
      >
        {t.tabSwing}
      </button>
      <button
        className={`tab-btn ${activeTab === TABS.PREMARKET ? "tab-active" : ""}`}
        onClick={() => setActiveTab(TABS.PREMARKET)}
      >
        {t.tabDayTrade}
      </button>
    </div>
  );
}

function ConfidenceBar({ value }) {
  const color =
    value >= 75 ? "#22c55e" :
    value >= 50 ? "#f59e0b" :
    "#ef4444";
  return (
    <div className="conf-bar-bg">
      <div className="conf-bar-fill" style={{ width: `${value}%`, background: color }} />
      <span className="conf-label">{value}%</span>
    </div>
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

function SignalBadge({ signal, t, type = "swing" }) {
  if (type === "premarket") {
    const colors = {
      TRADE: "signal-strong-buy",
      WATCH: "signal-neutral",
      AVOID: "signal-avoid",
    };
    return (
      <span className={`signal ${colors[signal] || "signal-neutral"}`}>
        {t.pmSignal[signal] || signal || "—"}
      </span>
    );
  }
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
// Swing Trade components
// ─────────────────────────────────────────────────────────────

function SwingCard({ item, side, t }) {
  const [expanded, setExpanded] = useState(false);
  const hasLlm = item.signal !== undefined && item.signal !== null;

  return (
    <div
      className={`stock-card ${side === "long" ? "card-long" : "card-short"}`}
      onClick={() => setExpanded(!expanded)}
    >
      <div className="card-header">
        <div className="card-left">
          <span className="ticker">{item.ticker}</span>
          {hasLlm && <SignalBadge signal={item.signal} t={t} type="swing" />}
        </div>
        <div className="card-right">
          <span className="score-label">score</span>
          <span className="score-value">
            {(item.score ?? item.factor_score ?? 0).toFixed(3)}
          </span>
        </div>
      </div>

      {hasLlm && (
        <div className="card-body">
          <ConfidenceBar value={item.confidence || 0} />
          {item.reason && <p className="reason">{item.reason}</p>}
          {expanded && item.risk_flag && item.risk_flag !== "none" && (
            <p className="risk-flag">⚠ {item.risk_flag}</p>
          )}
          {expanded && item.news_titles?.length > 0 && (
            <div className="news-list">
              <p className="news-heading">{t.recentNews}</p>
              {item.news_titles.map((title, i) => (
                <p key={i} className="news-item">· {title}</p>
              ))}
            </div>
          )}
        </div>
      )}
      {!hasLlm && <p className="no-llm">{t.factorOnly}</p>}
    </div>
  );
}

function SwingPanel({ lang, t }) {
  const [regime, setRegime]         = useState(null);
  const [scanData, setScanData]     = useState(null);
  const [loading, setLoading]       = useState(false);
  const [llmLoading, setLlmLoading] = useState(false);
  const [error, setError]           = useState(null);
  const [lastRun, setLastRun]       = useState(null);
  const [hasLlm, setHasLlm]         = useState(false);
  const [topN, setTopN]             = useState(5);

  useEffect(() => {
    axios.get(`${API}/regime`).then(r => setRegime(r.data)).catch(() => {});
  }, []);

  const runFactorScan = useCallback(async () => {
    setLoading(true);
    setError(null);
    setHasLlm(false);
    setScanData(null);
    try {
      const r = await axios.get(`${API}/scan?top_n=${topN}`);
      setScanData({ ...r.data, long_watchlist: null, short_watchlist: null });
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
      const r = await axios.post(`${API}/scan/full?top_n=${topN}&save=true&lang=${lang}`);
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
  }, [topN, t, lang, runFactorScan]);

  const longList  = scanData?.long_watchlist  || scanData?.long_candidates  || [];
  const shortList = scanData?.short_watchlist || scanData?.short_candidates || [];

  return (
    <div className="panel">
      {/* Controls */}
      <div className="controls">
        <div className="control-group">
          <label className="control-label">{t.topNLabel}</label>
          <select className="control-select" value={topN} onChange={e => setTopN(Number(e.target.value))}>
            {[3, 5, 10, 15].map(n => <option key={n} value={n}>{n}</option>)}
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
        {lastRun && <span className="last-run">{t.lastRun}: {lastRun}</span>}
      </div>

      {error && <div className="error-banner">{error}</div>}
      <RegimePanel regime={regime} t={t} />

      {scanData && (
        <div className="watchlist">
          <div className="watchlist-col">
            <div className="col-header long-header">
              <span className="col-title">{t.longTitle}</span>
              <span className="col-count">{longList.length}</span>
            </div>
            <div className="card-list">
              {longList.map(item => <SwingCard key={item.ticker} item={item} side="long" t={t} />)}
              {longList.length === 0 && <p className="empty">{t.noCandidates}</p>}
            </div>
          </div>
          <div className="watchlist-col">
            <div className="col-header short-header">
              <span className="col-title">{t.shortTitle}</span>
              <span className="col-count">{shortList.length}</span>
            </div>
            <div className="card-list">
              {shortList.map(item => <SwingCard key={item.ticker} item={item} side="short" t={t} />)}
              {shortList.length === 0 && <p className="empty">{t.noCandidates}</p>}
            </div>
          </div>
        </div>
      )}

      {!scanData && !loading && !llmLoading && (
        <div className="empty-state">
          <p className="empty-title">{t.noScanTitle}</p>
          <p className="empty-hint">{t.noSwingHint}</p>
        </div>
      )}

      {(loading || llmLoading) && !scanData && (
        <div className="loading-state">
          <div className="spinner" />
          <p>{llmLoading ? t.loadingLlm : t.loadingData}</p>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Day Trade (Premarket) components
// ─────────────────────────────────────────────────────────────

function PremarketCard({ item, t }) {
  const [expanded, setExpanded] = useState(false);
  const hasLlm  = item.signal !== undefined && item.signal !== null;
  const change  = item.premarket_change_pct ?? 0;
  const isUp    = change >= 0;

  return (
    <div
      className={`stock-card pm-card ${isUp ? "card-long" : "card-short"}`}
      onClick={() => setExpanded(!expanded)}
    >
      <div className="card-header">
        <div className="card-left">
          <span className="ticker">{item.ticker}</span>
          {hasLlm && <SignalBadge signal={item.signal} t={t} type="premarket" />}
          {item.catalyst_type && item.catalyst_type !== "UNKNOWN" && (
            <span className="catalyst-tag">{item.catalyst_type.replace("_", " ")}</span>
          )}
        </div>
        <div className="card-right">
          <span className={`pm-change ${isUp ? "pm-up" : "pm-down"}`}>
            {isUp ? "+" : ""}{change.toFixed(2)}%
          </span>
        </div>
      </div>

      <div className="pm-stats">
        <div className="pm-stat">
          <span className="stat-label">{t.rvol}</span>
          <span className="stat-value">{(item.rvol ?? 0).toFixed(1)}x</span>
        </div>
        <div className="pm-stat">
          <span className="stat-label">{t.volume}</span>
          <span className="stat-value">{((item.premarket_volume ?? 0) / 1000).toFixed(0)}K</span>
        </div>
        <div className="pm-stat">
          <span className="stat-label">{t.marketCap}</span>
          <span className="stat-value">${((item.market_cap ?? 0) / 1e6).toFixed(0)}M</span>
        </div>
        <div className="pm-stat">
          <span className="stat-label">{t.float}</span>
          <span className="stat-value">{((item.float ?? 0) / 1e6).toFixed(1)}M</span>
        </div>
      </div>

      {hasLlm && (
        <div className="card-body">
          <ConfidenceBar value={item.confidence || 0} />
          {item.reason && <p className="reason">{item.reason}</p>}
          {expanded && item.risk && (
            <p className="risk-flag">⚠ {item.risk}</p>
          )}
          {expanded && item.news?.length > 0 && (
            <div className="news-list">
              <p className="news-heading">{t.recentNews}</p>
              {item.news.map((n, i) => (
                <p key={i} className="news-item">· {n.headline}</p>
              ))}
            </div>
          )}
        </div>
      )}
      {!hasLlm && item.news?.length > 0 && expanded && (
        <div className="news-list">
          <p className="news-heading">{t.recentNews}</p>
          {item.news.map((n, i) => (
            <p key={i} className="news-item">· {n.headline}</p>
          ))}
        </div>
      )}
      {!hasLlm && <p className="no-llm">{t.dataOnly}</p>}
    </div>
  );
}

function PremarketPanel({ lang, t }) {
  const [pmData, setPmData]         = useState(null);
  const [loading, setLoading]       = useState(false);
  const [llmLoading, setLlmLoading] = useState(false);
  const [error, setError]           = useState(null);
  const [lastRun, setLastRun]       = useState(null);
  const [hasLlm, setHasLlm]         = useState(false);

  const runQuickScan = useCallback(async () => {
    setLoading(true);
    setError(null);
    setHasLlm(false);
    setPmData(null);
    try {
      const r = await axios.get(`${API}/premarket/scan`);
      setPmData(r.data);
      setLastRun(r.data.timestamp);
    } catch (e) {
      setError(e.response?.data?.detail || t.scanFailed);
    } finally {
      setLoading(false);
    }
  }, [t]);

  const runDeepScan = useCallback(async () => {
    setLlmLoading(true);
    setError(null);
    try {
      await runQuickScan();
      const r = await axios.post(`${API}/premarket/scan/full?lang=${lang}`);
      setPmData(r.data);
      setHasLlm(true);
      setLastRun(r.data.timestamp);
    } catch (e) {
      setError(e.response?.data?.detail || t.fullScanFailed);
    } finally {
      setLlmLoading(false);
      setLoading(false);
    }
  }, [t, lang, runQuickScan]);

  const candidates = pmData?.candidates || [];

  return (
    <div className="panel">
      {/* Controls */}
      <div className="controls">
        <button
          className={`btn ${!hasLlm ? "btn-active" : "btn-secondary"}`}
          onClick={runQuickScan}
          disabled={loading || llmLoading}
        >
          {loading ? t.scanning : t.quickScan}
        </button>
        <button
          className={`btn ${hasLlm ? "btn-active" : "btn-secondary"}`}
          onClick={runDeepScan}
          disabled={loading || llmLoading}
        >
          {llmLoading ? t.analyzing : t.deepScan}
        </button>
        {lastRun && <span className="last-run">{t.lastRun}: {lastRun}</span>}
      </div>

      {error && <div className="error-banner">{error}</div>}

      {/* Disclaimer */}
      <div className="pm-disclaimer">
        ⏰ {t.pmDisclaimer}
      </div>

      {/* Results */}
      {pmData && (
        <div className="pm-results">
          <div className="col-header pm-header">
            <span className="col-title">{t.pmTitle}</span>
            <span className="col-count">{candidates.length}</span>
          </div>
          <div className="card-list">
            {candidates.map(item => (
              <PremarketCard key={item.ticker} item={item} t={t} />
            ))}
            {candidates.length === 0 && (
              <div className="empty-state">
                <p className="empty-title">{t.noCandidates}</p>
                <p className="empty-hint">{t.noPmHint}</p>
              </div>
            )}
          </div>
        </div>
      )}

      {!pmData && !loading && !llmLoading && (
        <div className="empty-state">
          <p className="empty-title">{t.noScanTitle}</p>
          <p className="empty-hint">{t.noPmHint}</p>
        </div>
      )}

      {(loading || llmLoading) && !pmData && (
        <div className="loading-state">
          <div className="spinner" />
          <p>{llmLoading ? t.loadingLlm : t.loadingData}</p>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Main App
// ─────────────────────────────────────────────────────────────

export default function App() {
  const [lang, setLang]         = useState("en");
  const [activeTab, setActiveTab] = useState(TABS.SWING);
  const t = T[lang];

  return (
    <div className="app">
      <header className="header">
        <div className="header-left">
          <h1 className="title">
            <img src="/logo.png" alt="logo" className="logo" />
            {t.title}
          </h1>
          <p className="subtitle">{t.subtitle}</p>
        </div>
        <div className="header-right">
          <TabBar activeTab={activeTab} setActiveTab={setActiveTab} t={t} />
          <LangToggle lang={lang} setLang={setLang} />
        </div>
      </header>

      {activeTab === TABS.SWING && (
        <SwingPanel lang={lang} t={t} />
      )}

      {activeTab === TABS.PREMARKET && (
        <PremarketPanel lang={lang} t={t} />
      )}
    </div>
  );
}
