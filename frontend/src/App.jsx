import { useState, useEffect, useCallback } from "react";
import axios from "axios";
import "./App.css";

const API  = "https://llm-driven-stock-scanner-production.up.railway.app/api";
const TABS = { SWING: "swing", PREMARKET: "premarket" };

// ─────────────────────────────────────────────────────────────
// Default scanner params (matches Ross Pro strategy)
// ─────────────────────────────────────────────────────────────

const DEFAULT_PARAMS = {
  min_price:      1,
  max_price:      20,
  max_market_cap: 300,   // M$ (millions USD)
  max_float:      20000, // K shares (20000K = 20M shares, Ross standard)
  min_change:     4,
  max_change:     40,
  min_pm_volume:  200,   // K shares (200K = 200,000 shares)
  min_pm_amount:  1000,  // K$ (1000K$ = $1M)
  min_rvol:       2,
  min_day_change: 0,
  min_day_volume: 0,
  direction:      "both",
  sort_by:        "change",
};

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
    customScan:      "⚙ Custom Scan",
    scanning:        "Scanning…",
    analyzing:       "Analyzing…",
    longTitle:       "▲ Long Candidates",
    shortTitle:      "▼ Short Candidates",
    pmTitle:         "🌅 Premarket Movers",
    noCandidates:    "No candidates",
    noScanTitle:     "No scan data",
    noSwingHint:     'Click "Factor Scan" for a quick scan, or "Full Scan" for LLM analysis.',
    noPmHint:        "Best run between 4:00–9:30 AM ET. Adjust parameters and click Custom Scan.",
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
    resetDefaults:   "Reset",
    applyAndScan:    "▶ Run Custom Scan",
    // Custom scan labels
    cs_price:        "Price Range ($)",
    cs_mktcap:       "Market Cap (M$)",
    cs_float:        "Float",
    cs_change:       "Premarket Change (%)",
    cs_pmvol:        "Premarket Volume",
    cs_pmamt:        "Premarket Amount",
    cs_rvol:         "RVOL",
    cs_daychange:    "Day Change % (intraday)",
    cs_dayvol:       "Day Volume",
    cs_direction:    "Direction",
    cs_sortby:       "Sort By",
    cs_unlimited:    "No limit",
    dir_both:        "Both ↑↓",
    dir_up:          "Up ↑ only",
    dir_down:        "Down ↓ only",
    sort_change:     "By Change %",
    sort_rvol:       "By RVOL",
    sort_volume:     "By Volume",
    sort_amount:     "By Amount",
    regime: { TRENDING: "TRENDING", VOLATILE: "VOLATILE", NEUTRAL: "NEUTRAL" },
    signal: { STRONG_BUY: "STRONG BUY", BUY: "BUY", NEUTRAL: "NEUTRAL", SHORT: "SHORT", STRONG_SHORT: "STRONG SHORT", NO_POSITION: "NO POSITION", AVOID: "AVOID" },
    pmSignal: { TRADE: "TRADE", WATCH: "WATCH", AVOID: "AVOID" },
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
    customScan:      "⚙ 自定义扫描",
    scanning:        "扫描中…",
    analyzing:       "分析中…",
    longTitle:       "▲ 做多候选",
    shortTitle:      "▼ 做空候选",
    pmTitle:         "🌅 盘前异动",
    noCandidates:    "暂无候选",
    noScanTitle:     "暂无扫描数据",
    noSwingHint:     '点击"因子扫描"快速扫描，或"完整扫描"获取AI分析。',
    noPmHint:        "最佳运行时间：美东时间4:00–9:30。调整参数后点击自定义扫描。",
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
    resetDefaults:   "重置",
    applyAndScan:    "▶ 运行自定义扫描",
    cs_price:        "价格范围 ($)",
    cs_mktcap:       "总市值上限 (百万美元)",
    cs_float:        "流通股上限",
    cs_change:       "盘前涨跌幅 (%)",
    cs_pmvol:        "盘前成交量",
    cs_pmamt:        "盘前成交额",
    cs_rvol:         "量比 RVOL",
    cs_daychange:    "单日涨跌幅 % (盘中)",
    cs_dayvol:       "当日成交量",
    cs_direction:    "方向",
    cs_sortby:       "排序方式",
    cs_unlimited:    "不限",
    dir_both:        "涨跌都看",
    dir_up:          "只看涨 ↑",
    dir_down:        "只看跌 ↓",
    sort_change:     "按涨跌幅",
    sort_rvol:       "按量比",
    sort_volume:     "按成交量",
    sort_amount:     "按成交额",
    regime: { TRENDING: "趋势市", VOLATILE: "震荡市", NEUTRAL: "中性" },
    signal: { STRONG_BUY: "强烈买入", BUY: "买入", NEUTRAL: "中性", SHORT: "做空", STRONG_SHORT: "强烈做空", NO_POSITION: "不操作", AVOID: "回避" },
    pmSignal: { TRADE: "可交易", WATCH: "观察", AVOID: "回避" },
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
      >{t.tabSwing}</button>
      <button
        className={`tab-btn ${activeTab === TABS.PREMARKET ? "tab-active" : ""}`}
        onClick={() => setActiveTab(TABS.PREMARKET)}
      >{t.tabDayTrade}</button>
    </div>
  );
}

function ConfidenceBar({ value }) {
  const color = value >= 75 ? "#22c55e" : value >= 50 ? "#f59e0b" : "#ef4444";
  return (
    <div className="conf-bar-bg">
      <div className="conf-bar-fill" style={{ width: `${value}%`, background: color }} />
      <span className="conf-label">{value}%</span>
    </div>
  );
}

function RegimeBadge({ regime, t }) {
  const colors = { TRENDING: "badge-trending", VOLATILE: "badge-volatile", NEUTRAL: "badge-neutral" };
  return (
    <span className={`badge ${colors[regime] || "badge-neutral"}`}>
      {t.regime[regime] || regime}
    </span>
  );
}

function SignalBadge({ signal, t, type = "swing" }) {
  if (type === "premarket") {
    const colors = { TRADE: "signal-strong-buy", WATCH: "signal-neutral", AVOID: "signal-avoid" };
    return <span className={`signal ${colors[signal] || "signal-neutral"}`}>{t.pmSignal[signal] || signal || "—"}</span>;
  }
  const colors = {
    STRONG_BUY:   "signal-strong-buy",
    BUY:          "signal-buy",
    NEUTRAL:      "signal-neutral",
    SHORT:        "signal-short",
    STRONG_SHORT: "signal-strong-short",
    NO_POSITION:  "signal-no-position",
    AVOID:        "signal-avoid",
  };
  return <span className={`signal ${colors[signal] || "signal-neutral"}`}>{t.signal[signal] || signal || "—"}</span>;
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
          <span className="stat-value">{regime.realized_vol ? (regime.realized_vol * 100).toFixed(1) + "%" : "—"}</span>
        </div>
        <div className="stat">
          <span className="stat-label">{t.trend}</span>
          <span className="stat-value">{regime.trend_strength ? (regime.trend_strength * 100).toFixed(0) + "%" : "—"}</span>
        </div>
        <div className="stat">
          <span className="stat-label">{t.factors}</span>
          <span className="stat-value factors">{regime.recommended_factors?.join(", ")}</span>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Custom Scan Panel
// ─────────────────────────────────────────────────────────────

function ParamRow({ label, children }) {
  return (
    <div className="param-row">
      <span className="param-label">{label}</span>
      <div className="param-inputs">{children}</div>
    </div>
  );
}

function ParamInput({ value, onChange, min, max, step = 1, placeholder }) {
  return (
    <input
      className="param-input"
      type="number"
      value={value}
      onChange={e => onChange(e.target.value === "" ? "" : Number(e.target.value))}
      min={min}
      max={max}
      step={step}
      placeholder={placeholder}
    />
  );
}

function CustomScanPanel({ params, setParams, t }) {
  return (
    <div className="custom-scan-panel">
          {/* Group 1: Stock basics */}
          <div className="cs-group">
            <div className="cs-group-title">
              {t.cs_price === "Price Range ($)" ? "Stock Basics" : "股票基本条件"}
            </div>

            <ParamRow label={t.cs_price}>
              <ParamInput value={params.min_price} onChange={v => setParams(p => ({...p, min_price: v}))} min={0.1} max={500} step={0.5} />
              <span className="param-sep">—</span>
              <ParamInput value={params.max_price} onChange={v => setParams(p => ({...p, max_price: v}))} min={1} max={500} step={1} />
            </ParamRow>

            <ParamRow label={t.cs_mktcap}>
              <ParamInput value={params.max_market_cap} onChange={v => setParams(p => ({...p, max_market_cap: v}))} min={10} max={100000} step={50} />
              <span className="param-unit">M$</span>
            </ParamRow>

            <ParamRow label={t.cs_float}>
              <ParamInput value={params.max_float} onChange={v => setParams(p => ({...p, max_float: v}))} min={0} max={999999} step={100} placeholder={t.cs_unlimited} />
              <span className="param-unit">K sh</span>
            </ParamRow>
          </div>

          {/* Group 2: Premarket conditions */}
          <div className="cs-group">
            <div className="cs-group-title">
              {t.cs_change.includes("盘前") ? "盘前条件" : "Premarket Conditions"}
            </div>

            <ParamRow label={t.cs_change}>
              <ParamInput value={params.min_change} onChange={v => setParams(p => ({...p, min_change: v}))} min={0} max={100} step={1} />
              <span className="param-sep">—</span>
              <ParamInput value={params.max_change} onChange={v => setParams(p => ({...p, max_change: v}))} min={1} max={500} step={1} />
              <span className="param-unit">%</span>
            </ParamRow>

            <ParamRow label={t.cs_pmvol}>
              <ParamInput value={params.min_pm_volume} onChange={v => setParams(p => ({...p, min_pm_volume: v}))} min={0} max={999999} step={10} placeholder={t.cs_unlimited} />
              <span className="param-unit">K sh</span>
            </ParamRow>

            <ParamRow label={t.cs_pmamt}>
              <ParamInput value={params.min_pm_amount} onChange={v => setParams(p => ({...p, min_pm_amount: v}))} min={0} max={999999} step={10} placeholder={t.cs_unlimited} />
              <span className="param-unit">K$</span>
            </ParamRow>

            <ParamRow label={t.cs_rvol}>
              <ParamInput value={params.min_rvol} onChange={v => setParams(p => ({...p, min_rvol: v}))} min={0} max={20} step={0.5} placeholder={t.cs_unlimited} />
              <span className="param-unit">x</span>
            </ParamRow>
          </div>

          {/* Group 3: Intraday conditions */}
          <div className="cs-group">
            <div className="cs-group-title">
              {t.cs_daychange.includes("单日") ? "盘中条件（可选）" : "Intraday Conditions (optional)"}
            </div>

            <ParamRow label={t.cs_daychange}>
              <ParamInput value={params.min_day_change} onChange={v => setParams(p => ({...p, min_day_change: v}))} min={0} max={100} step={1} placeholder={t.cs_unlimited} />
              <span className="param-unit">%</span>
            </ParamRow>

            <ParamRow label={t.cs_dayvol}>
              <ParamInput value={params.min_day_volume} onChange={v => setParams(p => ({...p, min_day_volume: v}))} min={0} max={999999} step={10} placeholder={t.cs_unlimited} />
              <span className="param-unit">K sh</span>
            </ParamRow>
          </div>

          {/* Group 4: Sort and direction */}
          <div className="cs-group">
            <div className="cs-group-title">
              {t.cs_direction.includes("方向") ? "排序与方向" : "Sort & Direction"}
            </div>

            <ParamRow label={t.cs_direction}>
              <select
                className="param-select"
                value={params.direction}
                onChange={e => setParams(p => ({...p, direction: e.target.value}))}
              >
                <option value="both">{t.dir_both}</option>
                <option value="up">{t.dir_up}</option>
                <option value="down">{t.dir_down}</option>
              </select>
            </ParamRow>

            <ParamRow label={t.cs_sortby}>
              <select
                className="param-select"
                value={params.sort_by}
                onChange={e => setParams(p => ({...p, sort_by: e.target.value}))}
              >
                <option value="change">{t.sort_change}</option>
                <option value="rvol">{t.sort_rvol}</option>
                <option value="volume">{t.sort_volume}</option>
                <option value="amount">{t.sort_amount}</option>
              </select>
            </ParamRow>
          </div>

          {/* Reset button — bottom right of panel */}
          <div className="cs-reset-row">
            <button
              className="btn btn-secondary"
              onClick={() => setParams(DEFAULT_PARAMS)}
            >
              {t.resetDefaults}
            </button>
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
          <span className="score-value">{(item.score ?? item.factor_score ?? 0).toFixed(3)}</span>
        </div>
      </div>
      {hasLlm && (
        <div className="card-body">
          <ConfidenceBar value={item.confidence || 0} />
          {item.holding_period_days > 0 && (
            <span className="holding-badge">Hold {item.holding_period_days}d</span>
          )}
          {item.reason && <p className="reason">{item.reason}</p>}
          {expanded && item.risk_flag && item.risk_flag !== "none" && (
            <p className="risk-flag">⚠ {item.risk_flag}</p>
          )}
          {expanded && item.news_titles?.length > 0 && (
            <div className="news-list">
              <p className="news-heading">{t.recentNews}</p>
              {item.news_titles.map((title, i) => <p key={i} className="news-item">· {title}</p>)}
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
      <div className="controls">
        <div className="control-group">
          <label className="control-label">{t.topNLabel}</label>
          <select className="control-select" value={topN} onChange={e => setTopN(Number(e.target.value))}>
            {[3, 5, 10, 15].map(n => <option key={n} value={n}>{n}</option>)}
          </select>
        </div>
        <button className={`btn ${!hasLlm ? "btn-active" : "btn-secondary"}`} onClick={runFactorScan} disabled={loading || llmLoading}>
          {loading ? t.scanning : t.factorScan}
        </button>
        <button className={`btn ${hasLlm ? "btn-active" : "btn-secondary"}`} onClick={runFullScan} disabled={loading || llmLoading}>
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
  const hasLlm = item.signal !== undefined && item.signal !== null;
  const change = item.premarket_change_pct ?? 0;
  const isUp   = change >= 0;

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
            <span className="catalyst-tag">{item.catalyst_type.replace(/_/g, " ")}</span>
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
          <span className="stat-value">{((item.float ?? 0) / 1000).toFixed(0)}K</span>
        </div>
        {item.pm_amount > 0 && (
          <div className="pm-stat">
            <span className="stat-label">{lang === "zh" ? "成交额" : "Amount"}</span>
            <span className="stat-value">${((item.pm_amount ?? 0) / 1000).toFixed(0)}K</span>
          </div>
        )}
      </div>

      {hasLlm && (
        <div className="card-body">
          <ConfidenceBar value={item.confidence || 0} />
          {item.holding_period_days > 0 && (
            <span className="holding-badge">Hold {item.holding_period_days}d</span>
          )}
          {item.reason && <p className="reason">{item.reason}</p>}
          {expanded && item.risk && <p className="risk-flag">⚠ {item.risk}</p>}
          {expanded && item.news?.length > 0 && (
            <div className="news-list">
              <p className="news-heading">{t.recentNews}</p>
              {item.news.map((n, i) => <p key={i} className="news-item">· {n.headline}</p>)}
            </div>
          )}
        </div>
      )}
      {!hasLlm && expanded && item.news?.length > 0 && (
        <div className="news-list">
          <p className="news-heading">{t.recentNews}</p>
          {item.news.map((n, i) => <p key={i} className="news-item">· {n.headline}</p>)}
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
  const [params, setParams]         = useState(DEFAULT_PARAMS);

  // Build query string from current params
  const buildQueryString = (p) => [
    `min_price=${p.min_price}`,
    `max_price=${p.max_price}`,
    `max_market_cap=${p.max_market_cap * 1e6}`,
    p.max_float > 0 ? `max_float=${p.max_float * 1000}` : `max_float=1e12`,
    `min_change=${p.min_change}`,
    `max_change=${p.max_change}`,
    p.min_pm_volume > 0 ? `min_pm_volume=${p.min_pm_volume * 1000}` : `min_pm_volume=0`,
    p.min_pm_amount > 0 ? `min_pm_amount=${p.min_pm_amount * 1000}` : `min_pm_amount=0`,
    `min_rvol=${p.min_rvol}`,
    `min_day_change=${p.min_day_change}`,
    p.min_day_volume > 0 ? `min_day_volume=${p.min_day_volume * 1000}` : `min_day_volume=0`,
    `direction=${p.direction}`,
    `sort_by=${p.sort_by}`,
  ].join("&");

  // Validation
  const validate = (p) => {
    if (p.min_price <= 0)            return t.err_min_price || "Min price must be > 0";
    if (p.max_price < p.min_price)   return t.err_max_price || "Max price must be ≥ min price";
    if (p.max_price > 500)           return t.err_price_cap || "Max price cannot exceed $500";
    if (p.max_market_cap <= 0)       return t.err_mktcap   || "Market cap must be > 0";
    if (p.min_change < 0)            return t.err_change   || "Min change cannot be negative";
    if (p.max_change < p.min_change) return t.err_max_change || "Max change must be ≥ min change";
    if (p.min_rvol < 0)              return t.err_rvol     || "RVOL cannot be negative";
    return null;
  };

  // Custom scan using current params
  const runCustomScan = useCallback(async () => {
    const err = validate(params);
    if (err) { setError(err); return; }
    setLoading(true);
    setError(null);
    setHasLlm(false);
    setPmData(null);
    try {
      const r = await axios.get(`${API}/premarket/scan?${buildQueryString(params)}`);
      setPmData(r.data);
      setLastRun(r.data.timestamp);
    } catch (e) {
      setError(e.response?.data?.detail || t.scanFailed);
    } finally {
      setLoading(false);
    }
  }, [params, t]);

  // Quick scan uses default params
  const runQuickScan = useCallback(async (queryStr = null) => {
    setLoading(true);
    setError(null);
    setHasLlm(false);
    setPmData(null);
    try {
      const qs = queryStr || buildDefaultQuery();
      const r  = await axios.get(`${API}/premarket/scan?${qs}`);
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
      // First show quick results
      setLoading(true);
      setPmData(null);
      const r1 = await axios.get(`${API}/premarket/scan?${buildQueryString(params)}`);
      setPmData(r1.data);
      setLoading(false);
      // Then LLM analysis
      const r  = await axios.post(`${API}/premarket/scan/full?${buildQueryString(params)}&lang=${lang}`);
      setPmData(r.data);
      setHasLlm(true);
      setLastRun(r.data.timestamp);
    } catch (e) {
      setError(e.response?.data?.detail || t.fullScanFailed);
    } finally {
      setLlmLoading(false);
      setLoading(false);
    }
  }, [t, lang, params, buildQueryString]);

  // Build default query string from DEFAULT_PARAMS
  function buildDefaultQuery() {
    const p = DEFAULT_PARAMS;
    return [
      `min_price=${p.min_price}`,
      `max_price=${p.max_price}`,
      `max_market_cap=${p.max_market_cap * 1e6}`,
      `max_float=${p.max_float * 1000}`,
      `min_change=${p.min_change}`,
      `max_change=${p.max_change}`,
      `min_pm_volume=${p.min_pm_volume * 1000}`,
      `min_pm_amount=${p.min_pm_amount * 1000}`,
      `min_rvol=${p.min_rvol}`,
      `direction=${p.direction}`,
      `sort_by=${p.sort_by}`,
    ].join("&");
  }

  const candidates = pmData?.candidates || [];

  return (
    <div className="panel">
      {/* Buttons row: Custom Scan | Deep Scan | last run time */}
      <div className="controls">
        <button
          className={`btn ${!hasLlm && !loading && !llmLoading ? "btn-active" : "btn-secondary"}`}
          onClick={runCustomScan}
          disabled={loading || llmLoading}
        >
          {loading ? t.scanning : t.customScan}
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

      {/* Params panel — always visible, Reset button inside at bottom right */}
      <CustomScanPanel
        params={params}
        setParams={setParams}
        t={t}
      />

      {error && <div className="error-banner">{error}</div>}

      <div className="pm-disclaimer">⏰ {t.pmDisclaimer}</div>

      {pmData && (
        <div className="pm-results">
          <div className="col-header pm-header">
            <span className="col-title">{t.pmTitle}</span>
            <span className="col-count">{candidates.length}</span>
          </div>
          <div className="card-list">
            {candidates.map(item => <PremarketCard key={item.ticker} item={item} t={t} />)}
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
  const [lang, setLang]           = useState("en");
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

      {activeTab === TABS.SWING    && <SwingPanel    lang={lang} t={t} />}
      {activeTab === TABS.PREMARKET && <PremarketPanel lang={lang} t={t} />}
    </div>
  );
}
