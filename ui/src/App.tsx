import { useState, useEffect, useRef } from 'react';
import { parseAuditTrace, type AuditTrace } from './adapter';
import GraphView from './GraphView';
import {
  ChevronDown, ChevronRight, Activity, Search, Shield, Zap,
  Send, Settings2, Terminal, Loader2, CheckCircle2, XCircle,
  Brain, Sliders,
} from 'lucide-react';

// ──────────────────────────────────────────────────────────────────────────────
//  Types
// ──────────────────────────────────────────────────────────────────────────────
interface QueryParams {
  question: string;
  top_k: number;
  max_hops: number;
  neo4j_url: string;
  neo4j_username: string;
  neo4j_password: string;
  audit: boolean;
}

type RunStatus = 'idle' | 'running' | 'success' | 'error';

// ──────────────────────────────────────────────────────────────────────────────
//  Main Component
// ──────────────────────────────────────────────────────────────────────────────
export default function App() {
  // ── Query panel state ──
  const [params, setParams] = useState<QueryParams>({
    question: '',
    top_k: 3,
    max_hops: 2,
    neo4j_url: 'bolt://localhost:7687',
    neo4j_username: 'neo4j',
    neo4j_password: 'test1234',
    audit: true,
  });
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [runStatus, setRunStatus] = useState<RunStatus>('idle');
  const [logs, setLogs] = useState<string[]>([]);
  const [errorMsg, setErrorMsg] = useState('');
  const logEndRef = useRef<HTMLDivElement>(null);

  // ── Visualization state ──
  const [trace, setTrace] = useState<AuditTrace | null>(null);
  const [activeTab, setActiveTab] = useState<'graph' | 'details'>('graph');
  const [highlightedPathId, setHighlightedPathId] = useState<string | null>(null);
  const [highlightedEvidenceId, setHighlightedEvidenceId] = useState<string | null>(null);
  const [showRejectedPaths, setShowRejectedPaths] = useState(false);
  const [expandedRetrieval, setExpandedRetrieval] = useState(false);

  // Auto-scroll logs
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  // ── Run query ──
  const handleRun = async () => {
    if (!params.question.trim()) return;
    setRunStatus('running');
    setLogs(['⏳ Connecting to graph…']);
    setErrorMsg('');
    setTrace(null);

    // Animated progress ticks while waiting (real output arrives at the end)
    const progressMsgs = [
      '🔍 Querying Neo4j graph…',
      '🧠 Selecting top-K seed nodes…',
      '🕸️  Expanding neighborhood hops…',
      '📋 Collecting evidence…',
      '✍️  Generating answer with LLM…',
    ];
    let tick = 0;
    const ticker = setInterval(() => {
      if (tick < progressMsgs.length) {
        setLogs(prev => [...prev, progressMsgs[tick]]);
        tick++;
      }
    }, 6000);

    try {
      const res = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      });

      clearInterval(ticker);

      // Always parse body first — error responses also contain JSON with details
      let body: any;
      const rawText = await res.text();
      try {
        body = JSON.parse(rawText);
      } catch {
        throw new Error(`Server returned non-JSON response (HTTP ${res.status}): ${rawText.slice(0, 200)}`);
      }

      if (!res.ok) {
        // Show stderr / stdout from the script in the log panel for debugging
        const logLines: string[] = [];
        if (body.stdout) logLines.push(...body.stdout.split('\n').filter(Boolean));
        if (body.stderr) logLines.push(...body.stderr.split('\n').filter(Boolean));
        if (logLines.length) setLogs(logLines);
        throw new Error(body.error || `HTTP ${res.status}`);
      }

      // Show the script stdout in the log panel
      if (body.stdout) {
        const lines = body.stdout.split('\n').filter(Boolean);
        setLogs(lines);
      }

      setRunStatus('success');

      if (body.audit) {
        try {
          setTrace(parseAuditTrace(body.audit));
        } catch (e) {
          console.error('Failed to parse audit trace:', e);
          setLogs(prev => [...prev, '⚠️  Result received but visualization parse failed — check console.']);
        }
      }
    } catch (err: any) {
      clearInterval(ticker);
      setRunStatus('error');
      setErrorMsg(err.message ?? 'Unknown error');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      handleRun();
    }
  };

  // ── Citation helpers ──
  const handleCitationClick = (evidenceId: string) => {
    setHighlightedEvidenceId(evidenceId);
    const ev = trace?.evidence.find(e => e.id === evidenceId);
    if (ev) setHighlightedPathId(ev.path_id);
  };

  const handlePathClick = (pathId: string) => {
    setHighlightedPathId(pathId);
    const ev = trace?.evidence.find(e => e.path_id === pathId);
    if (ev) setHighlightedEvidenceId(ev.id);
  };

  const renderAnswerText = () => {
    if (!trace) return null;
    const parts = trace.answer.text.split(/(\[E\d+\])/g);
    return parts.map((part, i) => {
      if (part.match(/^\[E\d+\]$/)) {
        const evId = part.replace(/\[|\]/g, '');
        return (
          <span key={i} className="citation" onClick={() => handleCitationClick(evId)}>
            {part}
          </span>
        );
      }
      return <span key={i}>{part}</span>;
    });
  };

  // ── Render ──
  return (
    <div className="app-container">

      {/* ════════════════════════════════════════════
          QUERY INPUT PANEL
      ════════════════════════════════════════════ */}
      <section className="glass-panel query-panel">
        <div className="query-panel-header">
          <Brain size={22} className="query-icon" />
          <h1 className="query-panel-title">GraphRAG Medical QA</h1>
        </div>

        {/* Question textarea */}
        <div className="question-input-wrap">
          <textarea
            id="question-input"
            className="question-textarea"
            rows={3}
            placeholder="Ask a clinical question… e.g. What are the risk factors and treatment options for pneumonia?"
            value={params.question}
            onChange={e => setParams(p => ({ ...p, question: e.target.value }))}
            onKeyDown={handleKeyDown}
            disabled={runStatus === 'running'}
          />
          <button
            id="run-query-btn"
            className={`run-btn ${runStatus === 'running' ? 'running' : ''}`}
            onClick={handleRun}
            disabled={runStatus === 'running' || !params.question.trim()}
            title="Run query (Ctrl+Enter)"
          >
            {runStatus === 'running'
              ? <Loader2 size={20} className="spin" />
              : <Send size={20} />}
            <span>{runStatus === 'running' ? 'Running…' : 'Run'}</span>
          </button>
        </div>

        {/* Quick param badges */}
        <div className="quick-params">
          <label className="param-inline">
            <Search size={13} />
            Top-K
            <input
              type="number" min={1} max={20}
              value={params.top_k}
              onChange={e => setParams(p => ({ ...p, top_k: +e.target.value }))}
              className="param-number-input"
            />
          </label>
          <label className="param-inline">
            <Activity size={13} />
            Max Hops
            <input
              type="number" min={1} max={4}
              value={params.max_hops}
              onChange={e => setParams(p => ({ ...p, max_hops: +e.target.value }))}
              className="param-number-input"
            />
          </label>
          <label className="param-inline audit-toggle">
            <Shield size={13} />
            Audit Mode
            <input
              type="checkbox"
              checked={params.audit}
              onChange={e => setParams(p => ({ ...p, audit: e.target.checked }))}
            />
          </label>
          <button
            id="toggle-advanced-btn"
            className="advanced-btn"
            onClick={() => setShowAdvanced(v => !v)}
          >
            <Sliders size={13} />
            {showAdvanced ? 'Hide' : 'Advanced'}
          </button>
        </div>

        {/* Advanced settings */}
        {showAdvanced && (
          <div className="advanced-settings">
            <div className="adv-field">
              <label>Neo4j URL</label>
              <input
                type="text" value={params.neo4j_url}
                onChange={e => setParams(p => ({ ...p, neo4j_url: e.target.value }))}
                className="adv-input"
              />
            </div>
            <div className="adv-field">
              <label>Username</label>
              <input
                type="text" value={params.neo4j_username}
                onChange={e => setParams(p => ({ ...p, neo4j_username: e.target.value }))}
                className="adv-input"
              />
            </div>
            <div className="adv-field">
              <label>Password</label>
              <input
                type="password" value={params.neo4j_password}
                onChange={e => setParams(p => ({ ...p, neo4j_password: e.target.value }))}
                className="adv-input"
              />
            </div>
          </div>
        )}
      </section>

      {/* ════════════════════════════════════════════
          LIVE LOG / STATUS PANEL
      ════════════════════════════════════════════ */}
      {(runStatus !== 'idle' || logs.length > 0) && (
        <section className="glass-panel log-panel">
          <div className="log-header">
            <Terminal size={16} />
            <span>Live Output</span>
            {runStatus === 'running' && <Loader2 size={14} className="spin ml-auto" />}
            {runStatus === 'success' && <CheckCircle2 size={14} className="ml-auto status-ok" />}
            {runStatus === 'error' && <XCircle size={14} className="ml-auto status-err" />}
          </div>
          <div className="log-body">
            {logs.map((line, i) => (
              <div key={i} className="log-line">{line}</div>
            ))}
            {runStatus === 'error' && errorMsg && (
              <div className="log-line error-line">❌ {errorMsg}</div>
            )}
            <div ref={logEndRef} />
          </div>
        </section>
      )}

      {/* ════════════════════════════════════════════
          RESULTS (only shown after a successful run)
      ════════════════════════════════════════════ */}
      {trace && (() => {
        const { query, retrieval, graph_expansion, evidence, answer } = trace;
        return (
          <>
            {/* 1. QUESTION HEADER */}
            <section className="glass-panel header-section">
              <h2 className="query-title">"{query}"</h2>
              <div className="meta-info">
                <span className="meta-badge"><Search size={14} style={{ marginRight: 4, verticalAlign: 'middle' }} /> Top K: {retrieval.parameters?.top_k || params.top_k}</span>
                <span className="meta-badge"><Activity size={14} style={{ marginRight: 4, verticalAlign: 'middle' }} /> Max Hops: {retrieval.parameters?.max_hops || params.max_hops}</span>
                <span className="meta-badge"><Shield size={14} style={{ marginRight: 4, verticalAlign: 'middle' }} /> Min Confidence: {retrieval.parameters?.min_confidence_threshold || 0}</span>
                <span className="meta-badge"><Zap size={14} style={{ marginRight: 4, verticalAlign: 'middle' }} /> Candidates: {retrieval.candidate_count}</span>
              </div>
            </section>

            <div className="pipeline-layout">

              {/* 2. RETRIEVAL PANEL */}
              <aside className="glass-panel">
                <h2 style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  Retrieval
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{retrieval.selected.length} Selected</span>
                </h2>

                <div className="retrieval-list">
                  {retrieval.selected.map((cand, idx) => (
                    <div key={cand.gid} className="candidate-card selected">
                      <div className="candidate-header">
                        <strong>#{idx + 1} {cand.type}</strong>
                        <span className="candidate-score">{cand.score.toFixed(4)}</span>
                      </div>
                      <div className="candidate-preview">{cand.preview}</div>
                    </div>
                  ))}

                  <div
                    style={{ padding: '0.5rem', textAlign: 'center', cursor: 'pointer', color: 'var(--text-secondary)' }}
                    onClick={() => setExpandedRetrieval(!expandedRetrieval)}
                  >
                    {expandedRetrieval ? <ChevronDown size={20} /> : <ChevronRight size={20} />}
                    <span style={{ marginLeft: '0.5rem', verticalAlign: 'top' }}>
                      {expandedRetrieval ? 'Hide Rejected Candidates' : `Show ${retrieval.rejected.length} Rejected Candidates`}
                    </span>
                  </div>

                  {expandedRetrieval && retrieval.rejected.map((cand, idx) => (
                    <div key={cand.gid} className="candidate-card rejected" title={cand.rejection_reason || ''}>
                      <div className="candidate-header">
                        <strong>#{retrieval.selected.length + idx + 1} {cand.type}</strong>
                        <span className="candidate-score">{cand.score.toFixed(4)}</span>
                      </div>
                      <div className="candidate-preview" style={{ marginBottom: '0.5rem' }}>{cand.preview}</div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--danger)' }}>Reason: {cand.rejection_reason}</div>
                    </div>
                  ))}
                </div>
              </aside>

              {/* RIGHT PANEL */}
              <main className="main-view-area">

                {/* 3. GRAPH PATHS */}
                <section className="glass-panel" style={{ padding: '1rem' }}>
                  <div className="tabs">
                    <button
                      className={`tab ${activeTab === 'graph' ? 'active' : ''}`}
                      onClick={() => setActiveTab('graph')}
                    >
                      Graph Paths ({graph_expansion.selected_paths.length})
                    </button>
                    <label style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-secondary)', fontSize: '0.9rem', cursor: 'pointer' }}>
                      <input
                        type="checkbox"
                        checked={showRejectedPaths}
                        onChange={e => setShowRejectedPaths(e.target.checked)}
                      />
                      Show Rejected Paths ({graph_expansion.rejected_paths.length})
                    </label>
                  </div>

                  <div className="graph-container">
                    {activeTab === 'graph' && (
                      <GraphView
                        paths={graph_expansion.candidate_paths}
                        highlightedPathId={highlightedPathId}
                        onPathClick={handlePathClick}
                        showRejected={showRejectedPaths}
                      />
                    )}
                  </div>

                  {highlightedPathId && (
                    <div style={{ marginTop: '1rem', padding: '1rem', background: 'rgba(59, 130, 246, 0.1)', borderRadius: '8px' }}>
                      <h4 style={{ color: 'var(--accent-primary)', marginBottom: '0.5rem' }}>Path Details: {highlightedPathId}</h4>
                      {graph_expansion.candidate_paths.find(p => p.path_id === highlightedPathId)?.rejection_reason ? (
                        <div style={{ color: 'var(--danger)', fontSize: '0.9rem', marginBottom: '0.5rem' }}>
                          Rejected: {graph_expansion.candidate_paths.find(p => p.path_id === highlightedPathId)?.rejection_reason}
                        </div>
                      ) : null}
                      <div style={{ fontSize: '0.9rem', color: 'var(--text-primary)' }}>
                        {graph_expansion.candidate_paths.find(p => p.path_id === highlightedPathId)?.readable_path}
                      </div>
                    </div>
                  )}
                </section>

                {/* 4. EVIDENCE */}
                <section className="glass-panel">
                  <h2 style={{ marginBottom: '1rem' }}>Evidence Layer</h2>
                  <div className="evidence-panel">
                    {evidence.map(ev => (
                      <div
                        key={ev.id}
                        className={`evidence-item ${highlightedEvidenceId === ev.id ? 'highlighted' : ''}`}
                        onClick={() => handleCitationClick(ev.id)}
                        style={{ cursor: 'pointer' }}
                      >
                        <div className="evidence-id">[{ev.id}]</div>
                        <div style={{ flex: 1 }}>{ev.content}</div>
                        <div style={{ fontSize: '0.85rem', color: 'var(--accent-primary)', fontFamily: 'monospace' }}>
                          {(ev.confidence * 100).toFixed(1)}%
                        </div>
                      </div>
                    ))}
                    {evidence.length === 0 && (
                      <div style={{ color: 'var(--text-secondary)' }}>No evidence selected.</div>
                    )}
                  </div>
                </section>

                {/* 5. FINAL ANSWER */}
                <section className="glass-panel answer-section" style={{ borderColor: 'var(--accent-primary)' }}>
                  <h2 style={{ marginBottom: '1rem', color: 'var(--accent-primary)' }}>Final Answer</h2>
                  <div>{renderAnswerText()}</div>
                </section>

              </main>
            </div>
          </>
        );
      })()}

      {/* Empty state hint */}
      {runStatus === 'idle' && !trace && (
        <div className="empty-state">
          <Brain size={48} className="empty-icon" />
          <p>Type a clinical question above and press <kbd>Run</kbd> or <kbd>Ctrl+Enter</kbd></p>
        </div>
      )}
    </div>
  );
}
