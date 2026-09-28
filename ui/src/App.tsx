import { useState, useEffect } from 'react';
import { parseAuditTrace, type AuditTrace } from './adapter';
import rawAudit from './audit_new.json';
import GraphView from './GraphView';
import { ChevronDown, ChevronRight, Activity, Search, Shield, Zap } from 'lucide-react';

export default function App() {
  const [trace, setTrace] = useState<AuditTrace | null>(null);
  const [activeTab, setActiveTab] = useState<'graph' | 'details'>('graph');
  const [highlightedPathId, setHighlightedPathId] = useState<string | null>(null);
  const [highlightedEvidenceId, setHighlightedEvidenceId] = useState<string | null>(null);
  const [showRejectedPaths, setShowRejectedPaths] = useState(false);
  const [expandedRetrieval, setExpandedRetrieval] = useState(false);

  useEffect(() => {
    // Parse the raw JSON on mount
    try {
      const parsed = parseAuditTrace(rawAudit);
      setTrace(parsed);
    } catch (err) {
      console.error("Failed to parse audit trace:", err);
    }
  }, []);

  if (!trace) {
    return <div className="app-container">Loading Audit Trace...</div>;
  }

  const { query, retrieval, graph_expansion, evidence, answer } = trace;

  const handleCitationClick = (evidenceId: string) => {
    setHighlightedEvidenceId(evidenceId);
    const ev = evidence.find(e => e.id === evidenceId);
    if (ev) {
      setHighlightedPathId(ev.path_id);
    }
  };

  const handlePathClick = (pathId: string) => {
    setHighlightedPathId(pathId);
    const ev = evidence.find(e => e.path_id === pathId);
    if (ev) {
      setHighlightedEvidenceId(ev.id);
    }
  };

  // Convert answer text into parts with clickable citations
  const renderAnswerText = () => {
    const parts = answer.text.split(/(\[E\d+\])/g);
    return parts.map((part, i) => {
      if (part.match(/^\[E\d+\]$/)) {
        const evId = part.replace(/\[|\]/g, "");
        return (
          <span 
            key={i} 
            className="citation"
            onClick={() => handleCitationClick(evId)}
          >
            {part}
          </span>
        );
      }
      return <span key={i}>{part}</span>;
    });
  };

  return (
    <div className="app-container">
      
      {/* 1. QUESTION SECTION */}
      <section className="glass-panel header-section">
        <h1 className="query-title">"{query}"</h1>
        <div className="meta-info">
          <span className="meta-badge"><Search size={14} style={{marginRight: 4, verticalAlign: 'middle'}}/> Top K: {retrieval.parameters?.top_k || 'N/A'}</span>
          <span className="meta-badge"><Activity size={14} style={{marginRight: 4, verticalAlign: 'middle'}}/> Max Hops: {retrieval.parameters?.max_hops || 'N/A'}</span>
          <span className="meta-badge"><Shield size={14} style={{marginRight: 4, verticalAlign: 'middle'}}/> Min Confidence: {retrieval.parameters?.min_confidence_threshold || 0}</span>
          <span className="meta-badge"><Zap size={14} style={{marginRight: 4, verticalAlign: 'middle'}}/> Candidates: {retrieval.candidate_count}</span>
        </div>
      </section>

      <div className="pipeline-layout">
        
        {/* 2. RETRIEVAL VISUALIZATION (LEFT PANEL) */}
        <aside className="glass-panel">
          <h2 style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            Retrieval
            <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              {retrieval.selected.length} Selected
            </span>
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
                {expandedRetrieval ? "Hide Rejected Candidates" : `Show ${retrieval.rejected.length} Rejected Candidates`}
              </span>
            </div>

            {expandedRetrieval && retrieval.rejected.map((cand, idx) => (
              <div key={cand.gid} className="candidate-card rejected" title={cand.rejection_reason || ""}>
                <div className="candidate-header">
                  <strong>#{retrieval.selected.length + idx + 1} {cand.type}</strong>
                  <span className="candidate-score">{cand.score.toFixed(4)}</span>
                </div>
                <div className="candidate-preview" style={{ marginBottom: '0.5rem' }}>{cand.preview}</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--danger)' }}>
                  Reason: {cand.rejection_reason}
                </div>
              </div>
            ))}
          </div>
        </aside>

        {/* RIGHT PANEL: GRAPH EXPANSION & EVIDENCE */}
        <main className="main-view-area">
          
          {/* 3. & 4. PATH SELECTION GRAPH */}
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
                  onChange={(e) => setShowRejectedPaths(e.target.checked)}
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
            
            {/* Show details for highlighted path */}
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

          {/* 6. EVIDENCE LAYER */}
          <section className="glass-panel">
            <h2 style={{ marginBottom: '1rem' }}>Evidence Layer</h2>
            <div className="evidence-panel">
              {evidence.map((ev) => (
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

          {/* 7. FINAL ANSWER */}
          <section className="glass-panel answer-section" style={{ borderColor: 'var(--accent-primary)' }}>
            <h2 style={{ marginBottom: '1rem', color: 'var(--accent-primary)' }}>Final Answer</h2>
            <div>{renderAnswerText()}</div>
          </section>
          
        </main>
      </div>
    </div>
  );
}
