import { useMemo } from 'react';
import ReactFlow, { Background, Controls, MarkerType } from 'reactflow';
import 'reactflow/dist/style.css';
import type { CandidatePath } from './adapter';

interface GraphViewProps {
  paths: CandidatePath[];
  highlightedPathId: string | null;
  onPathClick: (pathId: string) => void;
  showRejected: boolean;
}



export default function GraphView({ paths, highlightedPathId, onPathClick, showRejected }: GraphViewProps) {
  const { nodes, edges } = useMemo(() => {
    const rfNodes: any[] = [];
    const rfEdges: any[] = [];
    const addedNodes = new Set<string>();

    let yOffset = 0;

    const visiblePaths = showRejected ? paths : paths.filter(p => p.selection_status === 'selected');

    visiblePaths.forEach((path) => {
      let xOffset = 0;
      
      const isHighlighted = highlightedPathId === path.path_id;
      const isRejected = path.selection_status === 'rejected';

      path.nodes.forEach((node) => {
        // Unique ID for node in React Flow to handle multiple paths containing same node
        // Actually, let's keep them unified if they share the same ID to form a real graph.
        const rfNodeId = node.id;
        
        if (!addedNodes.has(rfNodeId)) {
          addedNodes.add(rfNodeId);
          rfNodes.push({
            id: rfNodeId,
            data: { label: node.id },
            position: { x: xOffset, y: yOffset + Math.random() * 50 },
            style: {
              background: isHighlighted ? '#1e3a8a' : isRejected ? '#1f2937' : '#111827',
              borderColor: isHighlighted ? '#3b82f6' : isRejected ? '#4b5563' : '#374151',
              color: isRejected ? '#9ca3af' : '#f3f4f6',
              opacity: isRejected && !isHighlighted ? 0.6 : 1,
            }
          });
        }
        xOffset += 300;
      });

      path.edges.forEach((edge, edgeIdx) => {
        rfEdges.push({
          id: `${path.path_id}-${edgeIdx}`,
          source: edge.source_id,
          target: edge.target_id,
          label: edge.relation_type,
          animated: isHighlighted,
          style: {
            strokeWidth: isHighlighted ? 3 : 1,
            stroke: isHighlighted ? '#3b82f6' : isRejected ? '#4b5563' : '#9ca3af',
            opacity: isRejected && !isHighlighted ? 0.4 : 1,
          },
          labelStyle: { fill: '#9ca3af', fontWeight: 500, fontSize: 10 },
          labelBgStyle: { fill: '#111827', color: '#fff' },
          markerEnd: {
            type: MarkerType.ArrowClosed,
            color: isHighlighted ? '#3b82f6' : isRejected ? '#4b5563' : '#9ca3af',
          },
        });
      });

      yOffset += 100;
    });

    return { nodes: rfNodes, edges: rfEdges };
  }, [paths, highlightedPathId, showRejected]);

  return (
    <div style={{ width: '100%', height: '100%' }}>
      <ReactFlow 
        nodes={nodes} 
        edges={edges}
        onEdgeClick={(_, edge) => {
          const pathId = edge.id.split('-')[0];
          onPathClick(pathId);
        }}
        fitView
        attributionPosition="bottom-right"
      >
        <Background color="#374151" gap={16} />
        <Controls />
      </ReactFlow>
    </div>
  );
}
