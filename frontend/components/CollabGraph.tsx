'use client';

import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import type { StudentState } from '@/lib/types';

interface Node extends d3.SimulationNodeDatum {
  id: string;
  label: StudentState['label'];
  collabLabel: StudentState['collabLabel'];
  prob: number;
}
interface Link extends d3.SimulationLinkDatum<Node> {
  strength: number;
  collab: boolean;
}

interface CollabGraphProps {
  students: StudentState[];
}

function buildGraph(students: StudentState[]): { nodes: Node[]; links: Link[] } {
  const nodes: Node[] = students.map((s) => ({
    id: s.id,
    label: s.label,
    collabLabel: s.collabLabel,
    prob: s.engagementProb,
  }));

  const links: Link[] = [];
  for (let i = 0; i < students.length; i++) {
    for (let j = i + 1; j < students.length; j++) {
      const si = students[i];
      const sj = students[j];
      const areNeighbors = Math.abs(si.row - sj.row) <= 1 && Math.abs(si.col - sj.col) <= 1;
      if (!areNeighbors) continue;
      const bothCollab = si.collabLabel === 'Collaborative' && sj.collabLabel === 'Collaborative';
      links.push({
        source: si.id,
        target: sj.id,
        strength: bothCollab ? 0.8 + Math.random() * 0.2 : 0.1 + Math.random() * 0.3,
        collab: bothCollab,
      });
    }
  }
  return { nodes, links };
}

export function CollabGraph({ students }: CollabGraphProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const W = svgRef.current!.clientWidth || 300;
    const H = svgRef.current!.clientHeight || 220;
    const { nodes, links } = buildGraph(students);

    // Simulation
    const sim = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id((d: any) => d.id).distance(60))
      .force('charge', d3.forceManyBody().strength(-80))
      .force('center', d3.forceCenter(W / 2, H / 2))
      .force('collision', d3.forceCollide(28));

    // Arrow marker
    svg.append('defs').append('marker')
      .attr('id', 'arrow')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 20)
      .attr('refY', 0)
      .attr('markerWidth', 4)
      .attr('markerHeight', 4)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', 'rgba(56,189,248,0.4)');

    // Links
    const linkEl = svg.selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke', (d: Link) => d.collab ? 'rgba(56,189,248,0.5)' : 'rgba(148,163,184,0.15)')
      .attr('stroke-width', (d: Link) => d.strength * 3)
      .attr('stroke-dasharray', (d: Link) => d.collab ? 'none' : '3,3');

    // Node groups
    const nodeG = svg.selectAll('g')
      .data(nodes)
      .join('g')
      .attr('cursor', 'pointer')
      .call(
        d3.drag<any, Node>()
          .on('start', (event, d) => { if (!event.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
          .on('drag', (event, d) => { d.fx = event.x; d.fy = event.y; })
          .on('end', (event, d) => { if (!event.active) sim.alphaTarget(0); d.fx = null; d.fy = null; })
      );

    // Node circles
    nodeG.append('circle')
      .attr('r', 18)
      .attr('fill', (d: Node) => {
        if (d.label === 'Engaged') return 'rgba(34,211,166,0.12)';
        if (d.label === 'Not Engaged') return 'rgba(255,78,78,0.12)';
        return 'rgba(148,163,184,0.08)';
      })
      .attr('stroke', (d: Node) => {
        if (d.label === 'Engaged') return '#22D3A6';
        if (d.label === 'Not Engaged') return '#FF4E4E';
        return '#64748B';
      })
      .attr('stroke-width', 1.5);

    // Collab ring
    nodeG.filter((d: Node) => d.collabLabel === 'Collaborative')
      .append('circle')
      .attr('r', 22)
      .attr('fill', 'none')
      .attr('stroke', 'rgba(56,189,248,0.35)')
      .attr('stroke-width', 1)
      .attr('stroke-dasharray', '2,2');

    // Node labels
    nodeG.append('text')
      .text((d: Node) => d.id.split('-')[1])
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .attr('fill', (d: Node) => d.label === 'Engaged' ? '#22D3A6' : d.label === 'Not Engaged' ? '#FF4E4E' : '#94A3B8')
      .attr('font-size', 9)
      .attr('font-weight', 700)
      .attr('font-family', 'Inter, sans-serif');

    // Strength labels on links
    svg.selectAll('text.link-label')
      .data(links.filter((l: Link) => l.collab))
      .join('text')
      .attr('class', 'link-label')
      .attr('text-anchor', 'middle')
      .attr('fill', 'rgba(56,189,248,0.6)')
      .attr('font-size', 8)
      .attr('font-family', 'Inter, sans-serif');

    // Tick
    sim.on('tick', () => {
      linkEl
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      nodeG.attr('transform', (d: Node) => `translate(${d.x ?? 0},${d.y ?? 0})`);
    });

    return () => { sim.stop(); };
  }, [students]);

  return (
    <div className="w-full">
      <svg ref={svgRef} className="w-full h-56" />
      <div className="flex items-center justify-center gap-6 text-[10px] text-text-muted mt-2">
        <div className="flex items-center gap-1.5">
          <div className="w-5 h-0.5 bg-status-collab rounded" />
          Collaborative pair
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-5 h-0.5 bg-white/15 rounded border-dashed" style={{ borderBottom: '1px dashed #94A3B8' }} />
          Weak interaction
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-2 h-2 rounded-full border border-status-collab opacity-50" />
          Collaborative student
        </div>
      </div>
    </div>
  );
}
