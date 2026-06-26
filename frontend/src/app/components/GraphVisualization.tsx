'use client'

import React, { useEffect, useRef, useState, useCallback } from 'react'
import { ZoomIn, ZoomOut, Maximize2, RefreshCw } from 'lucide-react'

interface GraphNode {
  id: string
  label: string
  type: string
  x?: number
  y?: number
}

interface GraphEdge {
  source: string
  target: string
  label: string
  weight: number
}

interface GraphVisualizationProps {
  nodes: GraphNode[]
  edges: GraphEdge[]
  onNodeClick?: (nodeId: string) => void
}

const NODE_COLORS: Record<string, string> = {
  Person: '#10b981',
  System: '#3b82f6',
  Asset: '#8b5cf6',
  Process: '#f59e0b',
  Concept: '#ec4899',
  Document: '#6366f1',
  Unknown: '#64748b',
}

const NODE_RADII: Record<string, number> = {
  Person: 20,
  System: 28,
  Asset: 24,
  Process: 22,
  Concept: 18,
  Document: 16,
  Unknown: 18,
}

export default function GraphVisualization({
  nodes,
  edges,
  onNodeClick
}: GraphVisualizationProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [transform, setTransform] = useState({ x: 0, y: 0, scale: 1 })
  const [isDragging, setIsDragging] = useState(false)
  const [draggedNode, setDraggedNode] = useState<string | null>(null)
  const [nodePositions, setNodePositions] = useState<Map<string, { x: number; y: number }>>(new Map())
  const [initialized, setInitialized] = useState(false)

  // Initialize node positions with force-directed layout simulation
  const initializePositions = useCallback(() => {
    if (nodes.length === 0 || !containerRef.current) return

    const container = containerRef.current
    const width = container.clientWidth
    const height = container.clientHeight
    const centerX = width / 2
    const centerY = height / 2

    const positions = new Map<string, { x: number; y: number }>()
    
    // Place nodes in a circle initially
    const angleStep = (2 * Math.PI) / nodes.length
    const radius = Math.min(width, height) * 0.3

    nodes.forEach((node, index) => {
      const angle = angleStep * index - Math.PI / 2
      positions.set(node.id, {
        x: centerX + radius * Math.cos(angle),
        y: centerY + radius * Math.sin(angle)
      })
    })

    // Run force simulation iterations
    for (let iteration = 0; iteration < 50; iteration++) {
      const forces = new Map<string, { fx: number; fy: number }>()
      
      // Initialize forces
      nodes.forEach(node => {
        forces.set(node.id, { fx: 0, fy: 0 })
      })

      // Repulsion between all nodes
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const pos1 = positions.get(nodes[i].id)!
          const pos2 = positions.get(nodes[j].id)!
          
          const dx = pos2.x - pos1.x
          const dy = pos2.y - pos1.y
          const dist = Math.sqrt(dx * dx + dy * dy) || 1
          
          const force = 1000 / (dist * dist)
          const fx = (dx / dist) * force
          const fy = (dy / dist) * force
          
          const f1 = forces.get(nodes[i].id)!
          const f2 = forces.get(nodes[j].id)!
          f1.fx -= fx
          f1.fy -= fy
          f2.fx += fx
          f2.fy += fy
        }
      }

      // Attraction along edges
      edges.forEach(edge => {
        const pos1 = positions.get(edge.source)
        const pos2 = positions.get(edge.target)
        if (!pos1 || !pos2) return

        const dx = pos2.x - pos1.x
        const dy = pos2.y - pos1.y
        const dist = Math.sqrt(dx * dx + dy * dy) || 1

        const force = dist * 0.01
        const fx = (dx / dist) * force
        const fy = (dy / dist) * force

        const f1 = forces.get(edge.source)!
        const f2 = forces.get(edge.target)!
        f1.fx += fx
        f1.fy += fy
        f2.fx -= fx
        f2.fy -= fy
      })

      // Center gravity
      nodes.forEach(node => {
        const pos = positions.get(node.id)!
        const dx = centerX - pos.x
        const dy = centerY - pos.y
        const force = forces.get(node.id)!
        force.fx += dx * 0.001
        force.fy += dy * 0.001
      })

      // Apply forces
      nodes.forEach(node => {
        const pos = positions.get(node.id)!
        const force = forces.get(node.id)!
        pos.x += force.fx * 0.5
        pos.y += force.fy * 0.5
        pos.x = Math.max(50, Math.min(width - 50, pos.x))
        pos.y = Math.max(50, Math.min(height - 50, pos.y))
      })
    }

    setNodePositions(positions)
    setInitialized(true)
  }, [nodes, edges])

  // Initialize positions when nodes change
  useEffect(() => {
    if (nodes.length > 0) {
      initializePositions()
    }
  }, [nodes, edges, initializePositions])

  // Mouse wheel zoom
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault()
    const delta = e.deltaY > 0 ? 0.9 : 1.1
    setTransform(prev => ({
      ...prev,
      scale: Math.max(0.2, Math.min(3, prev.scale * delta))
    }))
  }, [])

  // Pan handling
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button === 0) {
      setIsDragging(true)
    }
  }, [])

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (isDragging && !draggedNode) {
      setTransform(prev => ({
        ...prev,
        x: prev.x + e.movementX,
        y: prev.y + e.movementY
      }))
    }
  }, [isDragging, draggedNode])

  const handleMouseUp = useCallback(() => {
    setIsDragging(false)
    setDraggedNode(null)
  }, [])

  // Zoom controls
  const zoomIn = () => setTransform(prev => ({
    ...prev,
    scale: Math.min(3, prev.scale * 1.2)
  }))

  const zoomOut = () => setTransform(prev => ({
    ...prev,
    scale: Math.max(0.2, prev.scale / 1.2)
  }))

  const resetView = () => {
    setTransform({ x: 0, y: 0, scale: 1 })
    initializePositions()
  }

  // Get node color
  const getNodeColor = (type: string) => NODE_COLORS[type] || NODE_COLORS.Unknown
  const getNodeRadius = (type: string) => NODE_RADII[type] || NODE_RADII.Unknown

  if (nodes.length === 0) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-slate-800/50 rounded-lg border border-slate-700">
        <div className="text-center text-slate-400">
          <svg className="w-16 h-16 mx-auto mb-4 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z" />
          </svg>
          <p className="text-sm">No graph data available</p>
          <p className="text-xs mt-1">Ask a question to see the knowledge graph</p>
        </div>
      </div>
    )
  }

  return (
    <div ref={containerRef} className="relative w-full h-full overflow-hidden bg-gradient-to-br from-slate-900 to-slate-800 rounded-lg border border-slate-700">
      {/* Zoom Controls */}
      <div className="absolute top-4 right-4 z-10 flex flex-col gap-2">
        <button
          onClick={zoomIn}
          className="p-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors"
          title="Zoom in"
        >
          <ZoomIn className="w-4 h-4" />
        </button>
        <button
          onClick={zoomOut}
          className="p-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors"
          title="Zoom out"
        >
          <ZoomOut className="w-4 h-4" />
        </button>
        <button
          onClick={resetView}
          className="p-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors"
          title="Reset view"
        >
          <Maximize2 className="w-4 h-4" />
        </button>
      </div>

      {/* Legend */}
      <div className="absolute bottom-4 left-4 z-10 bg-slate-800/90 backdrop-blur-sm rounded-lg p-3 border border-slate-700">
        <h4 className="text-xs font-semibold text-slate-300 mb-2">Node Types</h4>
        <div className="grid grid-cols-2 gap-x-4 gap-y-1">
          {Object.entries(NODE_COLORS).filter(([key]) => key !== 'Unknown').map(([type, color]) => (
            <div key={type} className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: color }} />
              <span className="text-xs text-slate-400">{type}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Graph SVG */}
      <svg
        ref={svgRef}
        className="w-full h-full cursor-grab active:cursor-grabbing"
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        <g transform={`translate(${transform.x}, ${transform.y}) scale(${transform.scale})`}>
          {/* Edges */}
          <g className="edges">
            {edges.map((edge, index) => {
              const sourcePos = nodePositions.get(edge.source)
              const targetPos = nodePositions.get(edge.target)
              if (!sourcePos || !targetPos) return null

              const dx = targetPos.x - sourcePos.x
              const dy = targetPos.y - sourcePos.y
              const dr = Math.sqrt(dx * dx + dy * dy)
              
              // Calculate edge label position (midpoint)
              const labelX = (sourcePos.x + targetPos.x) / 2
              const labelY = (sourcePos.y + targetPos.y) / 2

              return (
                <g key={`edge-${index}`}>
                  <line
                    x1={sourcePos.x}
                    y1={sourcePos.y}
                    x2={targetPos.x}
                    y2={targetPos.y}
                    stroke="#64748b"
                    strokeWidth={1.5 * edge.weight}
                    strokeOpacity={0.6}
                    markerEnd="url(#arrowhead)"
                  />
                  <text
                    x={labelX}
                    y={labelY - 8}
                    className="edge-label fill-slate-400"
                    textAnchor="middle"
                    fontSize={9}
                  >
                    {edge.label}
                  </text>
                </g>
              )
            })}
          </g>

          {/* Arrow marker definition */}
          <defs>
            <marker
              id="arrowhead"
              markerWidth="10"
              markerHeight="7"
              refX="10"
              refY="3.5"
              orient="auto"
            >
              <polygon points="0 0, 10 3.5, 0 7" fill="#64748b" />
            </marker>
          </defs>

          {/* Nodes */}
          <g className="nodes">
            {nodes.map((node) => {
              const pos = nodePositions.get(node.id)
              if (!pos) return null
              const radius = getNodeRadius(node.type)
              const color = getNodeColor(node.type)

              return (
                <g
                  key={node.id}
                  transform={`translate(${pos.x}, ${pos.y})`}
                  onClick={() => onNodeClick?.(node.id)}
                  className="cursor-pointer"
                >
                  {/* Glow effect */}
                  <circle
                    r={radius + 4}
                    fill={color}
                    opacity={0.2}
                    className="animate-pulse"
                  />
                  {/* Main node */}
                  <circle
                    r={radius}
                    fill={color}
                    stroke="#1e293b"
                    strokeWidth={2}
                    className="transition-all hover:scale-110"
                  />
                  {/* Icon based on type */}
                  <text
                    y={1}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fill="white"
                    fontSize={12}
                    fontWeight="bold"
                  >
                    {node.type[0]}
                  </text>
                  {/* Label */}
                  <text
                    y={radius + 16}
                    textAnchor="middle"
                    fill="#e2e8f0"
                    fontSize={11}
                    className="node-label font-medium"
                  >
                    {node.label.length > 15 ? node.label.slice(0, 15) + '...' : node.label}
                  </text>
                </g>
              )
            })}
          </g>
        </g>
      </svg>

      {/* Stats */}
      <div className="absolute top-4 left-4 z-10 bg-slate-800/90 backdrop-blur-sm rounded-lg px-3 py-2 border border-slate-700">
        <div className="flex gap-4 text-xs">
          <span className="text-slate-400">
            <span className="text-slate-200 font-semibold">{nodes.length}</span> nodes
          </span>
          <span className="text-slate-400">
            <span className="text-slate-200 font-semibold">{edges.length}</span> edges
          </span>
        </div>
      </div>
    </div>
  )
}
