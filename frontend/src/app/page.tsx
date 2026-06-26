'use client'

import React, { useState, useRef, useEffect } from 'react'
import { Send, Loader2, MessageSquare, Network, ChevronRight, Info, Clock, Database } from 'lucide-react'
import GraphVisualization from './components/GraphVisualization'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  context?: {
    nodes: Array<{ id: string; name: string; type: string; properties?: any }>
    edges: Array<{ source_id: string; target_id: string; relationship_type: string }>
    text_blocks: string[]
    anchor_entities: string[]
  }
  sources?: string[]
  reasoning_steps?: string[]
  confidence?: number
  query_time_ms?: number
}

interface GraphNode {
  id: string
  label: string
  type: string
}

interface GraphEdge {
  source: string
  target: string
  label: string
  weight: number
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [selectedMessage, setSelectedMessage] = useState<string | null>(null)
  const [viewMode, setViewMode] = useState<'split' | 'chat' | 'graph'>('split')
  const [systemStats, setSystemStats] = useState<{
    nodes: number
    edges: number
    documents: number
  }>({ nodes: 0, edges: 0, documents: 0 })
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  // API base URL - change for production
  const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Fetch system stats on mount
  useEffect(() => {
    fetchSystemStats()
  }, [])

  const fetchSystemStats = async () => {
    try {
      const response = await fetch(`${API_URL}/stats`)
      if (response.ok) {
        const data = await response.json()
        setSystemStats({
          nodes: data.total_nodes || 0,
          edges: data.total_edges || 0,
          documents: data.documents_processed || 0
        })
      }
    } catch (error) {
      console.error('Failed to fetch stats:', error)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    try {
      const response = await fetch(`${API_URL}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: userMessage.content,
          top_k: 5,
          max_hops: 2,
          include_reasoning: true
        })
      })

      if (!response.ok) {
        throw new Error('Query failed')
      }

      const data = await response.json()

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.answer,
        timestamp: new Date(),
        context: {
          nodes: data.retrieved_context?.nodes || [],
          edges: data.retrieved_context?.edges || [],
          text_blocks: data.retrieved_context?.text_blocks || [],
          anchor_entities: data.retrieved_context?.anchor_entities || []
        },
        sources: data.sources || [],
        reasoning_steps: data.reasoning_steps || [],
        confidence: data.confidence,
        query_time_ms: data.query_time_ms
      }

      setMessages(prev => [...prev, assistantMessage])
      setSelectedMessage(assistantMessage.id)
      
      // Update stats
      fetchSystemStats()
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'I encountered an error processing your query. Please make sure the GraphRAG backend is running and try again.',
        timestamp: new Date()
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  // Get graph data for visualization from selected message
  const getGraphData = () => {
    const selected = messages.find(m => m.id === selectedMessage)
    if (!selected?.context) return { nodes: [], edges: [] }

    const nodes: GraphNode[] = selected.context.nodes.map(n => ({
      id: n.id,
      label: n.name,
      type: n.type
    }))

    const edges: GraphEdge[] = selected.context.edges.map(e => ({
      source: e.source_id,
      target: e.target_id,
      label: e.relationship_type,
      weight: 1
    }))

    return { nodes, edges }
  }

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }

  return (
    <div className="flex h-full">
      {/* Main Chat Panel */}
      <div className={`${viewMode === 'graph' ? 'hidden' : 'flex-1'} flex flex-col border-r border-slate-700`}>
        {/* Chat Header */}
        <div className="px-6 py-3 bg-slate-800/50 border-b border-slate-700 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <MessageSquare className="w-5 h-5 text-primary-400" />
            <span className="font-medium text-white">Query Chat</span>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setViewMode('split')}
              className={`px-3 py-1 text-xs rounded-md transition-colors ${
                viewMode === 'split' 
                  ? 'bg-primary-500 text-white' 
                  : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
              }`}
            >
              Split
            </button>
            <button
              onClick={() => setViewMode('chat')}
              className={`px-3 py-1 text-xs rounded-md transition-colors ${
                viewMode === 'chat' 
                  ? 'bg-primary-500 text-white' 
                  : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
              }`}
            >
              Chat
            </button>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6 chat-container">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <div className="w-16 h-16 bg-primary-500/20 rounded-full flex items-center justify-center mb-4">
                <MessageSquare className="w-8 h-8 text-primary-400" />
              </div>
              <h2 className="text-xl font-semibold text-white mb-2">Ask about your knowledge graph</h2>
              <p className="text-slate-400 max-w-md">
                Query your documents using natural language. The system will retrieve relevant context from the knowledge graph.
              </p>
            </div>
          )}

          {messages.map(message => (
            <div
              key={message.id}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div className={`max-w-[80%] ${message.role === 'user' ? 'order-2' : 'order-1'}`}>
                <div
                  className={`rounded-2xl px-4 py-3 ${
                    message.role === 'user'
                      ? 'bg-primary-500 text-white'
                      : 'bg-slate-800 text-slate-100 border border-slate-700'
                  }`}
                >
                  <p className="whitespace-pre-wrap">{message.content}</p>
                </div>
                
                {/* Message metadata */}
                {message.role === 'assistant' && message.context && (
                  <div className="mt-2 flex items-center gap-4 text-xs text-slate-500">
                    <span className="flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      {message.query_time_ms?.toFixed(0)}ms
                    </span>
                    <span className="flex items-center gap-1">
                      <Database className="w-3 h-3" />
                      {message.context.nodes.length} entities
                    </span>
                    {message.confidence && (
                      <span className="flex items-center gap-1">
                        <span className="text-slate-400">Confidence:</span>
                        <span className={message.confidence > 0.7 ? 'text-emerald-400' : 'text-amber-400'}>
                          {(message.confidence * 100).toFixed(0)}%
                        </span>
                      </span>
                    )}
                    <button
                      onClick={() => setSelectedMessage(selectedMessage === message.id ? null : message.id)}
                      className="flex items-center gap-1 text-primary-400 hover:text-primary-300 transition-colors"
                    >
                      <Info className="w-3 h-3" />
                      {selectedMessage === message.id ? 'Hide' : 'Show'} Context
                    </button>
                  </div>
                )}

                <div className="mt-1 text-xs text-slate-600">
                  {formatTime(message.timestamp)}
                </div>
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-slate-800 rounded-2xl px-4 py-3 border border-slate-700">
                <div className="flex items-center gap-2 text-slate-400">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Processing query...</span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="p-4 bg-slate-800/50 border-t border-slate-700">
          <form onSubmit={handleSubmit} className="flex gap-3">
            <textarea
              ref={inputRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about your documents..."
              className="flex-1 bg-slate-900 border border-slate-700 rounded-xl px-4 py-3 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none"
              rows={2}
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className="px-6 py-3 bg-primary-500 hover:bg-primary-600 disabled:bg-slate-700 disabled:text-slate-500 text-white rounded-xl font-medium transition-colors flex items-center gap-2"
            >
              <Send className="w-4 h-4" />
              Send
            </button>
          </form>
        </div>
      </div>

      {/* Context/Metadata Panel */}
      {(viewMode === 'split' || viewMode === 'chat') && selectedMessage && (
        <div className={`${viewMode === 'graph' ? 'hidden' : 'w-96'} bg-slate-800/50 flex flex-col`}>
          <div className="px-4 py-3 border-b border-slate-700 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Network className="w-4 h-4 text-primary-400" />
              <span className="font-medium text-sm">Metadata & Context</span>
            </div>
            <button
              onClick={() => setSelectedMessage(null)}
              className="text-slate-400 hover:text-white transition-colors"
            >
              ×
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {(() => {
              const msg = messages.find(m => m.id === selectedMessage)
              if (!msg?.context) return null

              return (
                <>
                  {/* Reasoning Steps */}
                  {msg.reasoning_steps && msg.reasoning_steps.length > 0 && (
                    <div>
                      <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                        Reasoning Steps
                      </h3>
                      <div className="space-y-1">
                        {msg.reasoning_steps.map((step, i) => (
                          <div key={i} className="flex items-start gap-2 text-sm">
                            <ChevronRight className="w-4 h-4 text-slate-500 mt-0.5 flex-shrink-0" />
                            <span className="text-slate-300">{step}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Anchor Entities */}
                  {msg.context.anchor_entities.length > 0 && (
                    <div>
                      <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                        Anchor Entities
                      </h3>
                      <div className="flex flex-wrap gap-1">
                        {msg.context.anchor_entities.slice(0, 10).map((entity, i) => (
                          <span key={i} className="px-2 py-0.5 bg-slate-700 rounded text-xs text-slate-300">
                            {entity}
                          </span>
                        ))}
                        {msg.context.anchor_entities.length > 10 && (
                          <span className="px-2 py-0.5 text-xs text-slate-500">
                            +{msg.context.anchor_entities.length - 10} more
                          </span>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Extracted Nodes */}
                  <div>
                    <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                      Extracted Entities ({msg.context.nodes.length})
                    </h3>
                    <div className="space-y-2">
                      {msg.context.nodes.slice(0, 20).map((node, i) => (
                        <div key={i} className="bg-slate-900 rounded-lg p-2 border border-slate-700">
                          <div className="flex items-center gap-2">
                            <span className={`w-2 h-2 rounded-full ${
                              node.type === 'Person' ? 'bg-emerald-400' :
                              node.type === 'System' ? 'bg-blue-400' :
                              node.type === 'Asset' ? 'bg-purple-400' :
                              'bg-slate-400'
                            }`} />
                            <span className="font-medium text-sm text-white">{node.name}</span>
                          </div>
                          <span className="text-xs text-slate-500 ml-4">{node.type}</span>
                          {node.properties?.description && (
                            <p className="text-xs text-slate-400 mt-1 ml-4 line-clamp-2">
                              {node.properties.description}
                            </p>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Text Blocks */}
                  {msg.context.text_blocks.length > 0 && (
                    <div>
                      <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                        Extracted Facts
                      </h3>
                      <div className="space-y-2">
                        {msg.context.text_blocks.slice(0, 10).map((block, i) => (
                          <div key={i} className="text-sm text-slate-300 bg-slate-900/50 rounded p-2 border border-slate-700/50">
                            {block}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              )
            })()}
          </div>
        </div>
      )}

      {/* Graph Visualization Panel */}
      <div className={`${viewMode === 'chat' ? 'hidden' : 'flex-1'} flex flex-col`}>
        <div className="px-6 py-3 bg-slate-800/50 border-b border-slate-700 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Network className="w-5 h-5 text-primary-400" />
            <span className="font-medium text-white">Graph Visualization</span>
          </div>
          <div className="flex items-center gap-4 text-xs text-slate-400">
            <span>{systemStats.nodes} nodes</span>
            <span>{systemStats.edges} edges</span>
            <span>{systemStats.documents} docs</span>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setViewMode('graph')}
              className={`px-3 py-1 text-xs rounded-md transition-colors ${
                viewMode === 'graph' 
                  ? 'bg-primary-500 text-white' 
                  : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
              }`}
            >
              Graph
            </button>
          </div>
        </div>

        <div className="flex-1 p-4">
          <GraphVisualization
            nodes={getGraphData().nodes}
            edges={getGraphData().edges}
          />
        </div>
      </div>
    </div>
  )
}
