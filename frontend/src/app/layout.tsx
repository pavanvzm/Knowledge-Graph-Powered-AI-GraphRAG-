import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'GraphRAG - Knowledge Graph AI',
  description: 'Hybrid Vector + Graph retrieval system powered by AI',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-900 text-slate-100">
        <div className="flex flex-col h-screen">
          <header className="bg-slate-800 border-b border-slate-700 px-6 py-4">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-gradient-to-br from-primary-500 to-primary-700 rounded-lg flex items-center justify-center">
                <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <div>
                <h1 className="text-xl font-bold text-white">GraphRAG Engine</h1>
                <p className="text-xs text-slate-400">Knowledge Graph-Powered AI Retrieval</p>
              </div>
            </div>
          </header>
          <main className="flex-1 overflow-hidden">
            {children}
          </main>
        </div>
      </body>
    </html>
  )
}
