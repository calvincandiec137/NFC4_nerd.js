import { useState, useEffect } from 'react'
import Sidebar from './components/Sidebar'
import ChatInterface from './components/ChatInterface'
import FileUpload from './components/FileUpload'
import { askQuestion, prepareDocuments, checkHealth } from './services/api'

interface Message {
  id: number
  sender: 'bot' | 'user'
  text: string
}

export default function App() {
  const [activeMenu, setActiveMenu] = useState('file')
  const [messages, setMessages] = useState<Message[]>([])
  const [isPlaying, setIsPlaying] = useState(false)
  const [volume, setVolume] = useState(50)
  const [isDocumentReady, setIsDocumentReady] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [backendStatus, setBackendStatus] = useState<'checking' | 'online' | 'offline'>('checking')

  // Check backend health on mount
  useEffect(() => {
    checkHealth()
      .then(() => setBackendStatus('online'))
      .catch(() => setBackendStatus('offline'))
  }, [])

  const handleUploadComplete = async () => {
    setIsProcessing(true)
    try {
      // Prepare/ingest the uploaded document
      await prepareDocuments()
      setIsDocumentReady(true)
      setActiveMenu('chat')
      
      // Add welcome message
      setMessages([
        {
          id: 1,
          sender: 'bot',
          text: 'Document uploaded and processed successfully! You can now ask me questions about your document.',
        },
      ])
    } catch (error) {
      setMessages([
        {
          id: 1,
          sender: 'bot',
          text: `Error processing document: ${error instanceof Error ? error.message : 'Unknown error'}`,
        },
      ])
    } finally {
      setIsProcessing(false)
    }
  }

  const handleSendMessage = async (text: string) => {
    if (!text.trim()) return

    // Add user message
    const userMessage: Message = {
      id: messages.length + 1,
      sender: 'user',
      text: text,
    }
    setMessages((prev) => [...prev, userMessage])

    // Add loading message
    const loadingMessage: Message = {
      id: messages.length + 2,
      sender: 'bot',
      text: 'Thinking...',
    }
    setMessages((prev) => [...prev, loadingMessage])

    try {
      // Call backend API
      const response = await askQuestion(text)
      
      // Replace loading message with actual response
      setMessages((prev) => 
        prev.map((msg) =>
          msg.id === loadingMessage.id
            ? { ...msg, text: response.answer }
            : msg
        )
      )
    } catch (error) {
      // Replace loading message with error
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === loadingMessage.id
            ? { ...msg, text: `Error: ${error instanceof Error ? error.message : 'Failed to get response'}` }
            : msg
        )
      )
    }
  }

  return (
    <div className="flex h-screen bg-bg-light">
      <Sidebar activeMenu={activeMenu} setActiveMenu={setActiveMenu} />
      
      <div className="flex-1 flex flex-col">
        {/* Backend Status Indicator */}
        {backendStatus === 'offline' && (
          <div className="bg-red-500 text-white px-4 py-2 text-center text-sm">
            ⚠️ Backend is offline. Please start the backend server: <code>uvicorn modules.app:app --reload</code>
          </div>
        )}
        {backendStatus === 'online' && !isDocumentReady && activeMenu === 'file' && (
          <div className="bg-accent text-white px-4 py-2 text-center text-sm">
            ✓ Backend is online. Upload a PDF to get started!
          </div>
        )}
        {isProcessing && (
          <div className="bg-primary text-white px-4 py-2 text-center text-sm">
            🔄 Processing document... This may take a moment.
          </div>
        )}

        {/* Content based on active menu */}
        {activeMenu === 'file' ? (
          <div className="flex-1 flex flex-col items-center justify-center">
            <h2 className="text-2xl font-bold text-text mb-6">Upload Your Document</h2>
            <FileUpload onUploadComplete={handleUploadComplete} />
          </div>
        ) : activeMenu === 'chat' ? (
          <ChatInterface
            messages={messages}
            onSendMessage={handleSendMessage}
            isPlaying={isPlaying}
            setIsPlaying={setIsPlaying}
            volume={volume}
            setVolume={setVolume}
          />
        ) : (
          <div className="flex-1 flex items-center justify-center">
            <p className="text-text-light text-lg">
              {activeMenu === 'history' && 'Chat History (Coming Soon)'}
              {activeMenu === 'download' && 'Download Options (Coming Soon)'}
              {activeMenu === 'analytics' && 'Analytics (Coming Soon)'}
              {activeMenu === 'settings' && 'Settings (Coming Soon)'}
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
