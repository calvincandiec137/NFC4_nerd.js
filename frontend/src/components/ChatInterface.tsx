import { useState } from 'react'
import { Send, Download, Volume2, Volume, VolumeX, Mic, ChevronLeft, ChevronRight } from 'lucide-react'
import MessageBubble from './MessageBubble'

interface Message {
  id: number
  sender: 'bot' | 'user'
  text: string
}

interface ChatInterfaceProps {
  messages: Message[]
  onSendMessage: (text: string) => void
  isPlaying: boolean
  setIsPlaying: (playing: boolean) => void
  volume: number
  setVolume: (volume: number) => void
}

export default function ChatInterface({
  messages,
  onSendMessage,
  isPlaying,
  setIsPlaying,
  volume,
  setVolume,
}: ChatInterfaceProps) {
  const [inputValue, setInputValue] = useState('')
  const [currentConversation, setCurrentConversation] = useState(0)

  const handleSend = () => {
    if (inputValue.trim()) {
      onSendMessage(inputValue)
      setInputValue('')
    }
  }

  const getVolumeIcon = () => {
    if (volume === 0) return <VolumeX size={18} />
    if (volume < 50) return <Volume size={18} />
    return <Volume2 size={18} />
  }

  return (
    <div className="flex-1 flex flex-col bg-bg-white">
      {/* Header */}
      <div className="flex items-center justify-between p-6 border-b border-bg-tan">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-primary rounded-full flex items-center justify-center">
            <span className="text-white text-lg font-bold">📄</span>
          </div>
          <div>
            <h1 className="text-xl font-bold text-text">DOC.AI Assistant</h1>
            <p className="text-sm text-text-light">Upload, process, and chat with your documents using AI.</p>
          </div>
        </div>
        <button className="flex items-center gap-2 bg-accent text-white px-4 py-2 rounded-lg hover:bg-accent/90 transition-colors font-medium">
          <Download size={18} />
          Download All
        </button>
      </div>

      {/* Chat Container */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-center text-2xl font-bold text-text mb-8">Chat with DOC.AI</h2>

          <div className="space-y-4">
            {messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))}
          </div>
        </div>
      </div>

      {/* Navigation Arrows */}
      <div className="absolute right-8 top-1/2 -translate-y-1/2 flex flex-col gap-4">
        <button
          onClick={() => setCurrentConversation(Math.max(0, currentConversation - 1))}
          className="p-2 bg-primary rounded-full text-white hover:bg-primary/90 transition-colors shadow-lg"
        >
          <ChevronLeft size={24} />
        </button>
        <button
          onClick={() => setCurrentConversation(currentConversation + 1)}
          className="p-2 bg-primary rounded-full text-white hover:bg-primary/90 transition-colors shadow-lg"
        >
          <ChevronRight size={24} />
        </button>
      </div>

      {/* Input Area */}
      <div className="p-6 border-t border-bg-tan">
        <div className="max-w-4xl mx-auto">
          {/* Audio Controls */}
          <div className="flex items-center gap-3 mb-4 pb-4 border-b border-bg-tan">
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className="p-2 hover:bg-bg-tan rounded transition-colors text-primary"
              title="Play/Pause"
            >
              <Mic size={20} />
            </button>
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className="p-2 hover:bg-bg-tan rounded transition-colors text-primary"
              title="Toggle audio"
            >
              {getVolumeIcon()}
            </button>
            <input
              type="range"
              min="0"
              max="100"
              value={volume}
              onChange={(e) => setVolume(Number(e.target.value))}
              className="flex-1 h-2 bg-bg-tan rounded cursor-pointer accent-primary"
            />
            <span className="text-sm text-text-light w-8 text-right">{volume}</span>
          </div>

          {/* Input Field */}
          <div className="flex items-center gap-3">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSend()}
              placeholder="Ask me anything about your files..."
              className="flex-1 bg-bg-tan text-text placeholder-text-light px-4 py-3 rounded-full focus:outline-none focus:ring-2 focus:ring-primary"
            />
            <button
              onClick={handleSend}
              className="p-3 bg-primary text-white rounded-full hover:bg-primary/90 transition-colors"
              title="Send message"
            >
              <Send size={20} />
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
