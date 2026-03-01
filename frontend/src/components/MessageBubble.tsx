interface Message {
  id: number
  sender: 'bot' | 'user'
  text: string
}

interface MessageBubbleProps {
  message: Message
}

export default function MessageBubble({ message }: MessageBubbleProps) {
  const isBot = message.sender === 'bot'

  return (
    <div className={`flex ${isBot ? 'justify-start' : 'justify-end'} items-end gap-3`}>
      {isBot && (
        <div className="w-8 h-8 bg-primary rounded-full flex items-center justify-center flex-shrink-0">
          <span className="text-white text-sm font-bold">📄</span>
        </div>
      )}

      <div
        className={`max-w-2xl px-4 py-3 rounded-lg ${
          isBot
            ? 'bg-secondary text-text'
            : 'bg-primary text-white'
        }`}
      >
        <p className="text-sm leading-relaxed">{message.text}</p>
      </div>

      {!isBot && (
        <div className="w-8 h-8 bg-primary rounded-full flex items-center justify-center flex-shrink-0 text-white font-bold text-sm">
          {message.id}
        </div>
      )}
    </div>
  )
}
