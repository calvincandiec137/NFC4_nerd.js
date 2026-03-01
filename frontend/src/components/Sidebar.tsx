import { Menu, Folder, MessageCircle, History, Download, BarChart3, Settings, ChevronLeft, ChevronRight } from 'lucide-react'
import { useState } from 'react'

interface SidebarProps {
  activeMenu: string
  setActiveMenu: (menu: string) => void
}

export default function Sidebar({ activeMenu, setActiveMenu }: SidebarProps) {
  const [isExpanded, setIsExpanded] = useState(true)

  const menuItems = [
    { id: 'file', label: 'File Manager', icon: Folder },
    { id: 'chat', label: 'Chat Interface', icon: MessageCircle },
    { id: 'history', label: 'Chat History', icon: History },
    { id: 'download', label: 'Download Options', icon: Download },
    { id: 'analytics', label: 'Analytics', icon: BarChart3 },
    { id: 'settings', label: 'Settings', icon: Settings },
  ]

  return (
    <div className={`flex flex-col bg-secondary transition-all duration-300 ${isExpanded ? 'w-56' : 'w-20'} border-r border-primary`}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-primary">
        {isExpanded && (
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-primary rounded flex items-center justify-center">
              <span className="text-white text-sm font-bold">📄</span>
            </div>
            <span className="text-primary font-bold text-lg">DOC.AI</span>
          </div>
        )}
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="p-1 hover:bg-primary/20 rounded transition-colors"
        >
          {isExpanded ? (
            <ChevronLeft size={20} className="text-primary" />
          ) : (
            <ChevronRight size={20} className="text-primary" />
          )}
        </button>
      </div>

      {/* Menu Items */}
      <nav className="flex-1 p-3 space-y-2">
        {menuItems.map((item) => {
          const Icon = item.icon
          const isActive = activeMenu === item.id
          return (
            <button
              key={item.id}
              onClick={() => setActiveMenu(item.id)}
              className={`w-full flex items-center gap-3 px-3 py-2 rounded transition-colors ${
                isActive
                  ? 'bg-primary/20 text-primary font-medium'
                  : 'text-text-light hover:bg-primary/10'
              }`}
              title={item.label}
            >
              <Icon size={20} />
              {isExpanded && <span className="text-sm">{item.label}</span>}
            </button>
          )
        })}
      </nav>

      {/* Navigation Arrows at Bottom */}
      <div className="flex gap-2 p-4 border-t border-primary justify-center">
        <button
          onClick={() => console.log('Previous')}
          className="p-2 bg-primary rounded-full text-white hover:bg-primary/90 transition-colors"
        >
          <ChevronLeft size={20} />
        </button>
        <button
          onClick={() => console.log('Next')}
          className="p-2 bg-primary rounded-full text-white hover:bg-primary/90 transition-colors"
        >
          <ChevronRight size={20} />
        </button>
      </div>
    </div>
  )
}
