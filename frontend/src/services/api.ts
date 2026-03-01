// API service for communicating with the backend

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export interface Message {
  id: number
  sender: 'bot' | 'user'
  text: string
}

// Upload PDF file to backend
export async function uploadDocument(file: File): Promise<{ status: string }> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    
    reader.onload = async () => {
      try {
        const base64Data = reader.result as string
        // Remove the data:application/pdf;base64, prefix
        const base64String = base64Data.split(',')[1]
        
        const response = await fetch(`${API_BASE_URL}/upload`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            data: base64String,
            name: file.name,
          }),
        })
        
        if (!response.ok) {
          throw new Error('Upload failed')
        }
        
        const data = await response.json()
        resolve(data)
      } catch (error) {
        reject(error)
      }
    }
    
    reader.onerror = () => reject(new Error('Failed to read file'))
    reader.readAsDataURL(file)
  })
}

// Prepare/ingest the uploaded documents
export async function prepareDocuments(): Promise<{ status: string }> {
  const response = await fetch(`${API_BASE_URL}/prepare`, {
    method: 'GET',
  })
  
  if (!response.ok) {
    throw new Error('Failed to prepare documents')
  }
  
  return response.json()
}

// Ask a question to the RAG system
export async function askQuestion(question: string): Promise<{ question: string; answer: string }> {
  const response = await fetch(`${API_BASE_URL}/ask?q=${encodeURIComponent(question)}`, {
    method: 'GET',
  })
  
  if (!response.ok) {
    throw new Error('Failed to get answer')
  }
  
  return response.json()
}

// Check if backend is running
export async function checkHealth(): Promise<{ message: string }> {
  const response = await fetch(`${API_BASE_URL}/`, {
    method: 'GET',
  })
  
  if (!response.ok) {
    throw new Error('Backend is not responding')
  }
  
  return response.json()
}
