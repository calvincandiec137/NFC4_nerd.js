import { useState } from 'react'
import { Upload, FileText, CheckCircle, AlertCircle, Loader2 } from 'lucide-react'

interface FileUploadProps {
  onUploadComplete: () => void
}

export default function FileUpload({ onUploadComplete }: FileUploadProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle')
  const [uploadedFile, setUploadedFile] = useState<string | null>(null)
  const [errorMessage, setErrorMessage] = useState<string>('')

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    const files = Array.from(e.dataTransfer.files)
    const pdfFile = files.find((file) => file.type === 'application/pdf')

    if (pdfFile) {
      await uploadFile(pdfFile)
    } else {
      setErrorMessage('Please upload a PDF file')
      setUploadStatus('error')
    }
  }

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file && file.type === 'application/pdf') {
      await uploadFile(file)
    } else {
      setErrorMessage('Please select a PDF file')
      setUploadStatus('error')
    }
  }

  const uploadFile = async (file: File) => {
    setUploadStatus('uploading')
    setErrorMessage('')

    try {
      // Import API service
      const { uploadDocument } = await import('../services/api')
      
      await uploadDocument(file)
      setUploadedFile(file.name)
      setUploadStatus('success')
      onUploadComplete()
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : 'Upload failed')
      setUploadStatus('error')
    }
  }

  return (
    <div className="max-w-2xl mx-auto p-6">
      <div
        className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
          isDragging
            ? 'border-accent bg-accent/10'
            : uploadStatus === 'success'
            ? 'border-accent bg-accent/5'
            : uploadStatus === 'error'
            ? 'border-red-400 bg-red-50'
            : 'border-bg-tan hover:border-primary'
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <input
          type="file"
          accept=".pdf,application/pdf"
          onChange={handleFileSelect}
          className="hidden"
          id="file-upload"
          disabled={uploadStatus === 'uploading'}
        />

        <label htmlFor="file-upload" className="cursor-pointer">
          <div className="flex flex-col items-center gap-4">
            {uploadStatus === 'uploading' && (
              <>
                <Loader2 size={48} className="text-primary animate-spin" />
                <p className="text-text font-medium">Uploading...</p>
              </>
            )}

            {uploadStatus === 'success' && (
              <>
                <CheckCircle size={48} className="text-accent" />
                <p className="text-text font-medium">Upload Successful!</p>
                <p className="text-sm text-text-light flex items-center gap-2">
                  <FileText size={16} />
                  {uploadedFile}
                </p>
              </>
            )}

            {uploadStatus === 'error' && (
              <>
                <AlertCircle size={48} className="text-red-500" />
                <p className="text-text font-medium">Upload Failed</p>
                <p className="text-sm text-red-600">{errorMessage}</p>
              </>
            )}

            {uploadStatus === 'idle' && (
              <>
                <Upload size={48} className="text-primary" />
                <div>
                  <p className="text-text font-medium mb-1">
                    Drop your PDF here or click to browse
                  </p>
                  <p className="text-sm text-text-light">
                    Maximum file size: 10MB
                  </p>
                </div>
              </>
            )}
          </div>
        </label>
      </div>

      {uploadStatus === 'success' && (
        <p className="text-center text-sm text-text-light mt-4">
          You can now chat with your document below
        </p>
      )}
    </div>
  )
}
