import React, { useRef } from 'react'
import '../styles/ImageUpload.css'

const ImageUpload = ({ title, images, setImages, description }) => {
  const fileInputRef = useRef(null)

  const handleFileSelect = (event) => {
    const files = Array.from(event.target.files)
    setImages(prev => [...prev, ...files])
  }

  const removeImage = (indexToRemove) => {
    setImages(prev => prev.filter((_, index) => index !== indexToRemove))
  }

  const handleDrop = (event) => {
    event.preventDefault()
    const files = Array.from(event.dataTransfer.files)
    setImages(prev => [...prev, ...files])
  }

  const handleDragOver = (event) => {
    event.preventDefault()
  }

  return (
    <div className="image-upload">
      <h3>{title}</h3>
      <p className="upload-description">{description}</p>
      
      <div 
        className="upload-area"
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onClick={() => fileInputRef.current?.click()}
      >
        <div className="upload-placeholder">
          <div className="upload-icon">📸</div>
          <p>Click to upload or drag and drop</p>
          <small>Supports JPG, PNG, JPEG</small>
        </div>
        
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept="image/*"
          onChange={handleFileSelect}
          style={{ display: 'none' }}
        />
      </div>

      {images.length > 0 && (
        <div className="image-previews">
          <h4>Selected Images ({images.length})</h4>
          <div className="preview-grid">
            {images.map((image, index) => (
              <div key={index} className="image-preview">
                <img 
                  src={URL.createObjectURL(image)} 
                  alt={`Preview ${index + 1}`}
                />
                <button 
                  className="remove-button"
                  onClick={(e) => {
                    e.stopPropagation()
                    removeImage(index)
                  }}
                >
                  ✕
                </button>
                <span className="image-name">{image.name}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default ImageUpload