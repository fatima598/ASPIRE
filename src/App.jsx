import React, { useState } from 'react'
import ImageUpload from './components/ImageUpload'
import ResultsDisplay from './components/ResultsDisplay'
import { assessDamage } from './services/api'
import './styles/App.css'

function App() {
  const [pickupImages, setPickupImages] = useState([])
  const [returnImages, setReturnImages] = useState([])
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleAssessDamage = async () => {
    if (pickupImages.length === 0 || returnImages.length === 0) {
      setError('Please upload both pickup and return images')
      return
    }

    setLoading(true)
    setError(null)
    
    try {
      const assessmentResults = await assessDamage(pickupImages, returnImages)
      setResults(assessmentResults)
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to assess damage')
    } finally {
      setLoading(false)
    }
  }

  const resetForm = () => {
    setPickupImages([])
    setReturnImages([])
    setResults(null)
    setError(null)
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>🚗 Vehicle Damage Assessment</h1>
        <p>AI-powered vehicle inspection for rental businesses</p>
      </header>

      <main className="app-main">
        {!results ? (
          <div className="assessment-interface">
            <div className="upload-sections">
              <ImageUpload
                title="Pickup Images"
                images={pickupImages}
                setImages={setPickupImages}
                description="Upload images taken when vehicle was picked up"
              />
              
              <ImageUpload
                title="Return Images" 
                images={returnImages}
                setImages={setReturnImages}
                description="Upload images taken when vehicle was returned"
              />
            </div>

            {error && (
              <div className="error-message">
                ⚠️ {error}
              </div>
            )}

            <button 
              className="assess-button"
              onClick={handleAssessDamage}
              disabled={loading || pickupImages.length === 0 || returnImages.length === 0}
            >
              {loading ? '🔍 Analyzing Damages...' : '🚀 Assess Damage'}
            </button>
          </div>
        ) : (
          <ResultsDisplay 
            results={results}
            pickupImages={pickupImages}
            returnImages={returnImages}
            onReset={resetForm}
          />
        )}
      </main>

      <footer className="app-footer">
        <p>Powered by AI • Built for Aspire Software Hiring Sprint 2025</p>
      </footer>
    </div>
  )
}

export default App