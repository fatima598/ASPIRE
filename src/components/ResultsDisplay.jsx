import React from 'react'
import '../styles/ResultsDisplay.css'

const ResultsDisplay = ({ results, pickupImages, returnImages, onReset }) => {
  const { summary, damage_breakdown, detailed_damages } = results

  const getSeverityColor = (score) => {
    if (score >= 7) return '#e74c3c'
    if (score >= 4) return '#f39c12'
    return '#27ae60'
  }

  const getSeverityText = (score) => {
    if (score >= 7) return 'High'
    if (score >= 4) return 'Medium'
    return 'Low'
  }

  return (
    <div className="results-display">
      <div className="results-header">
        <h2>Damage Assessment Report</h2>
        <button className="new-assessment-button" onClick={onReset}>
          🆕 New Assessment
        </button>
      </div>

      {/* Summary Cards */}
      <div className="summary-cards">
        <div className="summary-card">
          <div className="card-icon">🔍</div>
          <div className="card-content">
            <h3>{summary.new_damages_count}</h3>
            <p>New Damages</p>
          </div>
        </div>

        <div className="summary-card">
          <div className="card-icon">💰</div>
          <div className="card-content">
            <h3>${summary.total_repair_cost}</h3>
            <p>Estimated Cost</p>
          </div>
        </div>

        <div className="summary-card">
          <div className="card-icon" style={{ color: getSeverityColor(summary.severity_score) }}>
            ⚠️
          </div>
          <div className="card-content">
            <h3 style={{ color: getSeverityColor(summary.severity_score) }}>
              {summary.severity_score}/10
            </h3>
            <p>Severity ({getSeverityText(summary.severity_score)})</p>
          </div>
        </div>
      </div>

      {/* Damage Breakdown */}
      {Object.keys(damage_breakdown).length > 0 && (
        <div className="damage-breakdown">
          <h3>Damage Breakdown</h3>
          <div className="breakdown-list">
            {Object.entries(damage_breakdown).map(([type, count]) => (
              <div key={type} className="breakdown-item">
                <span className="damage-type">{type}</span>
                <span className="damage-count">{count} found</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Detailed Damages */}
      {detailed_damages.length > 0 && (
        <div className="detailed-damages">
          <h3>Detailed Damage Report</h3>
          <div className="damages-list">
            {detailed_damages.map((damage, index) => (
              <div key={index} className="damage-item">
                <div className="damage-info">
                  <span className="damage-type-badge">{damage.type}</span>
                  <span className="confidence">
                    Confidence: {(damage.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="damage-location">
                  Location: {damage.image_source}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* No Damages Case */}
      {detailed_damages.length === 0 && (
        <div className="no-damages">
          <div className="success-icon">✅</div>
          <h3>No New Damages Detected!</h3>
          <p>The vehicle is in the same condition as when it was picked up.</p>
        </div>
      )}

      <div className="assessment-meta">
        <p><strong>Assessment ID:</strong> {results.assessment_id}</p>
        <p><strong>Processed:</strong> {summary.assessment_time}</p>
        <p><strong>Images Analyzed:</strong> {results.images_processed.pickup_images} pickup, {results.images_processed.return_images} return</p>
      </div>
    </div>
  )
}

export default ResultsDisplay