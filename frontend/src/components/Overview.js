import React from "react";

function Overview({ results }) {
  if (!results || !results.original) {
    return <p>No results yet...</p>;
  }

  return (
    <div className="card">
      <h2>Prediction Result: {results.result}</h2>

      <div className="metric-grid">
        <div className="metric"><span>Benign</span><h3>{results.benign.toFixed(2)}%</h3></div>
        <div className="metric"><span>Malignant</span><h3>{results.malignant.toFixed(2)}%</h3></div>
        <div className="metric"><span>Risk</span><h3>{results.risk}</h3></div>
      </div>

      <div className="img-grid">
        <div>
          <p className="muted">Original Image</p>
          <img src={results.original} alt="Original" />
        </div>

        <div>
          <p className="muted">Overlay Heatmap</p>
          <img src={results.overlay} alt="Overlay" />
        </div>
      </div>

      <div className="img-grid">
        <div>
          <p className="muted">Heatmap Only</p>
          <img src={results.heatmap} alt="Heatmap Only" />
        </div>

        <div>
          <p className="muted">Bounding Box</p>
          <img src={results.bbox} alt="bbox" />
        </div>
      </div>
    </div>
  );
}

export default Overview;
