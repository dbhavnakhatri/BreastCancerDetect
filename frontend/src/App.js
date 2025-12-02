import React, { useMemo, useState } from "react";
import "./App.css";
import { FiUploadCloud } from "react-icons/fi";

const getDefaultApiBase = () => {
  const envUrl = process.env.REACT_APP_API_BASE_URL;
  if (envUrl && envUrl.trim().length > 0) {
    return envUrl.replace(/\/$/, "");
  }

  if (typeof window !== "undefined") {
    const localHosts = ["localhost", "127.0.0.1", "0.0.0.0"];
    if (localHosts.includes(window.location.hostname)) {
      return "http://127.0.0.1:8000";
    }
  }

  return "/api";
};

const buildEndpoint = (base, endpoint) => {
  const safeBase = base.endsWith("/") ? base.slice(0, -1) : base;
  const safeEndpoint = endpoint.startsWith("/") ? endpoint : `/${endpoint}`;
  return `${safeBase}${safeEndpoint}`;
};

const asDataUrl = (value) => (value ? `data:image/png;base64,${value}` : null);

function App() {
  const apiBase = useMemo(() => getDefaultApiBase(), []);
  const apiUrl = (endpoint) => buildEndpoint(apiBase, endpoint);

  const [results, setResults] = useState({});
  const [file, setFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [analysisDone, setAnalysisDone] = useState(false);
  const [visualTab, setVisualTab] = useState("overlay");
  const [detailsTab, setDetailsTab] = useState("model");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");
  const [errorMessage, setErrorMessage] = useState("");

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      analyzeFile(selectedFile);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragActive(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setDragActive(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      setFile(droppedFile);
      analyzeFile(droppedFile);
    }
  };

  const handleBrowseClick = () => {
    const input = document.getElementById("fileInput");
    if (input) input.click();
  };

  const handleBackToUpload = () => {
    setAnalysisDone(false);
    setResults({});
    setFile(null);
    setVisualTab("overlay");
    setDetailsTab("model");
    setStatusMessage("");
    setErrorMessage("");
  };

  const analyzeFile = async (selectedFile) => {
    if (!selectedFile) return;
    const formData = new FormData();
    formData.append("file", selectedFile);

    setIsAnalyzing(true);
    setStatusMessage("Uploading image for analysis…");
    setErrorMessage("");

    try {
      const response = await fetch(apiUrl("/analyze"), {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorBody = await response.json().catch(() => ({}));
        throw new Error(errorBody.detail || "Analysis failed.");
      }

      const data = await response.json();
      const images = data.images || {};
      const confidencePercent =
        data.confidence !== undefined && data.confidence <= 1
          ? data.confidence * 100
          : data.confidence ?? null;

      setResults({
        original: asDataUrl(images.original),
        overlay: asDataUrl(images.overlay),
        heatmap: asDataUrl(images.heatmap_only),
        bbox: asDataUrl(images.bbox),
        malignant: data.malignant_prob ?? null,
        benign: data.benign_prob ?? null,
        risk: data.risk_level ?? "Unavailable",
        riskIcon: data.risk_icon,
        riskColor: data.risk_color,
        result: data.result ?? "Analysis Result",
        confidence: confidencePercent,
        rawScore: data.confidence ?? null,
        threshold: data.threshold ?? 0.5,
        stats: data.stats || {},
      });

      setAnalysisDone(true);
      setVisualTab("overlay");
      setDetailsTab("model");
      setStatusMessage("Analysis complete.");
    } catch (error) {
      console.error(error);
      setErrorMessage(error.message || "Backend not reachable.");
      setStatusMessage("");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleDownloadReport = async () => {
    if (!file) {
      setErrorMessage("Please upload a file before requesting the report.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    setIsGeneratingReport(true);
    setErrorMessage("");

    try {
      const response = await fetch(apiUrl("/report"), {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorBody = await response.json().catch(() => ({}));
        throw new Error(errorBody.detail || "Failed to generate report.");
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = "breast_cancer_report.pdf";
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error(error);
      setErrorMessage(error.message || "Error while downloading report.");
    } finally {
      setIsGeneratingReport(false);
    }
  };

  const getActiveVisualImage = () => {
    switch (visualTab) {
      case "heatmap":
        return results.heatmap;
      case "bbox":
        return results.bbox;
      case "original":
        return results.original;
      case "overlay":
      default:
        return results.overlay;
    }
  };

  const statsList = [
    {
      label: "Mean Intensity",
      value:
        results.stats?.mean_intensity !== undefined
          ? results.stats.mean_intensity.toFixed(1)
          : "—",
    },
    {
      label: "Std. Deviation",
      value:
        results.stats?.std_intensity !== undefined
          ? results.stats.std_intensity.toFixed(1)
          : "—",
    },
    {
      label: "Brightness",
      value:
        results.stats?.brightness !== undefined
          ? `${results.stats.brightness.toFixed(1)}%`
          : "—",
    },
    {
      label: "Contrast",
      value:
        results.stats?.contrast !== undefined
          ? `${results.stats.contrast.toFixed(1)}%`
          : "—",
    },
  ];

  return (
    <div className="App">
      <video autoPlay muted loop id="bg-video">
        <source src="/backgroundpink.mp4" type="video/mp4" />
      </video>
      <div className="bg-overlay" />

      <header className="header">
        <div className="logo">
          <img src="/Group 28.png" alt="logo" />
          <span>XrayAI</span>
        </div>
      </header>

      <section className="hero">
        <h1>Breast Cancer Analysis</h1>
        <p>AI-powered X-ray analysis for educational use only.</p>
      </section>

      {!analysisDone ? (
        <section className="upload-section">
          <div className="upload-card">
            <h3>Upload X-ray (JPG / PNG / DCM)</h3>
            <p>Max 200MB • Supported formats: JPG, JPEG, PNG, DCM</p>
            <div
              className={`dropzone ${dragActive ? "active" : ""}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={handleBrowseClick}
            >
              <FiUploadCloud
                size={50}
                style={{ color: "#AE70AF", marginBottom: "10px" }}
              />
              <p className="drop-main-text">
                {isAnalyzing ? "Analyzing…" : "Drag & drop file here"}
              </p>
              <p className="drop-sub-text">or click to browse files</p>
              <button
                type="button"
                className="btn-primary"
                onClick={(event) => {
                  event.stopPropagation();
                  handleBrowseClick();
                }}
                disabled={isAnalyzing}
              >
                {isAnalyzing ? "Processing…" : "Browse File"}
              </button>
              <input
                type="file"
                id="fileInput"
                style={{ display: "none" }}
                onChange={handleFileChange}
                accept=".jpg,.jpeg,.png,.dcm"
                disabled={isAnalyzing}
              />
            </div>

            {file && (
              <p className="selected-file">
                Selected File: <strong>{file.name}</strong>
              </p>
            )}
            {statusMessage && (
              <p className="muted small" style={{ marginTop: "10px" }}>
                {statusMessage}
              </p>
            )}
            {errorMessage && (
              <p className="muted small" style={{ color: "#ff8080" }}>
                {errorMessage}
              </p>
            )}
          </div>
        </section>
      ) : (
        <main className="analysis-container">
          <button className="back-btn" onClick={handleBackToUpload}>
            ← Back to Upload
          </button>

          <section className="analysis-card">
            <div className="result-header">
              <h2 className="result-title">
                {results.result || "Analysis Result"}
              </h2>
              <p className="risk-pill">
                Risk Level:&nbsp;
                <strong>
                  {results.riskIcon ? `${results.riskIcon} ` : ""}
                  {results.risk || "Not available"}
                </strong>
              </p>
              {results.confidence != null && (
                <p className="confidence-text">
                  Model Confidence:&nbsp;
                  <strong>{results.confidence.toFixed(2)}%</strong>
                </p>
              )}
            </div>

            {(statusMessage || errorMessage) && (
              <div style={{ marginBottom: "15px" }}>
                {statusMessage && (
                  <p className="muted small">{statusMessage}</p>
                )}
                {errorMessage && (
                  <p className="muted small" style={{ color: "#ff8080" }}>
                    {errorMessage}
                  </p>
                )}
              </div>
            )}

            <section className="section">
              <h3 className="section-title">Prediction Metrics</h3>
              <div className="metric-grid">
                <div className="metric">
                  <span className="metric-label">Benign</span>
                  <h3>
                    {results.benign != null
                      ? `${results.benign.toFixed(2)}%`
                      : "—"}
                  </h3>
                </div>
                <div className="metric">
                  <span className="metric-label">Malignant</span>
                  <h3>
                    {results.malignant != null
                      ? `${results.malignant.toFixed(2)}%`
                      : "—"}
                  </h3>
                </div>
                <div className="metric">
                  <span className="metric-label">Model Confidence</span>
                  <h3>
                    {results.confidence != null
                      ? `${results.confidence.toFixed(2)}%`
                      : "—"}
                  </h3>
                </div>
              </div>
            </section>

            <section className="section">
              <h3 className="section-title">Visual Analysis</h3>
              <p className="section-subtitle">
                Grad-CAM attention maps showing which regions influenced the
                model&apos;s decision.
              </p>

              <div className="visual-tabs">
                <button
                  className={`visual-tab ${
                    visualTab === "overlay" ? "active" : ""
                  }`}
                  onClick={() => setVisualTab("overlay")}
                >
                  Heatmap Overlay
                </button>
                <button
                  className={`visual-tab ${
                    visualTab === "heatmap" ? "active" : ""
                  }`}
                  onClick={() => setVisualTab("heatmap")}
                >
                  Heatmap Only
                </button>
                <button
                  className={`visual-tab ${visualTab === "bbox" ? "active" : ""}`}
                  onClick={() => setVisualTab("bbox")}
                >
                  Region Detection (BBox)
                </button>
                <button
                  className={`visual-tab ${
                    visualTab === "original" ? "active" : ""
                  }`}
                  onClick={() => setVisualTab("original")}
                >
                  Original Image
                </button>
              </div>

              <div className="visual-panel">
                <div className="visual-image-card">
                  {getActiveVisualImage() ? (
                    <img src={getActiveVisualImage()} alt="Visual analysis" />
                  ) : (
                    <p className="muted small">Image not available.</p>
                  )}
                </div>
              </div>
            </section>

            <section className="section">
              <h3 className="section-title">Model & Risk Details</h3>
              <div className="details-tabs">
                <button
                  className={`details-tab ${
                    detailsTab === "model" ? "active" : ""
                  }`}
                  onClick={() => setDetailsTab("model")}
                >
                  Model Information
                </button>
                <button
                  className={`details-tab ${
                    detailsTab === "risk" ? "active" : ""
                  }`}
                  onClick={() => setDetailsTab("risk")}
                >
                  Risk Guide
                </button>
                <button
                  className={`details-tab ${
                    detailsTab === "heatmapInfo" ? "active" : ""
                  }`}
                  onClick={() => setDetailsTab("heatmapInfo")}
                >
                  Heatmap Tips
                </button>
                <button
                  className={`details-tab ${
                    detailsTab === "clinical" ? "active" : ""
                  }`}
                  onClick={() => setDetailsTab("clinical")}
                >
                  Clinical Context
                </button>
              </div>

              <div className="details-panel">
                {detailsTab === "model" && (
                  <div>
                    <h4 className="details-heading">Deep Learning Model Details</h4>
                    <ul className="details-list">
                      <li>
                        <strong>Type:</strong> Convolutional Neural Network (CNN)
                      </li>
                      <li>
                        <strong>Input Size:</strong> 224 × 224 × 3 (RGB)
                      </li>
                      <li>
                        <strong>Output:</strong> Binary classification
                      </li>
                      <li>
                        <strong>Visualization:</strong> Grad-CAM attention maps
                      </li>
                    </ul>
                  </div>
                )}
                {detailsTab === "risk" && (
                  <div>
                    <h4 className="details-heading">Risk Assessment Guide</h4>
                    <ul className="details-list">
                      <li>0–20% → low risk, routine monitoring.</li>
                      <li>20–50% → moderate risk, closer observation.</li>
                      <li>50–80% → high risk, follow-up recommended.</li>
                      <li>80–100% → very high risk, urgent specialist review.</li>
                    </ul>
                  </div>
                )}
                {detailsTab === "heatmapInfo" && (
                  <div>
                    <h4 className="details-heading">Heatmap Interpretation</h4>
                    <ul className="details-list">
                      <li>Red/yellow regions show strong model attention.</li>
                      <li>Green/blue areas had less impact on the prediction.</li>
                      <li>
                        Overlay view keeps anatomical context, heatmap-only shows
                        pure activation.
                      </li>
                      <li>
                        Heatmaps highlight attention, not confirmed pathology.
                      </li>
                    </ul>
                  </div>
                )}
                {detailsTab === "clinical" && (
                  <div>
                    <h4 className="details-heading">Clinical Context</h4>
                    <ul className="details-list">
                      <li>This interface is an educational demo, not a medical device.</li>
                      <li>
                        Predictions can be wrong due to image quality or dataset bias.
                      </li>
                      <li>
                        Always seek professional medical advice before making decisions.
                      </li>
                    </ul>
                    <p className="muted small">
                      ❗ Never change treatment plans based solely on this tool.
                    </p>
                  </div>
                )}
              </div>
            </section>

            <section className="section">
              <h3 className="section-title">Image Quality & Intensity</h3>
              <div className="stats-grid">
                {statsList.map((item) => (
                  <div className="stat-item" key={item.label}>
                    <span>{item.label}</span>
                    <strong>{item.value}</strong>
                  </div>
                ))}
              </div>
            </section>

            <div className="btn-row" style={{ flexDirection: "column", gap: "16px" }}>
              <button
                className="btn-primary"
                onClick={handleDownloadReport}
                disabled={isGeneratingReport}
              >
                {isGeneratingReport ? "Preparing Report…" : "Download PDF Report"}
              </button>
              <button className="btn-secondary" onClick={handleBackToUpload}>
                Analyze Another Image
              </button>
            </div>

            <p className="muted small footer-note">
              ⚠ Educational demo only — not for real clinical diagnosis.
            </p>
          </section>
        </main>
      )}
    </div>
  );
}

export default App;
