import React from "react";
import { FaUpload } from "react-icons/fa";

function UploadCard({ setFile }) {
  const handleFileChange = (e) => {
    if (e.target.files[0]) {
      setFile(e.target.files[0]);
      analyzeFile(e.target.files[0]);

    }
  };

  return (
    <div className="upload-card">
      <h3>Upload X-ray (JPG/PNG/DCM)</h3>
      <p>Limit 200MB per file • JPG, JPEG, PNG, DCM</p>

      <div className="dropzone">
        <FaUpload size={32} style={{ marginBottom: "10px", color: "#AE70AF" }} />
        <p>Drag and drop file here</p>
        <input type="file" onChange={handleFileChange} />
        <button className="btn-primary">Browse File</button>
      </div>
    </div>
  );
}

export default UploadCard;