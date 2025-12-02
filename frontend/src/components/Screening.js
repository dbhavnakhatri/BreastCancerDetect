import React from "react";

function Screening({ tasks }) {
  return (
    <div className="card">
      <h2>Disease Screening</h2>
      {tasks.map((task, idx) => (
        <div className="d-card" key={idx}>
          <div className="d-head">
            <span>{task.name}</span>
            <span className="badge">{Math.round(task.prob * 100)}%</span>
          </div>
          <div className="bar">
            <div style={{ width: `${task.prob * 100}%` }}></div>
          </div>
          <p className="muted">{task.desc}</p>
        </div>
      ))}
    </div>
  );
}

export default Screening;