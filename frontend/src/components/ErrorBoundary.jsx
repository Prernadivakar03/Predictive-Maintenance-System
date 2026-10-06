import { Component } from "react";

/** Keeps one failing page from taking down the whole console. */
export default class ErrorBoundary extends Component {
  state = { error: null };
  static getDerivedStateFromError(error) { return { error }; }
  componentDidCatch(error, info) { console.error("UI error:", error, info?.componentStack); }
  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="banner bad" role="alert">
        <div style={{ flex: 1 }}>
          <strong>This page hit an unexpected error.</strong>
          <div className="muted">{String(this.state.error.message || this.state.error)}</div>
        </div>
        <button className="btn sm" onClick={() => { this.setState({ error: null }); this.props.onReset?.(); }}>Try again</button>
      </div>
    );
  }
}
