import { useMemo, useState } from "react";
import { marked } from "marked";
import DOMPurify from "dompurify";

marked.setOptions({ breaks: true, gfm: true });

function SourceCard({ source }) {
  const meta = source.metadata || {};
  const filename = (meta.source || "unknown").split(/[\\/]/).pop();
  const score = (source.similarity_score || 0).toFixed(3);
  const preview = (source.content || "").slice(0, 200);

  return (
    <div className="source-card">
      <div className="source-head">
        <span className="source-file">{filename}</span>
        <span className="source-score">{score}</span>
      </div>
      {preview && <p className="source-preview">{preview}…</p>}
    </div>
  );
}

function Sources({ sources }) {
  const [open, setOpen] = useState(false);
  if (!sources || sources.length === 0) return null;
  return (
    <div className="sources">
      <button className="sources-toggle" onClick={() => setOpen(!open)}>
        {open ? "Hide" : "View"} sources ({sources.length})
      </button>
      {open && (
        <div className="sources-list">
          {sources.map((s, i) => (
            <SourceCard key={i} source={s} />
          ))}
        </div>
      )}
    </div>
  );
}

function TypeBadge({ type }) {
  if (!type || type === "general") return null;
  const label = type === "api" ? "API spec" : type === "feature" ? "Feature explain" : type;
  return <span className={`type-badge ${type}`}>{label}</span>;
}

export default function Message({ message }) {
  const html = useMemo(() => {
    const raw = marked.parse(message.content || "");
    return DOMPurify.sanitize(raw);
  }, [message.content]);

  return (
    <div className={`message ${message.role}`}>
      <div className="avatar">{message.role === "user" ? "You" : "AI"}</div>
      <div className="bubble">
        {message.role === "assistant" ? (
          <>
            <TypeBadge type={message.answer_type} />
            <div
              className="markdown"
              dangerouslySetInnerHTML={{ __html: html }}
            />
            <Sources sources={message.sources} />
          </>
        ) : (
          <p>{message.content}</p>
        )}
        {message.streaming && <span className="cursor" />}
      </div>
    </div>
  );
}
