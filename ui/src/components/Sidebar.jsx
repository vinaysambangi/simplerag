export default function Sidebar({ sessions, activeId, onSelect, onNew, onDelete }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="logo">Simple<span>RAG</span></div>
        <button className="new-chat-btn" onClick={onNew}>
          + New chat
        </button>
      </div>

      <div className="session-list">
        {sessions.length === 0 && (
          <div className="empty-state">No chats yet — start a new one.</div>
        )}
        {sessions.map((s) => (
          <div
            key={s.id}
            className={`session-item ${s.id === activeId ? "active" : ""}`}
            onClick={() => onSelect(s.id)}
          >
            <span className="session-title">{s.title}</span>
            <button
              className="delete-btn"
              title="Delete chat"
              onClick={(e) => {
                e.stopPropagation();
                onDelete(s.id);
              }}
            >
              ✕
            </button>
          </div>
        ))}
      </div>

      <div className="sidebar-footer">
        <div className="status-dot" />
        New chats start fresh — no context from previous chats.
      </div>
    </aside>
  );
}
