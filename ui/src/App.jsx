import { useState, useEffect, useRef } from "react";
import { api } from "./api.js";
import Sidebar from "./components/Sidebar.jsx";
import ChatArea from "./components/ChatArea.jsx";

export default function App() {
  const [sessions, setSessions] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const firstLoad = useRef(true);

  const refreshSessions = async () => {
    try {
      setSessions(await api.listSessions());
    } catch {
      setSessions([]);
    }
  };

  useEffect(() => {
    refreshSessions();
  }, []);

  useEffect(() => {
    if (activeId) {
      api
        .listMessages(activeId)
        .then(setMessages)
        .catch(() => setMessages([]));
    } else {
      setMessages([]);
    }
  }, [activeId]);

  const newChat = async () => {
    const session = await api.createSession();
    setActiveId(session.id);
    await refreshSessions();
  };

  const removeSession = async (id) => {
    await api.deleteSession(id);
    if (id === activeId) setActiveId(null);
    await refreshSessions();
  };

  const send = async (content) => {
    if (streaming || !content.trim()) return;
    setStreaming(true);
    setLoading(true);

    let sessionId = activeId;
    try {
      if (!sessionId) {
        const session = await api.createSession();
        sessionId = session.id;
        await refreshSessions();
      }
    } catch (err) {
      setStreaming(false);
      setLoading(false);
      return;
    }

    const userMsg = {
      id: `local-user-${Date.now()}`,
      role: "user",
      content,
      created_at: new Date().toISOString(),
    };
    const assistantMsg = {
      id: `local-ai-${Date.now()}`,
      role: "assistant",
      content: "",
      sources: [],
      answer_type: null,
      streaming: true,
    };
    setMessages((prev) => [...prev, userMsg, assistantMsg]);

    api.streamMessage(sessionId, content, {
      onSources: (sources, answerType) =>
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsg.id
              ? { ...m, sources, answer_type: answerType }
              : m
          )
        ),
      onDelta: (delta) =>
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsg.id
              ? { ...m, content: m.content + delta }
              : m
          )
        ),
      onDone: (persisted) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsg.id
              ? {
                  ...m,
                  content: persisted.content,
                  sources: persisted.sources,
                  answer_type: persisted.answer_type,
                  streaming: false,
                }
              : m
          )
        );
        setStreaming(false);
        setLoading(false);
        refreshSessions();
        if (sessionId !== activeId) setActiveId(sessionId);
        firstLoad.current = false;
      },
      onError: (err) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsg.id
              ? {
                  ...m,
                  content:
                    m.content || `⚠ Error: ${err.message || "request failed"}`,
                  streaming: false,
                }
              : m
          )
        );
        setStreaming(false);
        setLoading(false);
      },
    });
  };

  return (
    <div className="app">
      <Sidebar
        sessions={sessions}
        activeId={activeId}
        onSelect={setActiveId}
        onNew={newChat}
        onDelete={removeSession}
      />
      <ChatArea
        messages={messages}
        loading={loading}
        streaming={streaming}
        onSend={send}
      />
    </div>
  );
}
