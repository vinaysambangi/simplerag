import { useEffect, useRef, useState } from "react";
import Message from "./Message.jsx";

export default function ChatArea({ messages, loading, streaming, onSend }) {
  const [input, setInput] = useState("");
  const bottomRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const submit = () => {
    const value = input.trim();
    if (!value) return;
    setInput("");
    onSend(value);
    requestAnimationFrame(() => textareaRef.current?.focus());
  };

  const onKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <main className="chat">
      <div className="chat-scroll">
        {messages.length === 0 ? (
          <div className="welcome">
            <h1>Documentation Assistant</h1>
            <p>
              Ask anything about the OEM API / protocol documentation.
              Each new chat is independent — previous chats never leak
              context into a new one.
            </p>
          </div>
        ) : (
          messages.map((m) => <Message key={m.id} message={m} />)
        )}
        {loading && (
          <div className="typing">
            <span />
            <span />
            <span />
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="composer">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Ask a question about the documentation…"
          rows={1}
          disabled={streaming}
        />
        <button className="send-btn" onClick={submit} disabled={streaming || !input.trim()}>
          Send
        </button>
      </div>
    </main>
  );
}
