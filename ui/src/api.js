const BASE = "/api";

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

export const api = {
  health: () => request("/health"),

  listSessions: () => request("/sessions"),
  createSession: (title = "New chat") =>
    request("/sessions", { method: "POST", body: JSON.stringify({ title }) }),
  deleteSession: (id) =>
    request(`/sessions/${id}`, { method: "DELETE" }),

  listMessages: (sessionId) => request(`/sessions/${sessionId}/messages`),

  sendMessage: (sessionId, content) =>
    request(`/sessions/${sessionId}/messages`, {
      method: "POST",
      body: JSON.stringify({ content }),
    }),

  streamMessage: (sessionId, content, { onSources, onDelta, onDone, onError }) => {
    fetch(`${BASE}/sessions/${sessionId}/messages/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content }),
    })
      .then(async (res) => {
        if (!res.ok || !res.body) {
          const text = await res.text();
          throw new Error(text || `Stream failed: ${res.status}`);
        }
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        const handle = (chunk) => {
          buffer += decoder.decode(chunk, { stream: true });
          const events = buffer.split("\n\n");
          buffer = events.pop() || "";
          for (const evt of events) {
            const lines = evt.split("\n");
            const eventLine = lines.find((l) => l.startsWith("event: "));
            const dataLine = lines.find((l) => l.startsWith("data: "));
            if (!eventLine || !dataLine) continue;
            const event = eventLine.slice(7).trim();
            let data;
            try {
              data = JSON.parse(dataLine.slice(6));
            } catch {
              continue;
            }
            if (event === "sources") onSources(data.sources, data.answer_type);
            else if (event === "message") onDelta(data.delta);
            else if (event === "done") onDone(data.message);
          }
        };

        const pump = () =>
          reader.read().then(({ done, value }) => {
            if (done) {
              handle(new Uint8Array(0));
              return;
            }
            handle(value);
            return pump();
          });

        return pump();
      })
      .catch(onError);
  },
};
