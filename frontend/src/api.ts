import type {
  EventItem,
  ExportResponse,
  Health,
  ImportResponse,
  Job,
  ReviewStatus,
  VolumeScan
} from "./types";

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {})
    },
    ...init
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch {
      // Keep status text.
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<Health>("/api/health"),
  volumes: () => request<VolumeScan>("/api/volumes"),
  importFolder: (folderPath: string) =>
    request<ImportResponse>("/api/import", {
      method: "POST",
      body: JSON.stringify({ folder_path: folderPath })
    }),
  analyze: (sessionId: string) =>
    request<Job>("/api/analyze", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId })
    }),
  job: (jobId: string) => request<Job>(`/api/jobs/${jobId}`),
  events: (sessionId?: string) =>
    request<EventItem[]>(sessionId ? `/api/events?session_id=${sessionId}` : "/api/events"),
  review: (eventId: string, reviewStatus: ReviewStatus, location: string, note: string) =>
    request<EventItem>(`/api/events/${eventId}/review`, {
      method: "PATCH",
      body: JSON.stringify({ review_status: reviewStatus, location, note })
    }),
  exportEvent: (eventId: string) =>
    request<ExportResponse>(`/api/events/${eventId}/export`, { method: "POST" })
};
