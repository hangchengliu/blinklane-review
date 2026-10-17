import type {
  EventItem,
  ExportResponse,
  Health,
  ImportResponse,
  Job,
  ReviewResponse,
  ReviewStatus,
  VolumeScan
} from "./types";

const DEFAULT_TIMEOUT_MS = 60_000;

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...init,
    signal: init?.signal ?? AbortSignal.timeout(DEFAULT_TIMEOUT_MS),
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {})
    }
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      if (typeof body?.detail === "string") {
        detail = body.detail;
      } else if (body?.detail != null) {
        detail = JSON.stringify(body.detail);
      }
    } catch {
      // Keep status text.
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: (signal?: AbortSignal) => request<Health>("/api/health", { signal }),
  volumes: (signal?: AbortSignal) => request<VolumeScan>("/api/volumes", { signal }),
  importFolder: (folderPath: string, signal?: AbortSignal) =>
    request<ImportResponse>("/api/import", {
      method: "POST",
      body: JSON.stringify({ folder_path: folderPath }),
      signal
    }),
  analyze: (sessionId: string, signal?: AbortSignal) =>
    request<Job>("/api/analyze", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId }),
      signal
    }),
  job: (jobId: string, signal?: AbortSignal) => request<Job>(`/api/jobs/${jobId}`, { signal }),
  events: (sessionId?: string, signal?: AbortSignal) =>
    request<EventItem[]>(
      sessionId ? `/api/events?${new URLSearchParams({ session_id: sessionId })}` : "/api/events",
      { signal }
    ),
  review: (
    eventId: string,
    reviewStatus: ReviewStatus,
    location: string,
    note: string,
    plate: string,
    signal?: AbortSignal
  ) =>
    request<ReviewResponse>(`/api/events/${eventId}/review`, {
      method: "PATCH",
      body: JSON.stringify({ review_status: reviewStatus, location, note, plate }),
      signal
    }),
  exportEvent: (eventId: string, signal?: AbortSignal) =>
    request<ExportResponse>(`/api/events/${eventId}/export`, { method: "POST", signal })
};
