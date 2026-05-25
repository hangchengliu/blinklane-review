export type ReviewStatus = "pending" | "confirmed" | "dismissed";

export interface Health {
  ok: boolean;
  ffmpeg_available: boolean;
  yolo_available: boolean;
  cv2_available: boolean;
  data_dir: string;
  messages: string[];
}

export interface Session {
  id: string;
  folder_path: string;
  status: string;
  source_kind: string;
  created_at: string;
  notes?: string;
}

export interface VideoSegment {
  id: string;
  session_id: string;
  camera: string;
  starts_at: string;
  path: string;
  duration_s?: number | null;
  fps?: number | null;
  width?: number | null;
  height?: number | null;
}

export interface ImportResponse {
  session: Session;
  segments: VideoSegment[];
  reused: boolean;
}

export interface Job {
  id: string;
  session_id: string;
  status: "queued" | "running" | "completed" | "failed";
  progress: number;
  message: string;
  model_name: string;
  sample_rate_fps: number;
  created_at: string;
  updated_at: string;
  event_count: number;
}

export interface EventItem {
  id: string;
  session_id: string;
  segment_id: string;
  camera: string;
  track_id: number;
  vehicle_type: string;
  start_s: number;
  end_s: number;
  key_s: number;
  lane_change_score: number;
  turn_signal_score: number;
  confidence: number;
  reason_labels: string[];
  review_status: ReviewStatus;
  location: string;
  note: string;
  raw_clip_url?: string | null;
  annotated_clip_url?: string | null;
  key_frame_url?: string | null;
}

export interface ExportResponse {
  id: string;
  event_id: string;
  path: string;
  zip_path: string;
  download_url: string;
}
