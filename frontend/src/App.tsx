import { AlertCircle, CheckCircle2, Download, FileVideo, Play, RefreshCw, XCircle } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import type {
  EventItem,
  ExportResponse,
  Health,
  ImportResponse,
  Job,
  ReviewStatus,
  VolumeScan
} from "./types";

const statusText: Record<ReviewStatus, string> = {
  pending: "待定",
  confirmed: "确认",
  dismissed: "排除"
};

export function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [folderPath, setFolderPath] = useState("");
  const [importResult, setImportResult] = useState<ImportResponse | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [events, setEvents] = useState<EventItem[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [exportResult, setExportResult] = useState<ExportResponse | null>(null);
  const [volumes, setVolumes] = useState<VolumeScan>({ scanning: false, volumes: [] });
  const loadedVolume = useRef<string | null>(null);

  const selected = useMemo(
    () => events.find((event) => event.id === selectedId) || events[0] || null,
    [events, selectedId]
  );

  useEffect(() => {
    api.health().then(setHealth).catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    let stop = false;
    async function pollVolumes() {
      try {
        const data = await api.volumes();
        if (!stop) setVolumes(data);
      } catch (err) {
        if (!stop) setError(err instanceof Error ? err.message : "卷扫描失败");
      }
    }
    void pollVolumes();
    const timer = window.setInterval(() => void pollVolumes(), 4000);
    return () => {
      stop = true;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    if (!job || (job.status !== "queued" && job.status !== "running")) return;
    const timer = window.setInterval(async () => {
      try {
        const fresh = await api.job(job.id);
        setJob(fresh);
        if (fresh.status === "completed") {
          await refreshEvents(fresh.session_id);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "任务状态获取失败");
      }
    }, 1200);
    return () => window.clearInterval(timer);
  }, [job]);

  async function refreshEvents(sessionId = importResult?.session.id) {
    if (!sessionId) return;
    const items = await api.events(sessionId);
    setEvents(items);
    if (items.length && !selectedId) setSelectedId(items[0].id);
  }

  async function loadFolder(folder: string) {
    setError("");
    setBusy(true);
    setJob(null);
    try {
      const result = await api.importFolder(folder);
      setFolderPath(folder);
      setImportResult(result);
      setEvents([]);
      setSelectedId(null);
      setExportResult(null);
      await refreshEvents(result.session.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "导入失败");
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    if (importResult || busy) return;
    const imported = volumes.volumes.filter((item) => item.status === "imported");
    if (imported.length !== 1) return;
    const folder = imported[0].folder_path;
    if (loadedVolume.current === folder) return;
    loadedVolume.current = folder;
    void loadFolder(folder);
  }, [volumes, importResult, busy]);

  async function handleImport() {
    const folder = folderPath.trim();
    if (!folder) return;
    await loadFolder(folder);
  }

  async function handleAnalyze() {
    if (!importResult) return;
    setError("");
    setBusy(true);
    try {
      const created = await api.analyze(importResult.session.id);
      setJob(created);
    } catch (err) {
      setError(err instanceof Error ? err.message : "分析启动失败");
    } finally {
      setBusy(false);
    }
  }

  async function handleReview(status: ReviewStatus, location: string, note: string, plate: string) {
    if (!selected) return;
    setError("");
    try {
      const updated = await api.review(selected.id, status, location, note, plate);
      setEvents((items) => items.map((item) => (item.id === updated.id ? updated : item)));
      setSelectedId(updated.id);
      if (status === "confirmed" && updated.zip_path && updated.download_url) {
        setExportResult({
          id: updated.id,
          event_id: updated.id,
          path: "",
          zip_path: updated.zip_path,
          download_url: updated.download_url,
          submission_message: updated.submission_message || undefined,
          report_url: updated.report_url || undefined
        });
        if (updated.report_url) {
          window.open(updated.report_url, "_blank", "noopener,noreferrer");
        }
      } else {
        setExportResult(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "保存复核失败");
    }
  }

  async function handleExport() {
    if (!selected) return;
    setError("");
    try {
      const exported = await api.exportEvent(selected.id);
      setExportResult(exported);
    } catch (err) {
      setError(err instanceof Error ? err.message : "导出失败");
    }
  }

  return (
    <main className="shell">
      <header className="topbar">
        <div>
          <h1>BlinkLane / 变道灯光复核助手</h1>
          <p>Public Preview v0.1.0 · 本地识别疑似变道未观察到转向灯片段，最终结论由人工确认。</p>
        </div>
        <HealthBadge health={health} />
      </header>

      {error && (
        <div className="alert error">
          <AlertCircle size={18} />
          <span>{error}</span>
        </div>
      )}

      {health?.messages.map((message) => (
        <div className="alert warning" key={message}>
          <AlertCircle size={18} />
          <span>{message}</span>
        </div>
      ))}

      <section className="volumes">
        <div className="sectionTitle">{volumes.scanning ? "已发现卷 · 导入中" : "已发现卷"}</div>
        {volumes.volumes.length === 0 && (
          <div className="empty">
            {volumes.scanning ? "正在扫描已挂载的卷…" : "还没有发现 Tesla 片段。可以在下面手填路径。"}
          </div>
        )}
        {volumes.volumes.map((volume) => (
          <div className="volumeRow" key={volume.folder_path}>
            <div>
              <strong>{volume.folder_path}</strong>
              <span>
                {volume.clip_count} 个片段 · {volume.message}
              </span>
            </div>
            <button
              onClick={() => void loadFolder(volume.folder_path)}
              disabled={busy || volume.status === "importing"}
            >
              {volume.status === "importing" ? "导入中" : "载入"}
            </button>
          </div>
        ))}
      </section>

      <section className="toolbar">
        <label className="pathInput">
          <span>Tesla Dashcam 文件夹路径</span>
          <input
            value={folderPath}
            onChange={(event) => setFolderPath(event.target.value)}
            placeholder="/Volumes/TESLADRIVE/TeslaCam/SavedClips/..."
          />
        </label>
        <button onClick={handleImport} disabled={busy || !folderPath.trim()}>
          <FileVideo size={18} />
          导入
        </button>
        <button onClick={handleAnalyze} disabled={busy || !importResult || job?.status === "running"}>
          <Play size={18} />
          开始分析
        </button>
        <button onClick={() => refreshEvents()} disabled={!importResult}>
          <RefreshCw size={18} />
          刷新事件
        </button>
      </section>

      <section className="summary">
        <Metric label="分段" value={importResult?.segments.length ?? 0} />
        <Metric label="疑似事件" value={events.length} />
        <Metric label="已确认" value={events.filter((event) => event.review_status === "confirmed").length} />
        <Metric label="数据目录" value={health?.data_dir || "-"} wide />
      </section>

      {job && (
        <section className="job">
          <div className="jobHeader">
            <strong>{job.status}</strong>
            <span>{job.message}</span>
          </div>
          <div className="progress">
            <div style={{ width: `${Math.round(job.progress * 100)}%` }} />
          </div>
        </section>
      )}

      <section className="workspace">
        <EventList events={events} selectedId={selected?.id || null} onSelect={setSelectedId} />
        <ReviewPanel
          event={selected}
          onReview={handleReview}
          onExport={handleExport}
          exportResult={exportResult}
        />
      </section>
    </main>
  );
}

function HealthBadge({ health }: { health: Health | null }) {
  if (!health) return <span className="badge muted">检查中</span>;
  if (health.ok && health.yolo_available && health.ffmpeg_available) {
    return (
      <span className="badge ok">
        <CheckCircle2 size={16} />
        就绪
      </span>
    );
  }
  return (
    <span className="badge warn">
      <AlertCircle size={16} />
      需处理依赖
    </span>
  );
}

function Metric({ label, value, wide = false }: { label: string; value: number | string; wide?: boolean }) {
  return (
    <div className={wide ? "metric wide" : "metric"}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function EventList({
  events,
  selectedId,
  onSelect
}: {
  events: EventItem[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <aside className="eventList">
      <div className="sectionTitle">疑似事件</div>
      {events.length === 0 && <div className="empty">还没有事件。导入文件夹后开始分析。</div>}
      {events.map((event) => (
        <button
          className={event.id === selectedId ? "eventRow active" : "eventRow"}
          key={event.id}
          onClick={() => onSelect(event.id)}
        >
          <div>
            <strong>{event.vehicle_type}</strong>
            <span>
              {event.start_s.toFixed(1)}s - {event.end_s.toFixed(1)}s · track {event.track_id}
            </span>
          </div>
          <span className={`review ${event.review_status}`}>{statusText[event.review_status]}</span>
          <small>置信度 {event.confidence.toFixed(2)}</small>
        </button>
      ))}
    </aside>
  );
}

function ReviewPanel({
  event,
  onReview,
  onExport,
  exportResult
}: {
  event: EventItem | null;
  onReview: (status: ReviewStatus, location: string, note: string, plate: string) => void;
  onExport: () => void;
  exportResult: ExportResponse | null;
}) {
  const [location, setLocation] = useState("");
  const [note, setNote] = useState("");
  const [plate, setPlate] = useState("");

  useEffect(() => {
    setLocation(event?.location || "");
    setNote(event?.note || "");
    setPlate(event?.plate || "");
  }, [event?.id]);

  if (!event) {
    return <section className="reviewPanel empty">选择一个疑似事件后复核。</section>;
  }

  const signalLabel =
    event.turn_signal_score < 0.25
      ? "未观察到稳定转向灯"
      : event.turn_signal_score >= 0.7
        ? "已观察到转向灯"
        : "转向灯不确定";

  return (
    <section className="reviewPanel">
      <div className="reviewHeader">
        <div>
          <h2>事件复核</h2>
          <p>
            {event.camera} · {event.vehicle_type} · {event.start_s.toFixed(1)}s 至{" "}
            {event.end_s.toFixed(1)}s
          </p>
        </div>
        <span className={`review ${event.review_status}`}>{statusText[event.review_status]}</span>
      </div>

      <div className="mediaGrid wide">
        <div className="mediaBox">
          <div className="sectionTitle">前视标注片段</div>
          {event.annotated_clip_url ? (
            <EventWindowVideo src={event.annotated_clip_url} startS={0} endS={event.end_s - event.start_s} />
          ) : (
            <div className="placeholder">暂无标注片段</div>
          )}
        </div>
        <div className="mediaBox">
          <div className="sectionTitle">关键帧</div>
          {event.key_frame_url ? (
            <img src={event.key_frame_url} alt="事件关键帧" />
          ) : (
            <div className="placeholder">暂无关键帧</div>
          )}
        </div>
        <div className="mediaBox">
          <div className="sectionTitle">左 repeater（对齐事件时段）</div>
          {event.left_repeater_url ? (
            <EventWindowVideo src={event.left_repeater_url} startS={event.start_s} endS={event.end_s} />
          ) : (
            <div className="placeholder">无 left_repeater 文件</div>
          )}
        </div>
        <div className="mediaBox">
          <div className="sectionTitle">右 repeater（对齐事件时段）</div>
          {event.right_repeater_url ? (
            <EventWindowVideo src={event.right_repeater_url} startS={event.start_s} endS={event.end_s} />
          ) : (
            <div className="placeholder">无 right_repeater 文件</div>
          )}
        </div>
      </div>

      <div className="scores">
        <Metric label="变道分" value={event.lane_change_score.toFixed(2)} />
        <Metric label="灯光分" value={event.turn_signal_score.toFixed(2)} />
        <Metric label="综合置信" value={event.confidence.toFixed(2)} />
        <Metric label="灯光结论" value={signalLabel} wide />
      </div>

      <div className="tags">
        {event.reason_labels
          .filter((label) => !label.includes("="))
          .map((label) => (
            <span key={label}>{label}</span>
          ))}
      </div>

      <label className="field">
        <span>地点</span>
        <input value={location} onChange={(evt) => setLocation(evt.target.value)} placeholder="道路、方向、附近参照物" />
      </label>
      <label className="field">
        <span>号牌</span>
        <input value={plate} onChange={(evt) => setPlate(evt.target.value)} placeholder="选填，人工填写" />
      </label>
      <label className="field">
        <span>备注</span>
        <textarea value={note} onChange={(evt) => setNote(evt.target.value)} placeholder="车道、现场情况" />
      </label>

      <div className="actions">
        <button onClick={() => onReview("confirmed", location, note, plate)}>
          <CheckCircle2 size={18} />
          确认
        </button>
        <button onClick={() => onReview("pending", location, note, plate)}>待定</button>
        <button className="secondary" onClick={() => onReview("dismissed", location, note, plate)}>
          <XCircle size={18} />
          排除
        </button>
        <button className="export" onClick={onExport} disabled={event.review_status !== "confirmed"}>
          <Download size={18} />
          导出证据包
        </button>
      </div>

      {exportResult && (
        <div className="handoff">
          <a className="download" href={exportResult.download_url}>
            下载 {exportResult.zip_path.split("/").pop()}
            {exportResult.submission_message ? ` · ${exportResult.submission_message}` : ""}
          </a>
          <p className="handoffPath">本机证据包：{exportResult.zip_path}</p>
          {exportResult.report_url ? (
            <p className="handoffHint">
              已在浏览器打开举报入口。请在该网站自己的页面上传证据包并完成验证码与登录。
            </p>
          ) : null}
        </div>
      )}
    </section>
  );
}

function EventWindowVideo({ src, startS, endS }: { src: string; startS: number; endS: number }) {
  const ref = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const video = ref.current;
    if (!video) return;
    const seekStart = () => {
      if (video.currentTime < startS - 0.05) {
        video.currentTime = startS;
      }
    };
    const clampEnd = () => {
      if (video.currentTime >= endS) {
        video.pause();
        video.currentTime = endS;
      }
    };
    video.addEventListener("loadedmetadata", seekStart);
    video.addEventListener("play", seekStart);
    video.addEventListener("timeupdate", clampEnd);
    return () => {
      video.removeEventListener("loadedmetadata", seekStart);
      video.removeEventListener("play", seekStart);
      video.removeEventListener("timeupdate", clampEnd);
    };
  }, [src, startS, endS]);

  return <video ref={ref} src={src} controls preload="metadata" />;
}
