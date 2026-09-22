import { AlertCircle, CheckCircle2, Download, FileVideo, Play, RefreshCw, XCircle } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { api } from "./api";
import type { EventItem, ExportResponse, Health, ImportResponse, Job, ReviewStatus } from "./types";

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

  const selected = useMemo(
    () => events.find((event) => event.id === selectedId) || events[0] || null,
    [events, selectedId]
  );

  useEffect(() => {
    api.health().then(setHealth).catch((err) => setError(err.message));
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

  async function handleImport() {
    setError("");
    setBusy(true);
    setJob(null);
    try {
      const result = await api.importFolder(folderPath.trim());
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

  async function handleReview(status: ReviewStatus, location: string, note: string) {
    if (!selected) return;
    setError("");
    try {
      const updated = await api.review(selected.id, status, location, note);
      setEvents((items) => items.map((item) => (item.id === updated.id ? updated : item)));
      setSelectedId(updated.id);
      setExportResult(null);
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
  onReview: (status: ReviewStatus, location: string, note: string) => void;
  onExport: () => void;
  exportResult: ExportResponse | null;
}) {
  const [location, setLocation] = useState("");
  const [note, setNote] = useState("");

  useEffect(() => {
    setLocation(event?.location || "");
    setNote(event?.note || "");
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

      <div className="mediaGrid">
        <div className="mediaBox">
          <div className="sectionTitle">标注片段</div>
          {event.annotated_clip_url ? (
            <video src={event.annotated_clip_url} controls />
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
        <span>备注</span>
        <textarea value={note} onChange={(evt) => setNote(evt.target.value)} placeholder="车牌、车道、现场情况" />
      </label>

      <div className="actions">
        <button onClick={() => onReview("confirmed", location, note)}>
          <CheckCircle2 size={18} />
          确认
        </button>
        <button onClick={() => onReview("pending", location, note)}>待定</button>
        <button className="secondary" onClick={() => onReview("dismissed", location, note)}>
          <XCircle size={18} />
          排除
        </button>
        <button className="export" onClick={onExport} disabled={event.review_status !== "confirmed"}>
          <Download size={18} />
          导出证据包
        </button>
      </div>

      {exportResult && (
        <a className="download" href={exportResult.download_url}>
          下载 {exportResult.zip_path.split("/").pop()}
        </a>
      )}
    </section>
  );
}
