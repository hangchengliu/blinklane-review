# BlinkLane Code Review 报告（2026-10-17）

**总体评价**：项目工程质量在同类个人项目中属于上乘——SQL 全部参数化、subprocess 用 list 形式传参、测试无 sleep/固定端口、前后端 API 契约基本一致、代码结构清晰。但存在 1 个核心流程 bug（确认事件后 UI 崩坏）、3 个与"纯本地、不出数据"承诺相矛盾的安全问题，以及若干会导致用户评审数据丢失的设计缺陷。

本分支（`code-review-fixes`）已修复下述全部问题；每条目注明了修复方式。

---

## 🔴 Critical

### 1. 确认事件后 UI 列表不更新、选中项跳走——后端响应覆盖了事件 `id`

前端提交复核后用响应里的 `id` 更新列表和选中态，但后端在 `confirmed` 时把导出记录的 `id` 直接 `update` 进了响应，把事件 id 覆盖成了 export id（`backend/app/main.py`）。

后果：(a) 列表按 `updated.id` 找不到事件，状态仍显示"待定"；(b) `setSelectedId(updated.id)` 指向不存在的 id，面板跳到另一个事件；(c) 表单重置 effect 触发，用户刚填的位置/备注被清掉。

**修复**：后端把导出结果嵌套到响应的 `export` 字段（不再平铺覆盖事件字段）；前端按事件 `id` 更新列表，从 `export.report_url` 读取交接信息。

---

## 🟠 High

### 2. `/api/media/source` 可被任何本地进程/网页读取磁盘上任意 MP4

该端点原先只校验后缀是 `.mp4`，没有路径白名单或鉴权。配合 DNS rebinding，任意网页都能枚举并外泄本机视频。

**修复**：仅允许提供 `video_segments` / `events` 相关表中登记过的路径，其余一律 403。

### 3. 用户可控的 `model_name` 直接传入 `YOLO()`

`AnalyzeRequest.model_name` 原先未做校验，可指向任意 URL/路径的 `.pt` 文件（pickle 反序列化 = 远程代码执行 + 网络外发）。

**修复**：只允许 `yolo*.pt` 形式的纯文件名（正则白名单，禁止路径分隔符与 URL），非法值返回 422。

### 4. Ultralytics 默认遥测（`sync=True`）未关闭

README 宣称"默认不上传"，但代码从未禁用 Ultralytics 的匿名使用分析。

**修复**：加载模型前显式关闭 Ultralytics analytics sync，并在隐私文档中列明全部网络出口。

### 5. 重新分析会清空该 session 全部人工复核结果

`replace_segment_events` 原先直接 `DELETE FROM events`，事件上的复核状态、位置、备注、审计记录、导出记录全部级联删除。

**修复**：已被人工复核（confirmed/dismissed）的事件予以保留，仅替换仍处于 pending 的事件。

### 6. 文件夹重同步会级联删除已消失片段的全部数据

Tesla 行车记录仪会自动轮换删除旧录像；原先重同步发现片段不在磁盘就删除，级联删掉事件、复核、导出记录，且与运行中的分析任务存在 TOCTOU 竞争。

**修复**：片段从磁盘消失时仅在没有任何关联事件时才删除；有关联事件的片段保留记录；空片段列表不再触发全量清除。

### 7. 轨迹横向位置混用两套坐标系，会凭空制造/抹掉"变道"

`_lateral_positions` 中：有车道线时用车道相对坐标（以 0 为中心），没有时用绝对坐标（0.3–0.7），混进同一列表。车道线检测时有时无时，直行车会出现约 0.4 的幻象"位移"（阈值 0.12）。

**修复**：车道几何部分可用时，仅使用有车道中心的采样点（同一参考系内比较）；不足时回退为全部绝对坐标。

### 8. 任务轮询 effect 无取消保护，旧任务可"复活"覆盖新会话

切换文件夹时已发出的 `api.job()` 请求仍会后到并把旧任务装回来，随后用旧会话的事件覆盖新列表；且 effect 依赖整个 `job` 对象导致每 1.2 秒重建定时器。

**修复**：加 `stop` 取消标志、依赖收敛为 `job?.id`/`job?.status`、修复闭包过期问题。

---

## 🟡 Medium

| # | 位置 | 问题 | 修复 |
|---|------|------|------|
| 9 | `main.py` + `App.tsx` | 无并发分析防护，可造出竞争同一 session 的两个任务 | 后端：同 session 已有 queued/running 任务时返回 409；前端：`queued` 状态同样禁用按钮 |
| 10 | `main.py` | 导出在复核写事务内执行，导出失败回滚用户复核且 zip 成孤儿 | 复核先行提交，导出在事务外执行，失败不丢失复核结果 |
| 11 | `signals.py` | 常亮琥珀色被满分置信判为"未打转向灯" | 高亮度低波动序列改判 `inconclusive` |
| 12 | `pipeline.py` | repeater 采样网格取自任意第一条轨迹，短轨迹会禁用所有轨迹的侧方证据 | 使用全部轨迹时间戳的并集驱动 repeater 读取 |
| 13 | `trajectory.py` | 转向灯评分窗口（中点 ±2.5s）与事件窗口脱钩 | 评分窗口至少覆盖完整事件窗口 |
| 14 | `pipeline.py` | 侧方摄像头能量无差别合并进每条轨迹，无关车辆打灯可压掉真实事件 | 侧方证据仅作辅助加权，不再单独将事件判为"已打灯" |
| 15 | `signals.py` | NaN/inf bbox 导致整个分析任务崩溃 | 构建 `TrackSample` 时过滤非有限值 |
| 16 | `main.py` | session 查-再-插非原子，并发导入 500 | 改用 `INSERT ... ON CONFLICT DO NOTHING` 后重查 |
| 17 | `video.py` | `VideoWriter` 不检查 `isOpened()`，静默产出 0 字节视频；ffmpeg 无超时 | 打开失败即抛错告警；ffmpeg 加 `timeout=` |
| 18 | `App.tsx` | `exportResult` 不与选中事件绑定，显示错事件的下载链接 | 按事件 id  keyed 存储与渲染 |
| 19 | `App.tsx` | `await` 后 `window.open` 被浏览器拦截，举报交接失效 | 改为渲染 `<a target="_blank">` 链接由用户点击 |
| 20 | 测试 | 两个媒体端点零测试；ffmpeg 双分支从不断言 | 新增媒体端点安全测试与 ffmpeg/OpenCV 双分支断言 |

---

## 🟢 Low

- **后端**：数据目录依赖进程 CWD（启动时记录解析后的路径）；卷扫描无变化时不再重复 probe 视频、错误不再静默吞掉；`/api/health` 移除不可达的 cv2 检查与重复 ffmpeg 探测；删除不可达死代码 `merge_overlapping_events`；`probe_video` 加 `try/finally`；`discover_tesla_folders` 深度 off-by-one 修复
- **算法**：`lane.py` 补充单元测试；fps 静默回退 30 时记录告警；分离 `detect_lane_lines` 与 overlay 绘制避免每帧全分辨率拷贝；repeater 读取改为顺序读取替代逐点 seek；统一 floor 帧索引约定；关键魔数补充注释
- **前端**：`api.ts` 修复 422 响应（`detail` 数组）显示、header 合并顺序、URL 编码；`refreshEvents` 错误处理；自动导入失败可重试；`EventWindowVideo` 到尾后可重播；错误横幅加 `role="alert"`、进度条加 `role="progressbar"`；`.metric.wide` 跨列修复；`vite`/`typescript` 移入 `devDependencies`；`tsconfig` 纳入 `vite.config.ts`
- **CI/打包**：GitHub Actions 固定 SHA + `permissions: contents: read`；`setuptools>=77.0.3` 匹配 PEP 639 license 声明；`package.json` 补 `license` 字段；e2e 测试清除 `BLINKLANE_REPORT_URL` 环境变量；移除重复的 smoke 测试步骤
- **文档**：SECURITY.md 补充威胁模型（无鉴权、loopback 约定、媒体端点白名单）；隐私文档列明网络出口（权重下载、遥测已禁用）；`docs/architecture.md` 环境变量表补 `BLINKLANE_REPORT_URL`

---

*本报告由自动化 code review 生成，修复内容见本分支全部 commit。*
