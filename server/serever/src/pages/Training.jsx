import { useEffect, useRef, useState } from 'react';
import { CheckCircle2, CloudUpload, Cpu, FileArchive, FolderUp, Loader2, Radio, Send, ShieldCheck } from 'lucide-react';
import './Training.css';

const api = async (url, options = {}) => {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || '请求未完成');
  return body;
};

const formatDuration = (seconds) => {
  if (!Number.isFinite(seconds) || seconds < 0) return '--';
  const value = Math.round(seconds);
  if (value < 60) return `${value}s`;
  const minutes = Math.floor(value / 60);
  return `${minutes}m ${value % 60}s`;
};

const stageLabel = (stage) => ({
  waiting_gpu: '等待空闲 GPU',
  training: '模型训练中',
  exporting: '导出 ONNX',
  quantizing: '转换 RKNN',
  completed: '训练完成',
})[stage] || '准备中';

export default function Training({ boardIp }) {
  const folderRef = useRef(null);
  const archiveRef = useRef(null);
  const [files, setFiles] = useState([]);
  const [dataset, setDataset] = useState(null);
  const [epochs, setEpochs] = useState(80);
  const [imageSize, setImageSize] = useState(640);
  const [job, setJob] = useState(null);
  const [message, setMessage] = useState('选择包含 images、labels 和 data.yaml 的 YOLO 数据集文件夹，或选择一个 ZIP 包。');
  const [busy, setBusy] = useState(false);
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!job || !['queued', 'running'].includes(job.status)) return undefined;
    const timer = setInterval(async () => {
      try { setJob(await api(`/api/yolo/jobs/${job.id}`)); } catch (error) { setMessage(error.message); }
    }, 1800);
    return () => clearInterval(timer);
  }, [job]);

  const selectFiles = (selected) => {
    const next = Array.from(selected || []);
    setFiles(next);
    setDataset(null);
    setJob(null);
    setMessage(next.length ? `已选择 ${next.length} 个文件，准备上传并校验目录结构。` : '未选择数据集文件。');
  };

  const uploadDataset = async () => {
    if (!files.length) return setMessage('请先选择 YOLO 数据集文件夹或 ZIP 包。');
    setBusy(true);
    try {
      const form = new FormData();
      files.forEach((file) => form.append('files', file, file.webkitRelativePath || file.name));
      const result = await api('/api/yolo/datasets', { method: 'POST', body: form });
      setDataset(result.dataset);
      setFiles([]);
      if (folderRef.current) folderRef.current.value = '';
      if (archiveRef.current) archiveRef.current.value = '';
      setMessage(`数据集已上传：${result.dataset.file_count} 个文件，${result.dataset.data_yaml || '未找到 data.yaml'}`);
    } catch (error) { setMessage(error.message); }
    finally { setBusy(false); }
  };

  const startTraining = async () => {
    if (!dataset) return setMessage('请先上传并校验数据集。');
    setBusy(true);
    try {
      const result = await api('/api/yolo/jobs', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_id: dataset.id, epochs: Number(epochs), image_size: Number(imageSize), model: 'yolov5s.pt' }),
      });
      setJob(result.job);
      setMessage('训练任务已排队。日志会在下方持续更新。');
    } catch (error) { setMessage(error.message); }
    finally { setBusy(false); }
  };

  const publish = async () => {
    if (!job) return;
    setBusy(true);
    try {
      await api(`/api/yolo/jobs/${job.id}/publish`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ board_host: boardIp.trim() }) });
      setMessage('RKNN 模型已传输到板子。');
    } catch (error) { setMessage(error.message); }
    finally { setBusy(false); }
  };

  const progress = job?.progress;
  const elapsed = job?.started_at ? Math.max(0, now / 1000 - job.started_at) : 0;
  const remaining = progress?.stage === 'training' && progress.current_epoch > 0 && progress.current_epoch < progress.total_epochs
    ? elapsed / progress.current_epoch * (progress.total_epochs - progress.current_epoch)
    : null;

  return (
    <div className="training-page fade-in-up">
      <div className="page-header fade-in-up stagger-1">
        <div>
          <h1 className="page-title">YOLO 训练与模型下发</h1>
          <p className="page-subtitle">在服务器上训练，只有 RKNN 成品才会发布到 ELF2 的 NPU 视觉模块。</p>
        </div>
        <div className="training-chip"><Cpu size={16} /> Server training</div>
      </div>

      <section className="pipeline-grid fade-in-up stagger-2">
        <article className={`pipeline-step ${dataset ? 'done' : ''}`}><FolderUp size={20} /><div><strong>1. 数据集</strong><span>images、labels、data.yaml</span></div></article>
        <article className={`pipeline-step ${job?.status === 'completed' ? 'done' : ''}`}><Cpu size={20} /><div><strong>2. 训练</strong><span>YOLOv5 兼容权重</span></div></article>
        <article className={`pipeline-step ${job?.artifact_rknn ? 'done' : ''}`}><ShieldCheck size={20} /><div><strong>3. RKNN</strong><span>板端 NPU 格式</span></div></article>
        <article className="pipeline-step"><Send size={20} /><div><strong>4. 下发</strong><span>替换板端模型</span></div></article>
      </section>

      <section className="training-workspace card fade-in-up stagger-3">
        <div className="workspace-heading"><div><h2>训练数据集</h2><p>选择整个数据集文件夹，不上传原始视频或无标注图片。</p></div></div>
        <div className={`dataset-drop ${files.length ? 'has-files' : ''}`}>
          <input ref={folderRef} type="file" multiple webkitdirectory="" directory="" hidden onChange={(event) => selectFiles(event.target.files)} />
          <input ref={archiveRef} type="file" accept=".zip,application/zip" hidden onChange={(event) => selectFiles(event.target.files)} />
          <CloudUpload size={30} />
          <strong>{dataset ? `数据集已校验：${dataset.file_count} 个文件` : files.length ? `${files.length} 个文件待上传` : '选择 YOLO 数据集文件夹'}</strong>
          <span>{dataset ? `已识别 ${dataset.data_yaml || 'data.yaml'}` : '必须含有 data.yaml 及 images / labels 目录，也可选择 ZIP 包'}</span>
          <div className="dataset-select-actions">
            <button type="button" className="secondary-action" onClick={() => folderRef.current?.click()}><FolderUp size={17} /> 选择文件夹</button>
            <button type="button" className="secondary-action" onClick={() => archiveRef.current?.click()}><FileArchive size={17} /> 选择 ZIP 包</button>
          </div>
        </div>
        <div className="training-controls">
          <label>训练轮次<input type="number" min="1" max="1000" value={epochs} onChange={(event) => setEpochs(event.target.value)} /></label>
          <label>输入尺寸<select value={imageSize} onChange={(event) => setImageSize(event.target.value)}><option value="640">640 px</option><option value="960">960 px</option></select></label>
          <button className="secondary-action" onClick={uploadDataset} disabled={busy || !files.length}>{busy ? <Loader2 className="spin" size={18} /> : <CloudUpload size={18} />} 上传并校验</button>
          <button className="submit-btn" onClick={startTraining} disabled={busy || !dataset}>{busy ? <Loader2 className="spin" size={18} /> : <Cpu size={18} />} 开始训练</button>
        </div>
      </section>

      <section className="job-console card fade-in-up stagger-4">
        <div className="console-header"><div><Radio size={16} /><span>任务状态</span></div><span className={`job-state ${job?.status || 'idle'}`}>{job?.status || '等待数据集'}</span></div>
        <p className="job-message">{message}</p>
        {job && <div className="job-progress" aria-live="polite">
          <div className="progress-summary"><strong>{stageLabel(progress?.stage)}</strong><span>{progress ? `${progress.current_epoch} / ${progress.total_epochs} 轮` : '等待服务器更新'}</span></div>
          <div className="progress-track" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow={progress?.percent || 0}><div style={{ width: `${progress?.percent || 0}%` }} /></div>
          <div className="progress-timing"><span>已用 {formatDuration(elapsed)}</span><span>{remaining === null ? '预计剩余 --' : `预计剩余 ${formatDuration(remaining)}`}</span></div>
        </div>}
        <div className="terminal-window">
          {(job?.logs || ['[System] 等待训练任务。']).slice(-12).map((line, index) => <div className="log-line" key={`${line}-${index}`}>{line}</div>)}
        </div>
        <div className="publish-row">
          <span>{job?.artifact_rknn ? '已生成 RKNN 成品，可安全下发。' : '板端当前使用 RKNN；.pt 权重不会被错误发布。'}</span>
          <button className="secondary-action" onClick={publish} disabled={busy || !job?.artifact_rknn}><Send size={17} /> 下发 RKNN 到板子</button>
        </div>
      </section>
    </div>
  );
}
