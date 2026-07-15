import { useEffect, useRef, useState } from 'react';
import { BookOpenCheck, CloudUpload, FileText, Loader2, Send, Trash2 } from 'lucide-react';
import './Knowledge.css';

const api = async (url, options = {}) => {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || '请求未完成');
  return body;
};

export default function Knowledge({ boardIp }) {
  const fileRef = useRef(null);
  const [entries, setEntries] = useState([]);
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [file, setFile] = useState(null);
  const [restart, setRestart] = useState(true);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('新内容会先保存到服务器，再由你确认发布到板子。');

  const loadEntries = async () => {
    try { setEntries((await api('/api/knowledge')).entries || []); } catch (error) { setMessage(error.message); }
  };
  useEffect(() => { loadEntries(); }, []);

  const createEntry = async (event) => {
    event.preventDefault();
    if (!title.trim() || (!content.trim() && !file)) return setMessage('请填写标题，并输入内容或选择一个文本文件。');
    setBusy(true);
    try {
      const form = new FormData();
      form.append('title', title.trim());
      form.append('content', content);
      if (file) form.append('file', file);
      const result = await api('/api/knowledge', { method: 'POST', body: form });
      setEntries((current) => [result.entry, ...current]);
      setTitle(''); setContent(''); setFile(null);
      setMessage(`“${result.entry.title}”已保存到服务器知识库。`);
    } catch (error) { setMessage(error.message); }
    finally { setBusy(false); }
  };

  const publish = async (entry) => {
    if (!boardIp.trim()) return setMessage('请先在右上角填写板子 IP 地址。');
    setBusy(true);
    try {
      await api(`/api/knowledge/${entry.id}/publish`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ board_host: boardIp.trim(), restart_service: restart }) });
      setMessage(`“${entry.title}”已传输到 ${boardIp}。`);
    } catch (error) { setMessage(error.message); }
    finally { setBusy(false); }
  };

  return (
    <div className="knowledge-page fade-in-up">
      <div className="page-header fade-in-up stagger-1">
        <div><h1 className="page-title">专用知识库</h1><p className="page-subtitle">为板端语音助手添加经过确认的领域资料，并按条目发布。</p></div>
        <div className="knowledge-chip"><BookOpenCheck size={16} /> Server → ELF2</div>
      </div>

      <section className="knowledge-layout">
        <form className="knowledge-editor card fade-in-up stagger-2" onSubmit={createEntry}>
          <div className="section-heading"><FileText size={19} /><div><h2>新建资料</h2><p>支持直接编写，或导入 `.md` / `.txt` 文件。</p></div></div>
          <label>资料标题<input value={title} onChange={(event) => setTitle(event.target.value)} placeholder="例如：矿井一氧化碳应急处置" maxLength="80" /></label>
          <label>正文内容<textarea value={content} onChange={(event) => setContent(event.target.value)} placeholder="输入板端回答时需要参考的专业知识…" rows="10" /></label>
          <div className="knowledge-actions">
            <button type="button" className="file-select" onClick={() => fileRef.current?.click()}><CloudUpload size={17} /> {file ? file.name : '导入文本文件'}</button>
            <input ref={fileRef} type="file" hidden accept=".md,.txt,text/markdown,text/plain" onChange={(event) => setFile(event.target.files?.[0] || null)} />
            <button className="submit-btn" disabled={busy}>{busy ? <Loader2 className="spin" size={17} /> : <FileText size={17} />} 保存到服务器</button>
          </div>
        </form>

        <aside className="publish-policy card fade-in-up stagger-3">
          <h2>发布策略</h2>
          <p>发布前会使用右上角的板子 IP。服务器不会把 SSH 密码写入资料库。</p>
          <label className="toggle-line"><input type="checkbox" checked={restart} onChange={(event) => setRestart(event.target.checked)} /><span>发布后重启板端语音服务</span></label>
          <div className="policy-note">重启仅在板端已配置 SSH 密钥时执行。未配置时页面会说明原因，不会伪造“已发布”。</div>
        </aside>
      </section>

      <section className="knowledge-list card fade-in-up stagger-4">
        <div className="list-header"><div><h2>服务器资料</h2><p>{message}</p></div><span>{entries.length} 条</span></div>
        {entries.length ? entries.map((entry) => <article className="knowledge-entry" key={entry.id}>
          <div className="entry-copy"><strong>{entry.title}</strong><span>{entry.filename} · {new Date(entry.created_at).toLocaleString()}</span></div>
          <button className="secondary-action" disabled={busy} onClick={() => publish(entry)}><Send size={16} /> 发布到板子</button>
        </article>) : <div className="empty-state">尚无资料。先添加一条可被语音助手直接参考的专业内容。</div>}
      </section>
    </div>
  );
}
