import { useState } from 'react';
import { HardHat, Fingerprint, ArrowRight, Loader2 } from 'lucide-react';
import './Login.css';

export default function Login({ onLogin }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [isAuthenticating, setIsAuthenticating] = useState(false);

  const handleLogin = (e) => {
    e.preventDefault();
    setIsAuthenticating(true);
    // Simulate network authentication delay
    setTimeout(() => {
      setIsAuthenticating(false);
      onLogin();
    }, 1500);
  };

  return (
    <div className="login-container">
      <div className="login-mesh-bg"></div>

      <div className="login-card glass-panel fade-in-up">
        <div className="login-header">
          <div className="login-logo pulse-glow-subtle">
            <HardHat size={36} weight="fill" color="white" />
          </div>
          <h1>水熊虫不动脑</h1>
          <p>矿区视觉识别系统终端</p>
        </div>

        <form onSubmit={handleLogin} className="login-form">
          <div className="form-group stagger-1 fade-in-up">
            <label>工号 / 用户名</label>
            <input
              type="text"
              placeholder="请输入管理员账号"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
              disabled={isAuthenticating}
            />
          </div>

          <div className="form-group stagger-2 fade-in-up">
            <label>安全密钥</label>
            <input
              type="password"
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              disabled={isAuthenticating}
            />
          </div>

          <div className="form-options stagger-3 fade-in-up">
            <label className="checkbox-label">
              <input type="checkbox" />
              <span>记住本设备</span>
            </label>
            <a href="#" className="forgot-link">忘记密钥?</a>
          </div>

          <button
            type="submit"
            className="login-btn stagger-4 fade-in-up"
            disabled={isAuthenticating || !username || !password}
          >
            {isAuthenticating ? (
              <>
                <Loader2 size={20} className="spin" />
                正在验证身份...
              </>
            ) : (
              <>
                <Fingerprint size={20} />
                安全登录
                <ArrowRight size={18} className="arrow-icon" />
              </>
            )}
          </button>
        </form>

        <div className="login-footer stagger-5 fade-in-up">
          <p>系统已接入 1号矿井 安全网络</p>
          <div className="secure-badge">
            <div className="dot safe"></div>
            <span>加密信道已就绪</span>
          </div>
        </div>
      </div>
    </div>
  );
}
