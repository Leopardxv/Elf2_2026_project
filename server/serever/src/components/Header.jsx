import { Bell, Network, Search, User, Wifi } from 'lucide-react';
import './Header.css';

export default function Header({ boardIp, onBoardIpChange }) {
  return (
    <header className="app-header glass-panel">
      <div className="header-search">
        <Search size={18} className="search-icon" />
        <input type="text" placeholder="搜索设备或日志..." />
      </div>

      <div className="header-actions">
        <label className="board-address" title="知识库和模型的发布目标">
          <Network size={15} />
          <input
            value={boardIp}
            onChange={(event) => onBoardIpChange(event.target.value)}
            placeholder="板子 IP 地址"
            inputMode="decimal"
            aria-label="板子 IP 地址"
          />
        </label>
        <div className="elf2-status pulse-glow">
          <Wifi size={16} />
          <span>{boardIp ? `ELF2: ${boardIp}` : 'ELF2: 待配置'}</span>
        </div>
        <button className="icon-btn">
          <Bell size={20} />
          <span className="badge"></span>
        </button>
        <div className="user-profile">
          <div className="avatar">
            <User size={18} />
          </div>
          <span>矿区控制中心</span>
        </div>
      </div>
    </header>
  );
}
