import { useState, useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { Line } from 'react-chartjs-2';
import { AlertTriangle, Thermometer, Wind, CloudFog, Camera, Wifi, Maximize2, X } from 'lucide-react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Filler,
  Legend,
} from 'chart.js';
import './Monitor.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Filler,
  Legend
);

export default function Monitor() {
  const [isAlert, setIsAlert] = useState(false);
  const [isChartExpanded, setIsChartExpanded] = useState(false);
  const [dataPoints, setDataPoints] = useState(Array.from({ length: 20 }, () => Math.random() * 2 + 0.5));

  const counterRef = useRef(0);

  // Simulate real-time data with specific pattern: 15 normal, 5 abnormal
  useEffect(() => {
    const interval = setInterval(() => {
      counterRef.current = (counterRef.current + 1) % 20;
      const isCurrentlyAlert = counterRef.current >= 15; // 0-14 normal; 15-19 alert

      setIsAlert(isCurrentlyAlert);

      setDataPoints(prev => {
        // Normal value: 0.5 ~ 2.5
        // Alert value: 6.0 ~ 9.0 (spikes up)
        const baseValue = isCurrentlyAlert ? 6.0 : 0.5;
        const randomFactor = isCurrentlyAlert ? Math.random() * 3 : Math.random() * 2;
        const newData = [...prev.slice(1), baseValue + randomFactor];
        return newData;
      });
    }, 1500); // 1.5 seconds per tick for a slightly faster, smoother demonstration
    return () => clearInterval(interval);
  }, []);

  const chartData = {
    labels: Array.from({ length: 20 }, (_, i) => `${i}s`),
    datasets: [
      {
        fill: true,
        label: '有害气体综合浓度',
        data: dataPoints,
        borderColor: isAlert ? '#c47667' : '#7d9685',
        backgroundColor: isAlert ? 'rgba(196, 118, 103, 0.2)' : 'rgba(125, 150, 133, 0.2)',
        tension: 0.4,
        pointRadius: 0,
        borderWidth: 2,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: { enabled: false },
    },
    scales: {
      y: { display: false, min: 0, max: 10 },
      x: { display: false },
    },
    animation: { duration: 0 },
  };

  return (
    <div className={`monitor-page fade-in-up ${isAlert ? 'alert-active' : ''}`}>
      <div className="page-header fade-in-up stagger-1">
        <div>
          <h1 className="page-title">1号矿井 - 巡检监控</h1>
          <p className="page-subtitle">实时环境数据与视觉监控</p>
        </div>
      </div>

      <div className="monitor-grid fade-in-up stagger-2">


        <div className="metrics-section">
          <div className="metric-card glass-panel">
            <div className="metric-icon"><CloudFog size={24} /></div>
            <div className="metric-info">
              <span className="metric-label">有害气体综合浓度</span>
              <div className="metric-value">
                <span className={`value ${isAlert ? 'danger' : ''}`}>
                  {dataPoints[dataPoints.length - 1].toFixed(2)}
                </span>
                <span className="unit">idx</span>
              </div>
            </div>
          </div>



          <div className="metric-card glass-panel">
            <div className="metric-icon"><Thermometer size={24} /></div>
            <div className="metric-info">
              <span className="metric-label">环境温度</span>
              <div className="metric-value">
                <span className="value">24.5</span>
                <span className="unit">°C</span>
              </div>
            </div>
          </div>
        </div>

        <div className="chart-section card fade-in-up stagger-5">
          <div className="chart-header">
            <h3>有害气体实时趋势</h3>
            <div className="chart-actions">
              <span className="status-indicator">
                <span className={`dot ${isAlert ? 'danger' : 'safe'}`}></span>
                {isAlert ? '浓度异常' : '正常'}
              </span>
              <button
                className="icon-btn expand-btn"
                onClick={() => setIsChartExpanded(true)}
                title="放大观察"
              >
                <Maximize2 size={16} />
              </button>
            </div>
          </div>
          <div className="chart-container" onClick={() => setIsChartExpanded(true)}>
            <Line data={chartData} options={chartOptions} />
          </div>
        </div>
      </div>

      {isAlert && createPortal(
        <div className="alert-overlay">
          <div className="alert-toast">
            <div className="alert-icon-wrapper pulse-glow-red">
              <AlertTriangle size={24} color="#ff453a" />
            </div>
            <div className="toast-content">
              <h4>危险警报</h4>
              <p>检测到未知有害气体浓度异常，请立即撤离！</p>
            </div>
          </div>
        </div>,
        document.body
      )}

      {isChartExpanded && createPortal(
        <div className="chart-modal-overlay" onClick={() => setIsChartExpanded(false)}>
          <div className="chart-modal-content card fade-in-up" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div className="modal-title">
                <CloudFog size={20} className="modal-icon" />
                <h2>详细波形分析</h2>
              </div>
              <button className="icon-btn close-btn" onClick={() => setIsChartExpanded(false)}>
                <X size={20} />
              </button>
            </div>

            <div className="modal-body">
              <div className="expanded-chart-container">
                <Line
                  data={chartData}
                  options={{
                    ...chartOptions,
                    scales: {
                      y: { display: true, min: 0, max: 10, grid: { color: 'rgba(0,0,0,0.05)' } },
                      x: { display: true, grid: { display: false } },
                    },
                    plugins: {
                      legend: { display: true },
                      tooltip: { enabled: true },
                    }
                  }}
                />
              </div>
              <div className="analysis-panel">
                <h4>智能实时诊断</h4>
                <p>{isAlert ? '⚠ 当前数据波动剧烈，波峰明显超出安全阈值 (6.0 idx)，判定为异常泄漏。' : '✅ 当前波形平稳，处于正常环境噪声波动区间内，无泄漏风险。'}</p>
              </div>
            </div>
          </div>
        </div>,
        document.body
      )}
    </div>
  );
}
