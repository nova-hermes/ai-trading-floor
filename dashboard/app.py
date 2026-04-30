"""
AI Trading Floor v2 — Complete Web Dashboard
Trading, Backtesting, ML, Scanning — all in one UI
"""
import json
import sqlite3
import threading
import traceback
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template_string, jsonify, request
import sys

sys.path.insert(0, '/home/doug/ai-trading-floor')
from config.settings import DB_PATH, TRADING_CONFIG

app = Flask(__name__)

DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Trading Floor v2</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0f0f0f;color:#e0e0e0;min-height:100vh}
.container{max-width:1400px;margin:0 auto;padding:20px}
header{background:linear-gradient(135deg,#1a1a2e,#16213e);padding:20px;border-bottom:2px solid #0f3460;margin-bottom:25px}
header h1{font-size:22px;color:#00d9ff}
header p{color:#888;margin-top:4px;font-size:13px}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:20px}
.grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px}
@media(max-width:900px){.grid-2,.grid-3{grid-template-columns:1fr}}
.stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:20px}
.stat-card{background:#1a1a2e;border-radius:8px;padding:15px;border:1px solid #0f3460}
.stat-card h3{color:#888;font-size:12px;margin-bottom:6px}
.stat-card .value{font-size:22px;font-weight:bold}
.positive{color:#00ff88}.negative{color:#ff4444}.neutral{color:#00d9ff}
.section{background:#1a1a2e;border-radius:10px;padding:18px;margin-bottom:18px;border:1px solid #0f3460}
.section h2{font-size:16px;color:#00d9ff;margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid #0f3460}
table{width:100%;border-collapse:collapse}
th,td{padding:8px 10px;text-align:left;border-bottom:1px solid #0f3460;font-size:13px}
th{color:#888;font-weight:600;font-size:12px}
tr:hover{background:rgba(0,217,255,0.04)}
.badge{display:inline-block;padding:2px 7px;border-radius:3px;font-size:11px;font-weight:600}
.badge-buy{background:rgba(0,255,136,0.2);color:#00ff88}
.badge-sell{background:rgba(255,68,68,0.2);color:#ff4444}
.badge-hold{background:rgba(0,217,255,0.2);color:#00d9ff}
.btn{padding:8px 16px;border:none;border-radius:5px;cursor:pointer;font-size:13px;font-weight:600;transition:all .2s}
.btn:hover{transform:translateY(-1px)}.btn:disabled{opacity:.5;cursor:not-allowed;transform:none}
.btn-primary{background:#0f3460;color:#00d9ff}
.btn-buy{background:rgba(0,255,136,0.15);color:#00ff88;border:1px solid #00ff88}
.btn-sell{background:rgba(255,68,68,0.15);color:#ff4444;border:1px solid #ff4444}
.btn-scan{background:rgba(255,200,0,0.15);color:#ffc800;border:1px solid #ffc800}
.btn-ml{background:rgba(160,100,255,0.15);color:#a064ff;border:1px solid #a064ff}
.btn-backtest{background:rgba(0,200,200,0.15);color:#00c8c8;border:1px solid #00c8c8}
.input-group{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
input,select{background:#0a0a1a;border:1px solid #0f3460;color:#e0e0e0;padding:8px 12px;border-radius:5px;font-size:13px}
input:focus,select:focus{outline:none;border-color:#00d9ff}
input[type=text]{width:160px}
.quick-symbols{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
.quick-sym{background:#0a0a1a;border:1px solid #0f3460;color:#e0e0e0;padding:4px 10px;border-radius:4px;cursor:pointer;font-size:11px}
.quick-sym:hover{border-color:#00d9ff;color:#00d9ff}
.panel{background:#0a0a1a;border:1px solid #0f3460;border-radius:8px;padding:15px;margin-top:12px;display:none}
.panel.active{display:block}
.panel h3{color:#00d9ff;margin-bottom:10px;font-size:15px}
.agent-row{display:flex;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid #0f3460;font-size:13px}
.agent-row:last-child{border:none}
.agent-name{width:130px;color:#888;font-size:12px}
.decision-box{margin-top:12px;padding:12px;border-radius:6px;text-align:center}
.decision-buy{background:rgba(0,255,136,0.08);border:1px solid #00ff88}
.decision-sell{background:rgba(255,68,68,0.08);border:1px solid #ff4444}
.decision-hold{background:rgba(0,217,255,0.08);border:1px solid #00d9ff}
.decision-box .action{font-size:24px;font-weight:bold}
.decision-box .details{color:#888;margin-top:5px;font-size:12px}
.execute-row{text-align:center;margin-top:12px;padding-top:12px;border-top:1px solid #0f3460}
.spinner{display:inline-block;width:14px;height:14px;border:2px solid #0f3460;border-top-color:#00d9ff;border-radius:50%;animation:spin .8s linear infinite;margin-right:6px}
@keyframes spin{to{transform:rotate(360deg)}}
.status-msg{padding:8px;border-radius:5px;margin-top:8px;font-size:12px}
.status-msg.info{background:rgba(0,217,255,0.1);color:#00d9ff}
.status-msg.success{background:rgba(0,255,136,0.1);color:#00ff88}
.status-msg.error{background:rgba(255,68,68,0.1);color:#ff4444}
.bt-row{display:flex;gap:8px;align-items:center;margin-bottom:10px;flex-wrap:wrap}
.bt-result{background:#0a0a1a;border:1px solid #0f3460;border-radius:6px;padding:12px;margin-top:10px}
.bt-metric{display:inline-block;margin-right:20px;margin-bottom:8px}
.bt-metric .label{color:#888;font-size:11px}
.bt-metric .val{font-size:16px;font-weight:bold}
footer{text-align:center;padding:15px;color:#666;font-size:11px}
</style>
</head>
<body>
<header>
<div class="container">
<h1>🏛️ AI Trading Floor v2</h1>
<p>AI-Powered Trading · Paper Trading Mode</p>
</div>
</header>
<div class="container">

<!-- Stats -->
<div class="stats-grid">
<div class="stat-card"><h3>Open Positions</h3><div class="value neutral" id="s-open">0</div></div>
<div class="stat-card"><h3>Total Trades</h3><div class="value" id="s-trades">0</div></div>
<div class="stat-card"><h3>Total P&L</h3><div class="value" id="s-pnl">$0.00</div></div>
<div class="stat-card"><h3>Win Rate</h3><div class="value" id="s-wr">0.0%</div></div>
<div class="stat-card"><h3>Best Trade</h3><div class="value positive" id="s-best">$0.00</div></div>
<div class="stat-card"><h3>Worst Trade</h3><div class="value negative" id="s-worst">$0.00</div></div>
</div>

<!-- ===== ANALYZE & TRADE ===== -->
<div class="section">
<h2>🎯 Analyze & Trade</h2>
<div class="input-group">
<input type="text" id="sym" placeholder="BTC/USDT or AAPL" value="BTC/USDT">
<select id="tf"><option value="1h">1H</option><option value="4h">4H</option><option value="1d">1D</option></select>
<button class="btn btn-primary" id="btn-analyze" onclick="doAnalyze()">🔍 Analyze</button>
<button class="btn btn-scan" id="btn-scan" onclick="doScan()">📡 Scan</button>
</div>
<div class="quick-symbols">
<span style="color:#888;font-size:11px">Quick:</span>
<button class="quick-sym" onclick="setSym('BTC/USDT')">BTC</button>
<button class="quick-sym" onclick="setSym('ETH/USDT')">ETH</button>
<button class="quick-sym" onclick="setSym('SOL/USDT')">SOL</button>
<button class="quick-sym" onclick="setSym('BNB/USDT')">BNB</button>
<button class="quick-sym" onclick="setSym('XRP/USDT')">XRP</button>
<button class="quick-sym" onclick="setSym('AAPL')">AAPL</button>
<button class="quick-sym" onclick="setSym('NVDA')">NVDA</button>
<button class="quick-sym" onclick="setSym('TSLA')">TSLA</button>
<button class="quick-sym" onclick="setSym('MSFT')">MSFT</button>
<button class="quick-sym" onclick="setSym('SPY')">SPY</button>
</div>
<div id="analysis-panel" class="panel"></div>
<div id="status-main"></div>
</div>

<!-- ===== BACKTEST ===== -->
<div class="section">
<h2>📊 Backtest</h2>
<div class="bt-row">
<input type="text" id="bt-sym" placeholder="Symbol" value="BTC/USDT" style="width:140px">
<select id="bt-strat">
<option value="rsi">RSI Mean Reversion</option>
<option value="ma_crossover">MA Crossover</option>
<option value="bollinger">Bollinger Breakout</option>
<option value="macd">MACD Crossover</option>
<option value="multi">Multi-Indicator</option>
</select>
<select id="bt-days">
<option value="30">30 days</option>
<option value="60">60 days</option>
<option value="90" selected>90 days</option>
<option value="180">180 days</option>
</select>
<button class="btn btn-backtest" id="btn-bt" onclick="doBacktest()">▶️ Run Backtest</button>
</div>
<div id="bt-result" class="bt-result" style="display:none"></div>
<div id="status-bt"></div>
</div>

<!-- ===== ML SIGNALS ===== -->
<div class="section">
<h2>🤖 ML Alpha Engine</h2>
<div class="bt-row">
<input type="text" id="ml-sym" placeholder="Symbol" value="BTC/USDT" style="width:140px">
<select id="ml-days">
<option value="90">90 days</option>
<option value="180" selected>180 days</option>
<option value="365">365 days</option>
</select>
<button class="btn btn-ml" id="btn-ml" onclick="doML()">🧠 Train & Predict</button>
</div>
<div id="ml-result" class="bt-result" style="display:none"></div>
<div id="status-ml"></div>
</div>

<div class="grid-2">
<!-- Open Positions -->
<div class="section">
<h2>📈 Open Positions</h2>
<table>
<thead><tr><th>Symbol</th><th>Side</th><th>Entry</th><th>SL</th><th>TP</th><th>P&L</th><th></th></tr></thead>
<tbody id="tbl-open"><tr><td colspan="7" style="text-align:center;color:#666">No open positions</td></tr></tbody>
</table>
</div>

<!-- Trade History -->
<div class="section">
<h2>📜 Trade History</h2>
<table>
<thead><tr><th>Time</th><th>Action</th><th>Symbol</th><th>Price</th><th>P&L</th></tr></thead>
<tbody id="tbl-hist"><tr><td colspan="5" style="text-align:center;color:#666">No trades yet</td></tr></tbody>
</table>
</div>
</div>

<!-- Controls -->
<div class="section">
<h2>⚙️ Controls</h2>
<div class="input-group">
<button class="btn btn-primary" onclick="loadPortfolio()">🔄 Refresh</button>
<button class="btn btn-sell" onclick="closeAll()">🛑 Close All</button>
<span style="color:#666;font-size:11px" id="last-upd">Last update: —</span>
</div>
</div>

</div>
<footer>AI Trading Floor v2 · Paper Trading · All 5 AI Agents Active</footer>

<script>
const $ = id => document.getElementById(id);
function setSym(s){$('sym').value=s}
function status(id,msg,type='info'){$(id).innerHTML='<div class="status-msg '+type+'">'+msg+'</div>';if(type!=='info')setTimeout(()=>$(id).innerHTML='',6000)}

// ===== PORTFOLIO =====
async function loadPortfolio(){
try{
const r=await fetch('/api/portfolio');const d=await r.json();
$('s-open').textContent=d.open_positions;
$('s-trades').textContent=d.total_trades;
const pnl=d.total_pnl;$('s-pnl').textContent='$'+pnl.toFixed(2);$('s-pnl').className='value '+(pnl>=0?'positive':'negative');
$('s-wr').textContent=d.win_rate.toFixed(1)+'%';
$('s-best').textContent='$'+(d.best_trade||0).toFixed(2);
$('s-worst').textContent='$'+(d.worst_trade||0).toFixed(2);

const ot=$('tbl-open');
if(d.open_positions_list?.length){
ot.innerHTML=d.open_positions_list.map(p=>{
const cp=p.current_price||p.entry_price;
const pnlVal=p.side==='long'?(cp-p.entry_price)*p.quantity:(p.entry_price-cp)*p.quantity;
const cls=pnlVal>=0?'positive':'negative';
return `<tr>
<td><strong>${p.symbol}</strong></td>
<td><span class="badge badge-${p.side}">${p.side.toUpperCase()}</span></td>
<td>$${p.entry_price.toLocaleString()}</td>
<td>$${p.stop_loss.toLocaleString()}</td>
<td>$${p.take_profit.toLocaleString()}</td>
<td class="${cls}">$${pnlVal.toFixed(2)}</td>
<td><button class="btn btn-sell" style="padding:3px 8px;font-size:11px" onclick="closePos('${p.id}')">✕</button></td>
</tr>`}).join('');
}else{ot.innerHTML='<tr><td colspan="7" style="text-align:center;color:#666">No open positions</td></tr>'}

const ht=$('tbl-hist');
if(d.recent_trades?.length){
ht.innerHTML=d.recent_trades.map(t=>{
const cls=t.pnl>0?'positive':t.pnl<0?'negative':'';
return `<tr>
<td style="font-size:11px">${t.timestamp}</td>
<td><span class="badge badge-${t.action.includes('BUY')?'buy':t.action.includes('SELL')?'sell':'hold'}">${t.action}</span></td>
<td>${t.symbol}</td>
<td>$${(t.price||0).toLocaleString()}</td>
<td class="${cls}">$${(t.pnl||0).toFixed(2)}</td>
</tr>`}).join('');
}else{ht.innerHTML='<tr><td colspan="5" style="text-align:center;color:#666">No trades yet</td></tr>'}

$('last-upd').textContent='Last update: '+new Date().toLocaleTimeString();
}catch(e){console.error(e)}
}

// ===== ANALYZE =====
async function doAnalyze(){
const sym=$('sym').value.trim();if(!sym){status('status-main','Enter a symbol','error');return}
const btn=$('btn-analyze');btn.disabled=true;btn.innerHTML='<span class="spinner"></span> Analyzing...';
status('status-main','Running 5 AI agents... ~15s','info');
try{
const r=await fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({symbol:sym})});
const d=await r.json();
if(d.error){status('status-main',d.error,'error')}
else{showAnalysis(d);status('status-main','Analysis complete!','success')}
}catch(e){status('status-main','Error: '+e.message,'error')}
btn.disabled=false;btn.innerHTML='🔍 Analyze';
}

function showAnalysis(d){
const p=$('analysis-panel');p.classList.add('active');
const ac={BUY:'positive',SELL:'negative',HOLD:'neutral'};
const ae={BUY:'🟢',SELL:'🔴',HOLD:'⏸️'};
const c=ac[d.action]||'neutral', e=ae[d.action]||'❓';

let votes='';
if(d.agent_votes)votes=d.agent_votes.map(v=>{
const vc=ac[v.action]||'neutral', ve=ae[v.action]||'❓';
return `<div class="agent-row">
<span class="agent-name">${v.agent_name}</span>
<span class="${vc}" style="font-weight:600">${ve} ${v.action}</span>
<span style="color:#888;font-size:11px">(${v.confidence}%)</span>
<span style="color:#555;font-size:11px;flex:1;text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${v.reasoning}</span>
</div>`}).join('');

const btnHtml=d.action!=='HOLD'?`<div class="execute-row">
<button class="btn btn-${d.action.toLowerCase()}" onclick="doExecute('${d.symbol}','${d.action}',${d.entry_price},${d.stop_loss||0},${d.take_profit||0})">
⚡ Execute ${d.action} — Paper Trade
</button>
<p style="color:#666;font-size:11px;margin-top:6px">Entry: $${d.entry_price?.toLocaleString()} · SL: $${(d.stop_loss||0).toLocaleString()} · TP: $${(d.take_profit||0).toLocaleString()} · Size: 5% of portfolio</p>
</div>`:`<div class="execute-row"><p style="color:#666;font-size:12px">⏸️ No trade — agents recommend holding</p></div>`;

p.innerHTML=`
<h3>${d.symbol} @ $${d.current_price?.toLocaleString()}</h3>
<div style="display:flex;gap:20px;margin:10px 0;font-size:13px">
<div><span style="color:#888">RSI:</span> <strong>${d.rsi||'—'}</strong></div>
<div><span style="color:#888">MACD:</span> <strong>${d.macd||'—'}</strong></div>
<div><span style="color:#888">Volume:</span> <strong>${d.volume_signal||'—'}</strong></div>
<div><span style="color:#888">TV Signal:</span> <strong>${d.tv_signal||'—'}</strong></div>
</div>
<div style="margin:10px 0"><strong style="color:#888;font-size:12px">Agent Votes:</strong></div>
${votes}
<div class="decision-box decision-${(d.action||'hold').toLowerCase()}">
<div class="action ${c}">${e} ${d.action}</div>
<div class="details">Confidence: ${d.confidence}% · Consensus: ${d.consensus_score}%</div>
</div>
${btnHtml}`;
}

// ===== EXECUTE =====
async function doExecute(sym,action,entry,sl,tp){
status('status-main','Executing '+action+' '+sym+'...','info');
try{
const r=await fetch('/api/execute',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({symbol:sym,action,entry_price:entry,stop_loss:sl,take_profit:tp})});
const d=await r.json();
if(d.error)status('status-main',d.error,'error');
else{status('status-main','✅ '+action+' '+sym+' executed! ID: '+d.position_id,'success');loadPortfolio()}
}catch(e){status('status-main','Error: '+e.message,'error')}
}

// ===== CLOSE =====
async function closePos(id){
try{const r=await fetch('/api/close-position',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({position_id:id})});
const d=await r.json();if(d.error)status('status-main',d.error,'error');else{status('status-main','Position closed','success');loadPortfolio()}}
catch(e){status('status-main','Error: '+e.message,'error')}
}
async function closeAll(){
if(!confirm('Close ALL open positions?'))return;
try{const r=await fetch('/api/close-all',{method:'POST'});const d=await r.json();status('status-main',d.message,'success');loadPortfolio()}
catch(e){status('status-main','Error: '+e.message,'error')}
}

// ===== SCAN =====
async function doScan(){
const btn=$('btn-scan');btn.disabled=true;btn.innerHTML='<span class="spinner"></span> Scanning...';
status('status-main','Scanning BTC, ETH, SOL, BNB, XRP... ~1min','info');
try{
const r=await fetch('/api/scan',{method:'POST'});const d=await r.json();
if(d.error){status('status-main',d.error,'error')}
else{
const p=$('analysis-panel');p.classList.add('active');
const ac={BUY:'positive',SELL:'negative',HOLD:'neutral'},ae={BUY:'🟢',SELL:'🔴',HOLD:'⏸️'};
let rows=d.signals.map(s=>`<tr style="cursor:pointer" onclick="setSym('${s.symbol}');doAnalyze()">
<td><strong>${s.symbol}</strong></td><td class="${ac[s.action]}">${ae[s.action]} ${s.action}</td>
<td>${s.confidence}%</td><td>$${s.current_price?.toLocaleString()}</td></tr>`).join('');
p.innerHTML=`<h3>📡 Market Scan</h3><table><thead><tr><th>Symbol</th><th>Signal</th><th>Conf</th><th>Price</th></tr></thead><tbody>${rows}</tbody></table>
<p style="color:#666;font-size:11px;margin-top:8px">Click a row to analyze in detail</p>`;
status('status-main','Scan complete! '+d.signals.length+' signals','success');
}
}catch(e){status('status-main','Error: '+e.message,'error')}
btn.disabled=false;btn.innerHTML='📡 Scan';
}

// ===== BACKTEST =====
async function doBacktest(){
const sym=$('bt-sym').value.trim(),strat=$('bt-strat').value,days=$('bt-days').value;
const btn=$('btn-bt');btn.disabled=true;btn.innerHTML='<span class="spinner"></span> Running...';
status('status-bt','Backtesting '+strat+' on '+sym+'...','info');
try{
const r=await fetch('/api/backtest',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({symbol:sym,strategy:strat,days:parseInt(days)})});
const d=await r.json();
if(d.error){status('status-bt',d.error,'error')}
else{showBacktest(d);status('status-bt','Backtest complete!','success')}
}catch(e){status('status-bt','Error: '+e.message,'error')}
btn.disabled=false;btn.innerHTML='▶️ Run Backtest';
}

function showBacktest(d){
const el=$('bt-result');el.style.display='block';
const rc=d.total_return>=0?'positive':'negative';
el.innerHTML=`
<h3 style="color:#00c8c8;margin-bottom:10px">📊 ${d.strategy_name} · ${d.symbol} · ${d.start_date} → ${d.end_date}</h3>
<div>
<div class="bt-metric"><div class="label">Return</div><div class="val ${rc}">${d.total_return}%</div></div>
<div class="bt-metric"><div class="label">Final Capital</div><div class="val">$${d.final_capital?.toLocaleString()}</div></div>
<div class="bt-metric"><div class="label">Sharpe</div><div class="val">${d.sharpe_ratio}</div></div>
<div class="bt-metric"><div class="label">Max Drawdown</div><div class="val negative">${d.max_drawdown}%</div></div>
<div class="bt-metric"><div class="label">Win Rate</div><div class="val">${d.win_rate}%</div></div>
<div class="bt-metric"><div class="label">Profit Factor</div><div class="val">${d.profit_factor}</div></div>
<div class="bt-metric"><div class="label">Total Trades</div><div class="val">${d.total_trades}</div></div>
<div class="bt-metric"><div class="label">Win / Loss</div><div class="val"><span class="positive">${d.winning_trades}</span> / <span class="negative">${d.losing_trades}</span></div></div>
<div class="bt-metric"><div class="label">Avg Win</div><div class="val positive">$${d.avg_win}</div></div>
<div class="bt-metric"><div class="label">Avg Loss</div><div class="val negative">$${d.avg_loss}</div></div>
</div>`;
}

// ===== ML =====
async function doML(){
const sym=$('ml-sym').value.trim(),days=$('ml-days').value;
const btn=$('btn-ml');btn.disabled=true;btn.innerHTML='<span class="spinner"></span> Training...';
status('status-ml','Training ML models on '+sym+'... ~30s','info');
try{
const r=await fetch('/api/ml',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({symbol:sym,days:parseInt(days)})});
const d=await r.json();
if(d.error){status('status-ml',d.error,'error')}
else{showML(d);status('status-ml','ML analysis complete!','success')}
}catch(e){status('status-ml','Error: '+e.message,'error')}
btn.disabled=false;btn.innerHTML='🧠 Train & Predict';
}

function showML(d){
const el=$('ml-result');el.style.display='block';
const ac={BUY:'positive',SELL:'negative',HOLD:'neutral'},ae={BUY:'🟢',SELL:'🔴',HOLD:'⏸️'};
const c=ac[d.prediction]||'neutral';

let models='';
if(d.models)models=d.models.map(m=>{
const mc=m.accuracy>60?'positive':m.accuracy>50?'':'negative';
return `<div class="bt-metric"><div class="label">${m.name}</div><div class="val ${mc}">${m.accuracy}%</div></div>`}).join('');

let feats='';
if(d.top_features)feats=d.top_features.map(f=>`<span style="margin-right:12px;font-size:12px"><span style="color:#888">${f.name}:</span> ${f.importance}</span>`).join('');

el.innerHTML=`
<h3 style="color:#a064ff;margin-bottom:10px">🤖 ML Prediction: ${d.symbol}</h3>
<div style="margin-bottom:10px">
<span class="${c}" style="font-size:22px;font-weight:bold">${ae[d.prediction]||''} ${d.prediction}</span>
<span style="color:#888;margin-left:10px">Confidence: ${d.confidence}%</span>
</div>
<div style="margin-bottom:10px;font-size:13px">
<span style="margin-right:15px">📈 BUY: ${d.prob_buy||0}%</span>
<span style="margin-right:15px">⏸️ HOLD: ${d.prob_hold||0}%</span>
<span>📉 SELL: ${d.prob_sell||0}%</span>
</div>
<div style="margin:10px 0"><strong style="color:#888;font-size:12px">Models:</strong></div>
${models}
<div style="margin-top:8px"><strong style="color:#888;font-size:12px">Top Features:</strong><br>${feats}</div>`;
}

// Auto-refresh
setInterval(loadPortfolio, 30000);
loadPortfolio();
</script>
</body>
</html>
"""

# ===================== ROUTES =====================

@app.route('/')
def dashboard():
    return render_template_string(DASHBOARD_HTML)


@app.route('/api/portfolio')
def api_portfolio():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute('SELECT COUNT(*) FROM positions WHERE status="open"')
    open_count = cur.fetchone()[0]
    cur.execute('SELECT COUNT(*) FROM positions WHERE status="closed"')
    total = cur.fetchone()[0]
    cur.execute('SELECT SUM(pnl) FROM positions WHERE status="closed"')
    total_pnl = cur.fetchone()[0] or 0
    cur.execute('SELECT COUNT(*) FROM positions WHERE status="closed" AND pnl>0')
    wins = cur.fetchone()[0]
    cur.execute('SELECT MAX(pnl) FROM positions WHERE status="closed"')
    best = cur.fetchone()[0] or 0
    cur.execute('SELECT MIN(pnl) FROM positions WHERE status="closed"')
    worst = cur.fetchone()[0] or 0

    wr = (wins / total * 100) if total > 0 else 0

    cur.execute('SELECT * FROM positions WHERE status="open" ORDER BY entry_time DESC')
    open_list = [{'id': r['id'], 'symbol': r['symbol'], 'side': r['side'],
                  'entry_price': r['entry_price'], 'stop_loss': r['stop_loss'],
                  'take_profit': r['take_profit'], 'quantity': r['quantity']} for r in cur.fetchall()]

    cur.execute('SELECT * FROM trades_log ORDER BY timestamp DESC LIMIT 15')
    recent = []
    for r in cur.fetchall():
        details = json.loads(r['details']) if r['details'] else {}
        recent.append({'timestamp': str(r['timestamp'])[:19], 'action': r['action'],
                       'symbol': r['symbol'], 'price': r['price'] or 0,
                       'quantity': r['quantity'] or 0, 'pnl': details.get('pnl', 0)})
    conn.close()

    return jsonify({'open_positions': open_count, 'total_trades': total,
                    'total_pnl': total_pnl, 'win_rate': wr,
                    'best_trade': best, 'worst_trade': worst,
                    'open_positions_list': open_list, 'recent_trades': recent})


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'BTC/USDT')

        from data.provider import data_provider
        from data.technical import tech_analyzer
        from data.tradingview import tradingview_analyzer
        from agents.trading_floor import trading_floor

        is_crypto = '/' in symbol
        if is_crypto:
            df = data_provider.get_crypto_data(symbol, days=30, timeframe='1h')
        else:
            df = data_provider.get_stock_data(symbol, days=30, interval='1h')

        if df is None or df.empty:
            return jsonify({'error': f'No data for {symbol}'})

        technicals = tech_analyzer.analyze(df)

        # Get TradingView signal
        tv_signal = 'N/A'
        try:
            if is_crypto:
                tv = tradingview_analyzer.get_crypto_analysis(symbol.replace('/', ''), '1h')
            else:
                tv = tradingview_analyzer.get_stock_analysis(symbol, '1d')
            if tv:
                tv_signal = tv.recommendation
        except:
            pass

        context = {
            'symbol': symbol,
            'current_price': technicals['price']['current'],
            'technicals': technicals,
            'recent_performance': technicals['price'],
            'fundamentals': {},
            'news': [],
        }

        decision = trading_floor.analyze_and_decide(context)

        return jsonify({
            'symbol': symbol,
            'action': decision.action,
            'confidence': round(decision.confidence * 100),
            'consensus_score': round(decision.consensus_score * 100),
            'entry_price': decision.entry_price,
            'stop_loss': decision.stop_loss,
            'take_profit': decision.take_profit,
            'current_price': technicals['price']['current'],
            'rsi': round(technicals['momentum']['rsi_14'], 1) if technicals['momentum'].get('rsi_14') else None,
            'macd': round(technicals['momentum']['macd_line'], 2),
            'volume_signal': technicals['volume_analysis'].get('volume_signal', 'N/A'),
            'tv_signal': tv_signal,
            'agent_votes': [{'agent_name': v.agent_name, 'action': v.action,
                             'confidence': round(v.confidence * 100), 'reasoning': v.reasoning}
                            for v in decision.agent_votes],
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)})


@app.route('/api/execute', methods=['POST'])
def api_execute():
    try:
        data = request.get_json()
        from execution.engine import execution_engine
        from agents.trading_floor import TradeDecision

        decision = TradeDecision(
            symbol=data['symbol'], action=data['action'], confidence=0.7,
            entry_price=data['entry_price'],
            stop_loss=data.get('stop_loss', 0), take_profit=data.get('take_profit', 0),
            position_size_pct=0.05, reasoning='Manual execution from dashboard',
        )
        position = execution_engine.execute_decision(decision)
        if position:
            return jsonify({'success': True, 'position_id': position.id})
        return jsonify({'error': 'Trade not executed (HOLD or risk limit)'})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/close-position', methods=['POST'])
def api_close_pos():
    try:
        data = request.get_json()
        from execution.engine import execution_engine
        for pos in execution_engine._get_open_positions():
            if pos.id == data['position_id']:
                price = execution_engine._get_current_price(pos.symbol)
                if price:
                    execution_engine._close_position(pos, price, 'Manual close')
                    return jsonify({'success': True})
        return jsonify({'error': 'Position not found'})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/close-all', methods=['POST'])
def api_close_all():
    try:
        from execution.engine import execution_engine
        closed = 0
        for pos in execution_engine._get_open_positions():
            price = execution_engine._get_current_price(pos.symbol)
            if price:
                execution_engine._close_position(pos, price, 'Close all')
                closed += 1
        return jsonify({'message': f'Closed {closed} positions'})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/scan', methods=['POST'])
def api_scan():
    try:
        from data.provider import data_provider
        from data.technical import tech_analyzer
        from agents.trading_floor import trading_floor

        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT']
        signals = []
        for sym in symbols:
            try:
                df = data_provider.get_crypto_data(sym, days=30, timeframe='1h')
                if df is None:
                    continue
                technicals = tech_analyzer.analyze(df)
                context = {'symbol': sym, 'current_price': technicals['price']['current'],
                           'technicals': technicals, 'recent_performance': technicals['price'],
                           'fundamentals': {}, 'news': []}
                decision = trading_floor.analyze_and_decide(context)
                signals.append({'symbol': sym, 'action': decision.action,
                                'confidence': round(decision.confidence * 100),
                                'current_price': technicals['price']['current']})
            except Exception as e:
                print(f'Scan {sym} failed: {e}')
                continue
        return jsonify({'signals': signals})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/backtest', methods=['POST'])
def api_backtest():
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'BTC/USDT')
        strategy_name = data.get('strategy', 'rsi')
        days = data.get('days', 90)

        from backtest.engine import (BacktestEngine, RSIMeanReversion, MACrossover,
                                      BollingerBreakout, MACDStrategy, MultiIndicatorStrategy)
        from data.provider import data_provider

        strategies = {'rsi': RSIMeanReversion, 'ma_crossover': MACrossover,
                      'bollinger': BollingerBreakout, 'macd': MACDStrategy,
                      'multi': MultiIndicatorStrategy}

        strat_cls = strategies.get(strategy_name)
        if not strat_cls:
            return jsonify({'error': f'Unknown strategy: {strategy_name}'})

        is_crypto = '/' in symbol
        if is_crypto:
            df = data_provider.get_crypto_data(symbol, days=days, timeframe='1h')
        else:
            df = data_provider.get_stock_data(symbol, days=days, interval='1h')

        if df is None or df.empty:
            return jsonify({'error': f'No data for {symbol}'})

        engine = BacktestEngine()
        result = engine.run(strat_cls(), df, symbol)

        return jsonify({
            'strategy_name': result.strategy_name, 'symbol': result.symbol,
            'start_date': str(result.start_date.date()), 'end_date': str(result.end_date.date()),
            'initial_capital': round(result.initial_capital, 2),
            'final_capital': round(result.final_capital, 2),
            'total_return': round(result.total_return * 100, 2),
            'annual_return': round(result.annual_return * 100, 2),
            'max_drawdown': round(result.max_drawdown * 100, 2),
            'sharpe_ratio': round(result.sharpe_ratio, 2),
            'sortino_ratio': round(result.sortino_ratio, 2),
            'win_rate': round(result.win_rate * 100, 1),
            'profit_factor': round(result.profit_factor, 2),
            'total_trades': result.total_trades,
            'winning_trades': result.winning_trades,
            'losing_trades': result.losing_trades,
            'avg_win': round(result.avg_win, 2),
            'avg_loss': round(result.avg_loss, 2),
            'largest_win': round(result.largest_win, 2),
            'largest_loss': round(result.largest_loss, 2),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)})


@app.route('/api/ml', methods=['POST'])
def api_ml():
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'BTC/USDT')
        days = data.get('days', 180)

        from data.ml_alpha import ml_engine
        from data.provider import data_provider

        is_crypto = '/' in symbol
        if is_crypto:
            df = data_provider.get_crypto_data(symbol, days=days, timeframe='1h')
        else:
            df = data_provider.get_stock_data(symbol, days=days, interval='1h')

        if df is None or df.empty:
            return jsonify({'error': f'No data for {symbol}'})

        # Train
        results = ml_engine.train(df, symbol)

        # Predict
        signal = ml_engine.predict(df, symbol)
        if not signal:
            return jsonify({'error': 'ML prediction failed'})

        models = []
        for name, res in results.items():
            models.append({'name': name.replace('_', ' ').title(),
                           'accuracy': round(res['cv_accuracy'] * 100, 1)})

        top_features = [{'name': k, 'importance': round(v, 4)}
                        for k, v in list(signal.feature_importance.items())[:8]]

        return jsonify({
            'symbol': symbol,
            'prediction': signal.prediction,
            'confidence': round(signal.confidence * 100),
            'prob_buy': round(signal.probability.get('BUY', 0) * 100),
            'prob_hold': round(signal.probability.get('HOLD', 0) * 100),
            'prob_sell': round(signal.probability.get('SELL', 0) * 100),
            'models': models,
            'top_features': top_features,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    print("=" * 50)
    print("  AI Trading Floor v2 — Dashboard")
    print("  http://localhost:5000")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=False)
