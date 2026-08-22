/* CNN 卷积过程可视化 - 主逻辑（Canvas 2D 固定等轴测视角渲染，替代 WebGL 3D） */
'use strict';

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

if (window.self !== window.top) {
  document.body.classList.add('embedded');
}

const OUTPUT_GAP = 2.5;   // 输出距输入右缘的间距
const OUTPUT_MARGIN = 1.4; // 输出底部距其投影区域顶部的净空
const KERNEL_RAISE = 0.5;  // 卷积核分区的小幅抬高
const HIGHLIGHT = '#ffe066'; // 滑动遍历的统一高亮色

const COLORS = {
  input: '#4a90d9',
  pad: '#55606e',
  kernel: '#e67e22',
  outputIdle: '#2a3344',
  outputDone: '#2ecc71',
  edge: '#9aa7bd',
  titleIn: '#6fa8ff',
  titleKernel: '#ffb45e',
  titleOut: '#7ee2a8',
  region: '#ffe066',
};

function fmt(v) {
  const r = Math.round(v * 10) / 10;
  return Number.isInteger(r) ? String(r) : r.toFixed(1);
}

/* ---------------- 2D 渲染器（固定等轴测视角） ---------------- */
const container = $('scene-container');
const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');
container.appendChild(canvas);

const U_COS = Math.cos(25 * Math.PI / 180);
const U_SIN = Math.sin(25 * Math.PI / 180);

let scale = 40, ox = 0, oy = 0, dpr = 1, W = 0, H = 0;

/* 世界坐标(x,y,z) -> 等轴测单位坐标。ux 右为 +x，uy 下为 +z（越往下越近），ud 为深度排序值 */
function proj(x, y, z) {
  return {
    ux: (x - z) * U_COS,
    uy: (x + z) * U_SIN - y,
    ud: (x + z) * U_SIN + y * U_COS,
  };
}
const toX = (p) => ox + p.ux * scale;
const toY = (p) => oy + p.uy * scale;

/* 立方体轮廓（8 个角投影后的凸包 = 等轴测可见轮廓） */
function boxHullUnit(cx, cy, cz, sx, sy, sz) {
  const pts = [];
  for (const dx of [-1, 1]) {
    for (const dy of [-1, 1]) {
      for (const dz of [-1, 1]) {
        pts.push(proj(cx + dx * sx / 2, cy + dy * sy / 2, cz + dz * sz / 2));
      }
    }
  }
  return convexHull(pts);
}

function convexHull(pts) {
  const sorted = pts.slice().sort((a, b) => a.ux - b.ux || a.uy - b.uy);
  const cross = (o, a, b) => (a.ux - o.ux) * (b.uy - o.uy) - (a.uy - o.uy) * (b.ux - o.ux);
  const lower = [];
  for (const p of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  const upper = [];
  for (let i = sorted.length - 1; i >= 0; i--) {
    const p = sorted[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  lower.pop(); upper.pop();
  return lower.concat(upper);
}

function drawPoly(pts, fill, alpha, stroke, lw, strokeColor) {
  ctx.beginPath();
  ctx.moveTo(toX(pts[0]), toY(pts[0]));
  for (let i = 1; i < pts.length; i++) ctx.lineTo(toX(pts[i]), toY(pts[i]));
  ctx.closePath();
  if (fill) { ctx.globalAlpha = alpha; ctx.fillStyle = fill; ctx.fill(); }
  if (stroke) {
    ctx.globalAlpha = stroke;
    ctx.strokeStyle = strokeColor || COLORS.edge;
    ctx.lineWidth = lw || 1;
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

/* 纯色方块：单个颜色填充轮廓，不加三面明暗 */
function drawBox(cx, cy, cz, sx, sy, sz, base, alpha, glow) {
  const hull = boxHullUnit(cx, cy, cz, sx, sy, sz);
  const ea = alpha < 0.5 ? 0.35 : 0.55;
  drawPoly(hull, base, alpha, ea, 1);
  if (glow) {
    drawPoly(hull, glow, 0.25, 1, Math.max(2, scale * 0.08), glow);
  }
}

function drawText(p, text, o) {
  o = o || {};
  const size = o.size || 13;
  const x = toX(p), y = toY(p);
  ctx.font = (o.bold ? '600 ' : '') + size + 'px "Segoe UI", "Microsoft YaHei", sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.lineJoin = 'round';
  ctx.lineWidth = Math.max(2, size / 5);
  ctx.strokeStyle = 'rgba(0,0,0,0.72)';
  ctx.strokeText(text, x, y);
  ctx.fillStyle = o.color || '#ffffff';
  ctx.fillText(text, x, y);
}

/* ---------------- 状态 ---------------- */
const state = {
  m: 5, n: 5, s: 3, t: 3, stride: 1,
  padMode: 'valid', ph: 1, pw: 1,
  padT: 0, padB: 0, padL: 0, padR: 0,
  speed: 50,
};
let inputVals = [], kernelVals = [];
let P = [], M_ = 0, N_ = 0;          // 填充后网格
let outH = 0, outW = 0, outCX = 0;
let positions = [], outputVals = [];
let currentIndex = 0;
let isPlaying = false, isAnimating = false;
let animGen = 0;

let kernelZone = { x: -6, y: KERNEL_RAISE, z: 0 };   // 卷积核独立分区（右侧，静态）
let windowPos = null;               // 当前卷积覆盖的输入区域 {x, z}
let beamFrom = null;                // 投影光束来源输出格 {i, j}
let OUTY = 0;                       // 输出矩阵组中心高度（右上方）
let sweepIn = null;                 // 输入格滑动高亮位置 {x, z}
let sweepK = null;                  // 卷积核格滑动高亮位置 {x, y, z}

let inputMeta = [];   // [pr][pc] = { isPad, glow }
let kernelMeta = [];  // [ki][kj] = { glow }
let outputMeta = [];  // [i][j] = { done, pop }
let clickCells = [];

/* ---------------- 计算 ---------------- */
function compute() {
  const { m, n, s, t, stride } = state;
  let padT = 0, padB = 0, padL = 0, padR = 0;
  switch (state.padMode) {
    case 'same': {
      padT = Math.floor((s - 1) / 2); padB = (s - 1) - padT;
      padL = Math.floor((t - 1) / 2); padR = (t - 1) - padL;
      break;
    }
    case 'full':
      padT = padB = s - 1; padL = padR = t - 1;
      break;
    case 'custom':
      padT = padB = Math.max(0, state.ph); padL = padR = Math.max(0, state.pw);
      break;
    default:
      break;
  }
  state.padT = padT; state.padB = padB; state.padL = padL; state.padR = padR;

  M_ = m + padT + padB;
  N_ = n + padL + padR;
  P = Array.from({ length: M_ }, () => Array(N_).fill(0));
  for (let r = 0; r < m; r++) for (let c = 0; c < n; c++) P[r + padT][c + padL] = inputVals[r][c];

  outH = Math.floor((M_ - s) / stride) + 1;
  outW = Math.floor((N_ - t) / stride) + 1;

  positions = [];
  if (outH > 0 && outW > 0) {
    for (let i = 0; i < outH; i++) {
      for (let j = 0; j < outW; j++) {
        const r0 = i * stride, c0 = j * stride;
        let sum = 0;
        for (let ki = 0; ki < s; ki++) for (let kj = 0; kj < t; kj++) sum += P[r0 + ki][c0 + kj] * kernelVals[ki][kj];
        positions.push({
          i, j, r0, c0, sum,
          cx: c0 + (t - 1) / 2 - (N_ - 1) / 2,
          cz: r0 + (s - 1) / 2 - (M_ - 1) / 2,
        });
      }
    }
  }
  outputVals = outH > 0 && outW > 0 ? Array.from({ length: outH }, () => Array(outW).fill(null)) : [];
  currentIndex = 0;
}

/* ---------------- 单元格世界坐标 ---------------- */
function inputCellWorld(pr, pc) {
  return { x: pc - (N_ - 1) / 2, y: 0, z: pr - (M_ - 1) / 2, sx: 0.94, sy: 0.12, sz: 0.94 };
}
function kernelCellWorld(ki, kj) {
  const lx = kj - (state.t - 1) / 2, lz = ki - (state.s - 1) / 2;
  return { x: lx + kernelZone.x, y: kernelZone.y, z: lz + kernelZone.z, sx: 0.94, sy: 0.24, sz: 0.94 };
}
function outputCellWorld(i, j) {
  const lx = j - (outW - 1) / 2, ly = OUTY + (outH - 1) / 2 - i;
  const s = outputMeta[i][j] ? outputMeta[i][j].pop : 1;
  return { x: lx + outCX, y: ly, z: 0, sx: 0.94 * s, sy: 0.94 * s, sz: 0.12 * s };
}
function isPadCell(pr, pc) {
  return pr < state.padT || pr >= state.padT + state.m || pc < state.padL || pc >= state.padL + state.n;
}

/* ---------------- 场景构建（元数据） ---------------- */
function buildInput() {
  inputMeta = [];
  clickCells = [];
  for (let pr = 0; pr < M_; pr++) {
    inputMeta[pr] = [];
    for (let pc = 0; pc < N_; pc++) {
      const isPad = isPadCell(pr, pc);
      inputMeta[pr][pc] = { isPad, glow: null };
      if (!isPad) clickCells.push({ kind: 'input', pr, pc });
    }
  }
}

function buildKernel() {
  kernelMeta = [];
  kernelZone = {
    x: (N_ - 1) / 2 + (M_ - 1) / 2 + (state.t - 1) / 2 + (state.s - 1) / 2 + 2.0,
    y: KERNEL_RAISE,
    z: 0,
  };
  for (let ki = 0; ki < state.s; ki++) {
    kernelMeta[ki] = [];
    for (let kj = 0; kj < state.t; kj++) {
      kernelMeta[ki][kj] = { glow: null };
      clickCells.push({ kind: 'kernel', ki, kj });
    }
  }
}

function buildOutput() {
  outputMeta = [];
  outCX = (N_ - 1) / 2 + OUTPUT_GAP + (outW - 1) / 2;
  const refUy = ((N_ - 1) / 2 - (M_ - 1) / 2) * U_SIN - 0.12;
  const outBottomUy = (outCX + (outW - 1) / 2) * U_SIN;
  OUTY = outBottomUy - refUy + (outH - 1) / 2 + 0.47 + OUTPUT_MARGIN;
  for (let i = 0; i < outH; i++) {
    outputMeta[i] = [];
    for (let j = 0; j < outW; j++) outputMeta[i][j] = { done: false, pop: 1, active: false, live: null };
  }
}

function setOutputCellState(i, j, done) {
  if (outputMeta[i][j]) outputMeta[i][j].done = done;
}

function restoreGlows() {
  for (let pr = 0; pr < M_; pr++) for (let pc = 0; pc < N_; pc++) inputMeta[pr][pc].glow = null;
  for (let ki = 0; ki < state.s; ki++) for (let kj = 0; kj < state.t; kj++) kernelMeta[ki][kj].glow = null;
}

/* ---------------- 渲染 ---------------- */
function render() {
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#1a1f28';
  ctx.fillRect(0, 0, W, H);

  const items = [];

  for (let pr = 0; pr < M_; pr++) {
    for (let pc = 0; pc < N_; pc++) {
      const m = inputMeta[pr][pc];
      const w = inputCellWorld(pr, pc);
      const c = m.isPad ? COLORS.pad : COLORS.input;
      const a = m.isPad ? 0.32 : 0.92;
      items.push({ d: proj(w.x, 0, w.z).ud, draw: () => drawBox(w.x, w.y, w.z, w.sx, w.sy, w.sz, c, a, m.glow) });
    }
  }

  for (let ki = 0; ki < state.s; ki++) {
    for (let kj = 0; kj < state.t; kj++) {
      const m = kernelMeta[ki][kj];
      const w = kernelCellWorld(ki, kj);
      items.push({ d: proj(w.x, w.y, w.z).ud, draw: () => drawBox(w.x, w.y, w.z, w.sx, w.sy, w.sz, COLORS.kernel, 0.94, m.glow) });
    }
  }

  for (let i = 0; i < outH; i++) {
    for (let j = 0; j < outW; j++) {
      const m = outputMeta[i][j];
      const w = outputCellWorld(i, j);
      const c = m.done ? COLORS.outputDone : COLORS.outputIdle;
      const a = m.done ? 0.92 : 0.28;
      items.push({ d: proj(w.x, w.y, 0).ud, draw: () => drawBox(w.x, w.y, w.z, w.sx, w.sy, w.sz, c, a, m.active ? '#ffffff' : null) });
    }
  }

  items.sort((a, b) => a.d - b.d);
  for (const it of items) it.draw();

  if (sweepIn) drawSweepRing(sweepIn.x, 0.12, sweepIn.z);
  if (sweepK) drawSweepRing(sweepK.x, sweepK.y + 0.12, sweepK.z);

  if (windowPos) drawProjection();

  const fbase = Math.max(11, Math.min(20, Math.round(scale * 0.32)));

  for (let pr = 0; pr < M_; pr++) {
    for (let pc = 0; pc < N_; pc++) {
      const m = inputMeta[pr][pc];
      const w = inputCellWorld(pr, pc);
      const p = proj(w.x, 0.06, w.z);
      drawText(p, m.isPad ? '0' : fmt(inputVals[pr - state.padT][pc - state.padL]), {
        size: m.isPad ? fbase - 1 : fbase,
        color: m.isPad ? '#97a4b8' : '#ffffff',
      });
    }
  }

  for (let ki = 0; ki < state.s; ki++) {
    for (let kj = 0; kj < state.t; kj++) {
      const w = kernelCellWorld(ki, kj);
      drawText(proj(w.x, w.y + 0.12, w.z), fmt(kernelVals[ki][kj]), { size: fbase, color: '#ffffff' });
    }
  }

  for (let i = 0; i < outH; i++) {
    for (let j = 0; j < outW; j++) {
      const w = outputCellWorld(i, j);
      const p = proj(w.x, w.y, 0.06);
      if (outputMeta[i][j].live != null) drawText(p, fmt(outputMeta[i][j].live), { size: fbase, color: '#ffe066' });
      else if (outputMeta[i][j].done) drawText(p, fmt(outputVals[i][j]), { size: fbase, color: '#ffffff' });
      else drawText(p, '–', { size: fbase, color: '#5a667a' });
    }
  }

  drawText(proj(-(N_ - 1) / 2 - 1.2, 0.9, -(M_ - 1) / 2 - 0.4), '输入矩阵', { size: fbase + 4, color: COLORS.titleIn, bold: true });
  drawText(proj(kernelZone.x, kernelZone.y + 0.9, kernelZone.z - (state.s - 1) / 2 - 0.85), '卷积核', { size: fbase + 4, color: COLORS.titleKernel, bold: true });
  if (outH > 0 && outW > 0) {
    drawText(proj(outCX - (outW - 1) / 2 - 1.5, OUTY + (outH - 1) / 2 + 0.9, 0.7), '输出矩阵', { size: fbase + 4, color: COLORS.titleOut, bold: true });
  }
}

/* 投影动画：覆盖区淡色帧 + 输出格四角到区域四角的虚线连接，带体积感淡影 */
function drawProjection() {
  const { x, z } = windowPos;
  const regPts = [
    proj(x - state.t / 2, 0.04, z - state.s / 2),
    proj(x + state.t / 2, 0.04, z - state.s / 2),
    proj(x + state.t / 2, 0.04, z + state.s / 2),
    proj(x - state.t / 2, 0.04, z + state.s / 2),
  ];
  drawPoly(regPts, COLORS.region, 0.06, 0.9, 2.2, COLORS.region);
  if (beamFrom) {
    const w = outputCellWorld(beamFrom.i, beamFrom.j);
    const hx = w.sx / 2, hy = w.sy / 2, fz = w.z + w.sz / 2;
    const outCorners = [
      [w.x - hx, w.y - hy, fz],
      [w.x + hx, w.y - hy, fz],
      [w.x + hx, w.y + hy, fz],
      [w.x - hx, w.y + hy, fz],
    ];
    // 体积感淡影：输出格到覆盖区之间填充一个极淡的“投影体”
    const all = outCorners.map((c) => proj(c[0], c[1], c[2])).concat(regPts);
    drawPoly(convexHull(all), COLORS.region, 0.045, null, 0);
    // 四角虚线连接
    ctx.save();
    ctx.setLineDash([6, 6]);
    ctx.strokeStyle = 'rgba(255,224,102,0.55)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let k = 0; k < 4; k++) {
      const p1 = proj(outCorners[k][0], outCorners[k][1], outCorners[k][2]);
      ctx.moveTo(toX(p1), toY(p1));
      ctx.lineTo(toX(regPts[k]), toY(regPts[k]));
    }
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();
  }
}

/* ---------------- 动画 ---------------- */
function stepDelay() { return Math.max(120, 1300 - state.speed * 12); }

/* 单格节拍：speed 1..100，默认(50) ≈ 0.7s/格；导出可用 speed>100 提速 */
function cellDelay() {
  return Math.max(60, Math.round(1300 - 1200 * (state.speed - 1) / 99));
}

function easeInOutCubic(t) {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

/* 窗口框在输入上平滑滑动（滑动的感觉） */
function tweenWindow(tx, tz, dur) {
  if (!windowPos) { windowPos = { x: tx, z: tz }; return Promise.resolve(); }
  const from = { x: windowPos.x, z: windowPos.z };
  const t0 = performance.now();
  return new Promise((resolve) => {
    const step = () => {
      const t = Math.min(1, (performance.now() - t0) / dur);
      const e = easeInOutCubic(t);
      windowPos.x = from.x + (tx - from.x) * e;
      windowPos.z = from.z + (tz - from.z) * e;
      if (t < 1) requestAnimationFrame(step);
      else resolve();
    };
    step();
  });
}

/* 高亮环在输入格/卷积核格之间平滑滑动 */
function tweenSweep(txIn, tzIn, txK, tyK, tzK, dur) {
  const fIn = { x: sweepIn.x, z: sweepIn.z };
  const fK = { x: sweepK.x, y: sweepK.y, z: sweepK.z };
  const t0 = performance.now();
  return new Promise((resolve) => {
    const step = () => {
      const t = Math.min(1, (performance.now() - t0) / dur);
      const e = easeInOutCubic(t);
      sweepIn.x = fIn.x + (txIn - fIn.x) * e;
      sweepIn.z = fIn.z + (tzIn - fIn.z) * e;
      sweepK.x = fK.x + (txK - fK.x) * e;
      sweepK.y = fK.y + (tyK - fK.y) * e;
      sweepK.z = fK.z + (tzK - fK.z) * e;
      if (t < 1) requestAnimationFrame(step);
      else resolve();
    };
    step();
  });
}

/* 单个滑动的菱形高亮环 */
function drawSweepRing(cx, cy, cz) {
  const c = (dx, dz) => proj(cx + dx, cy, cz + dz);
  const pts = [c(-0.47, -0.47), c(0.47, -0.47), c(0.47, 0.47), c(-0.47, 0.47)];
  drawPoly(pts, HIGHLIGHT, 0.16, 1, Math.max(2, scale * 0.07), HIGHLIGHT);
}

async function animateStep(pos, gen) {
  isAnimating = true;
  try {
    const D = stepDelay();
    const { i, j, r0, c0, cx, cz, sum } = pos;

    // 滑动到当前覆盖位置
    await tweenWindow(cx, cz, Math.min(420, D));
    if (gen !== animGen) return;

    windowPos = { x: cx, z: cz };
    beamFrom = { i, j };
    outputMeta[i][j].active = true;
    beginFormulaStep(currentIndex + 1);

    const terms = [];
    const prods = [];
    let running = 0;
    for (let ki = 0; ki < state.s; ki++) {
      for (let kj = 0; kj < state.t; kj++) {
        const iv = P[r0 + ki][c0 + kj], kv = kernelVals[ki][kj], prod = iv * kv;
        running += prod;
        const tstr = (kv < 0 ? '(' + fmt(kv) + ')' : fmt(kv)) + ' × ' + (iv < 0 ? '(' + fmt(iv) + ')' : fmt(iv)) + ' = ' + fmt(prod);
        terms.push(tstr);
        prods.push({ ki, kj, pr: r0 + ki, pc: c0 + kj, running });
      }
    }

    // 单色滑动遍历：高亮环在覆盖格之间平滑滑动，逐格展示并实时更新输出动态累计值
    const cellSleep = cellDelay();
    const glideMs = Math.max(60, Math.round(cellSleep * 0.55));
    const dwellMs = Math.max(0, cellSleep - glideMs);

    for (let k = 0; k < prods.length; k++) {
      if (gen !== animGen) return;
      const c = prods[k];
      const iw = inputCellWorld(c.pr, c.pc);
      const kw = kernelCellWorld(c.ki, c.kj);
      if (k === 0) {
        sweepIn = { x: iw.x, z: iw.z };
        sweepK = { x: kw.x, y: kw.y, z: kw.z };
      } else {
        await tweenSweep(iw.x, iw.z, kw.x, kw.y, kw.z, glideMs);
        if (gen !== animGen) return;
      }
      outputMeta[i][j].live = c.running;
      addFormulaEl(terms[k]);
      updateStatus('输入 (' + c.pr + ',' + c.pc + ') ↔ 核 (' + c.ki + ',' + c.kj + ')　|　第 ' + (k + 1) + '/' + prods.length + ' 项，累计和 = ' + fmt(c.running));
      await sleep(dwellMs);
      if (gen !== animGen) return;
    }
    sweepIn = null;
    sweepK = null;
    endFormulaStep();

    if (gen !== animGen) return;
    restoreGlows();

    outputVals[i][j] = sum;
    outputMeta[i][j].done = true;
    outputMeta[i][j].active = false;
    outputMeta[i][j].live = null;
    outputMeta[i][j].pop = 1.15;
    await sleep(D * 0.25);
    outputMeta[i][j].pop = 1;
    beamFrom = null;

    updateStatus('输出[' + i + '][' + j + '] = ' + fmt(sum) + '　|　核覆盖范围：行 ' + r0 + '–' + (r0 + state.s - 1) + '，列 ' + c0 + '–' + (c0 + state.t - 1));
  } finally {
    isAnimating = false;
  }
}

async function playLoop(gen) {
  while (isPlaying && animGen === gen && currentIndex < positions.length) {
    await animateStep(positions[currentIndex], gen);
    if (animGen !== gen) return;
    currentIndex++;
    updateProgress();
    if (currentIndex >= positions.length) {
      isPlaying = false;
      updatePlayBtn();
      updateStatus('全部完成！共 ' + positions.length + ' 个卷积位置。');
      isAnimating = false;
      updateControls();
      return;
    }
  }
  isAnimating = false;
  updateControls();
}

async function stepForward() {
  if (isAnimating || isPlaying) return;
  if (currentIndex >= positions.length) return;
  const gen = animGen;
  await animateStep(positions[currentIndex], gen);
  if (animGen === gen) { currentIndex++; updateProgress(); }
  updateControls();
}

function stepBackward() {
  if (isAnimating || isPlaying) return;
  if (currentIndex <= 0) return;
  currentIndex--;
  const pos = positions[currentIndex];
  const { i, j, cx, cz } = pos;
  outputVals[i][j] = null;
  outputMeta[i][j].done = false;
  outputMeta[i][j].active = false;
  outputMeta[i][j].live = null;
  outputMeta[i][j].pop = 1;
  windowPos = { x: cx, z: cz };
  beamFrom = null;
  sweepIn = null;
  sweepK = null;
  restoreGlows();
  removeLastFormulaStep();
  updateProgress();
  updateStatus('已回退到第 ' + currentIndex + ' 步');
  updateControls();
}

function resetAnimation() {
  animGen++;
  isPlaying = false;
  isAnimating = false;
  currentIndex = 0;
  outputVals = positions.length ? Array.from({ length: outH }, () => Array(outW).fill(null)) : [];
  for (let i = 0; i < outH; i++) for (let j = 0; j < outW; j++) { outputMeta[i][j].done = false; outputMeta[i][j].active = false; outputMeta[i][j].live = null; outputMeta[i][j].pop = 1; }
  restoreGlows();
  beamFrom = null;
  sweepIn = null;
  sweepK = null;
  clearFormula();
  const first = positions[0];
  windowPos = { x: first ? first.cx : 0, z: first ? first.cz : 0 };
  updatePlayBtn();
  updateProgress();
  updateStatus('已重置，点击「播放」开始');
  updateControls();
}

/* ---------------- UI 更新 ---------------- */
function updateProgress() {
  $('progress').textContent = currentIndex + ' / ' + positions.length;
  $('outSize').textContent = outH > 0 && outW > 0 ? outW + ' × ' + outH + '（列 × 行）' : '非法';
}

function updateStatus(text) { $('status').textContent = text || ''; }

/* ---------------- 计算过程滑动窗口 ---------------- */
let formulaBlocks = [];   // 每步一个块：{ els: [元素] }
let currentBlock = null;  // 当前正在追加的块

function scrollFormulaToBottom() {
  const el = $('formula');
  el.scrollTop = el.scrollHeight;
}

function addFormulaEl(html, cls) {
  const el = document.createElement('div');
  el.className = cls || '';
  el.innerHTML = html;
  $('formula').appendChild(el);
  if (currentBlock) currentBlock.els.push(el);
  scrollFormulaToBottom();
}

function beginFormulaStep(stepNo) {
  currentBlock = { els: [] };
  formulaBlocks.push(currentBlock);
  addFormulaEl('step-' + stepNo, 'fstep');
}

function endFormulaStep() { currentBlock = null; }

function clearFormula() {
  $('formula').innerHTML = '';
  formulaBlocks = [];
  currentBlock = null;
}

function removeLastFormulaStep() {
  const block = formulaBlocks.pop();
  if (!block) return;
  for (const el of block.els) el.remove();
  currentBlock = null;
  scrollFormulaToBottom();
}

function updatePlayBtn() {
  $('btnPlay').textContent = isPlaying ? '⏸ 暂停' : '▶ 播放';
}

function updateControls() {
  const busy = isPlaying || isAnimating || positions.length === 0;
  $('btnPlay').disabled = positions.length === 0;
  $('btnPrev').disabled = busy || currentIndex <= 0;
  $('btnNext').disabled = busy || currentIndex >= positions.length;
  $('btnRandom').disabled = busy;
  $('btnClear').disabled = busy;
  updateFormulaScrollState();
}

/* 动画进行中锁定滚动并自动滚到底；空闲/暂停时允许手动滚动查看历史 */
function updateFormulaScrollState() {
  const el = $('formula');
  if (isPlaying || isAnimating) {
    el.style.overflowY = 'hidden';
    scrollFormulaToBottom();
  } else {
    el.style.overflowY = 'auto';
  }
}

function showError(msg) {
  const el = $('error');
  el.textContent = msg;
  el.classList.add('show');
}
function hideError() { $('error').classList.remove('show'); }

/* ---------------- 视野适配 ---------------- */
function computeBounds() {
  const xs = [], ys = [], zs = [];
  xs.push(-(N_ - 1) / 2 - 2.2, (N_ - 1) / 2 + 0.6);
  zs.push(-(M_ - 1) / 2 - 1.2, (M_ - 1) / 2 + 0.6);
  ys.push(0);
  const s = state.s, t = state.t;
  xs.push(kernelZone.x - (t - 1) / 2 - 0.6, kernelZone.x + (t - 1) / 2 + 0.6);
  zs.push(kernelZone.z - (s - 1) / 2 - 1.2, kernelZone.z + (s - 1) / 2 + 0.6);
  ys.push(kernelZone.y + 1.1);
  if (outH > 0 && outW > 0) {
    xs.push(outCX - (outW - 1) / 2 - 1.7, outCX + (outW - 1) / 2 + 0.6);
    ys.push(-(outH - 1) / 2 - 0.6, OUTY + (outH - 1) / 2 + 1.1);
    zs.push(-0.6, 1.1);
  }
  return {
    minX: Math.min.apply(null, xs), maxX: Math.max.apply(null, xs),
    minY: Math.min.apply(null, ys), maxY: Math.max.apply(null, ys),
    minZ: Math.min.apply(null, zs), maxZ: Math.max.apply(null, zs),
  };
}

function fitView() {
  const b = computeBounds();
  const pts = [];
  for (const dx of [b.minX, b.maxX]) {
    for (const dy of [b.minY, b.maxY]) {
      for (const dz of [b.minZ, b.maxZ]) pts.push(proj(dx, dy, dz));
    }
  }
  let minUx = Infinity, maxUx = -Infinity, minUy = Infinity, maxUy = -Infinity;
  for (const p of pts) {
    if (p.ux < minUx) minUx = p.ux;
    if (p.ux > maxUx) maxUx = p.ux;
    if (p.uy < minUy) minUy = p.uy;
    if (p.uy > maxUy) maxUy = p.uy;
  }
  const pad = 26;
  const bw = maxUx - minUx, bh = maxUy - minUy;
  scale = Math.max(4, Math.min((W - pad * 2) / bw, (H - pad * 2) / bh));
  ox = W / 2 - ((minUx + maxUx) / 2) * scale;
  oy = H / 2 - ((minUy + maxUy) / 2) * scale;
}

/* 场景内容在屏幕上的包围盒（CSS 像素），用于导出时裁剪去除大片空白背景 */
function contentScreenBox() {
  let minUx = Infinity, maxUx = -Infinity, minUy = Infinity, maxUy = -Infinity;
  const acc = (pts) => {
    for (const p of pts) {
      if (p.ux < minUx) minUx = p.ux;
      if (p.ux > maxUx) maxUx = p.ux;
      if (p.uy < minUy) minUy = p.uy;
      if (p.uy > maxUy) maxUy = p.uy;
    }
  };
  for (let pr = 0; pr < M_; pr++) for (let pc = 0; pc < N_; pc++) acc(boxHullUnit(inputCellWorld(pr, pc).x, 0, inputCellWorld(pr, pc).z, 0.94, 0.12, 0.94));
  for (let ki = 0; ki < state.s; ki++) for (let kj = 0; kj < state.t; kj++) {
    const w = kernelCellWorld(ki, kj);
    acc(boxHullUnit(w.x, w.y, w.z, w.sx, w.sy, w.sz));
  }
  for (let i = 0; i < outH; i++) for (let j = 0; j < outW; j++) {
    const w = outputCellWorld(i, j);
    acc(boxHullUnit(w.x, w.y, w.z, w.sx, w.sy, w.sz));
  }
  // 标题文本需按其真实尺寸计入包围盒，否则导出时会被裁剪
  const titleAcc = (p, chars) => {
    const size = Math.max(11, Math.min(20, Math.round(scale * 0.32))) + 4;
    const hw = (chars * size) / 2 / scale;
    const hh = (size * 0.7) / scale;
    if (p.ux - hw < minUx) minUx = p.ux - hw;
    if (p.ux + hw > maxUx) maxUx = p.ux + hw;
    if (p.uy - hh < minUy) minUy = p.uy - hh;
    if (p.uy + hh > maxUy) maxUy = p.uy + hh;
  };
  titleAcc(proj(-(N_ - 1) / 2 - 1.2, 0.9, -(M_ - 1) / 2 - 0.4), 4);
  titleAcc(proj(kernelZone.x, kernelZone.y + 0.9, kernelZone.z - (state.s - 1) / 2 - 0.85), 3);
  if (outH > 0 && outW > 0) titleAcc(proj(outCX - (outW - 1) / 2 - 1.5, OUTY + (outH - 1) / 2 + 0.9, 0.7), 4);
  return {
    left: ox + minUx * scale,
    top: oy + minUy * scale,
    width: (maxUx - minUx) * scale,
    height: (maxUy - minUy) * scale,
  };
}

/* ---------------- 重建 ---------------- */
function rebuild() {
  const m = clampInt($('inM').value, 1, 12, 5);
  const n = clampInt($('inN').value, 1, 12, 5);
  const s = clampInt($('kerS').value, 1, 7, 3);
  const t = clampInt($('kerT').value, 1, 7, 3);
  const stride = clampInt($('stride').value, 1, 3, 1);
  state.m = m; state.n = n; state.s = s; state.t = t; state.stride = stride;
  state.ph = clampInt($('padPH').value, 0, 6, 1);
  state.pw = clampInt($('padPW').value, 0, 6, 1);

  const sizeChanged =
    !inputVals.length || inputVals.length !== m || !inputVals[0] || inputVals[0].length !== n ||
    !kernelVals.length || kernelVals.length !== s || !kernelVals[0] || kernelVals[0].length !== t;
  if (sizeChanged) {
    inputVals = Array.from({ length: m }, () => Array.from({ length: n }, () => 0));
    kernelVals = Array.from({ length: s }, () => Array.from({ length: t }, () => 0));
    randomizeValues(); // 首次或尺寸变化时生成一组随机初值，便于观察
  }

  compute();
  buildInput();
  buildKernel();
  if (outH > 0 && outW > 0) {
    buildOutput();
    hideError();
  } else {
    showError('卷积输出尺寸非法：outH=' + outH + ', outW=' + outW + '。<br>请减小卷积核尺寸 / 步长，或改用填充方式。');
    outputMeta = [];
    outCX = 0;
  }
  resetAnimation();
  fitView();
  updateProgress();
}

function clampInt(v, lo, hi, def) {
  const n = parseInt(v, 10);
  if (isNaN(n)) return def;
  return Math.max(lo, Math.min(hi, n));
}

function randomizeValues() {
  for (let r = 0; r < inputVals.length; r++)
    for (let c = 0; c < inputVals[r].length; c++) inputVals[r][c] = Math.floor(Math.random() * 9) - 4;
  for (let r = 0; r < kernelVals.length; r++)
    for (let c = 0; c < kernelVals[r].length; c++) kernelVals[r][c] = Math.floor(Math.random() * 9) - 4;
}

function rerunAll() {
  compute();
  currentIndex = 0;
  resetAnimation();
  updateProgress();
}

/* ---------------- 点击编辑数值 ---------------- */
let dragStart = null;
let editing = null;

function topFaceUnit(w) {
  return boxHullUnit(w.x, w.y, w.z, w.sx, w.sy, w.sz);
}

function pointInPoly(ux, uy, pts) {
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const a = pts[i], b = pts[j];
    if ((a.uy > uy) !== (b.uy > uy) && ux < (b.ux - a.ux) * (uy - a.uy) / (b.uy - a.uy) + a.ux) inside = !inside;
  }
  return inside;
}

function pickHits(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  const mx = clientX - rect.left, my = clientY - rect.top;
  const ux = (mx - ox) / scale, uy = (my - oy) / scale;
  const out = [];
  for (const cell of clickCells) {
    const w = cell.kind === 'input' ? inputCellWorld(cell.pr, cell.pc) : kernelCellWorld(cell.ki, cell.kj);
    if (pointInPoly(ux, uy, topFaceUnit(w))) {
      out.push({ kind: cell.kind, pr: cell.pr, pc: cell.pc, ki: cell.ki, kj: cell.kj });
    }
  }
  return out;
}

function targetVal(t) {
  if (t.kind === 'input') return inputVals[t.pr - state.padT][t.pc - state.padL];
  return kernelVals[t.ki][t.kj];
}

function targetLabel(t) {
  if (t.kind === 'input') return '输入[' + (t.pr - state.padT) + '][' + (t.pc - state.padL) + '] = ' + fmt(targetVal(t));
  return '卷积核[' + t.ki + '][' + t.kj + '] = ' + fmt(targetVal(t));
}

function openEditorAt(hits, clientX, clientY) {
  if (!hits.length) return;
  const sel = $('editorTarget');
  sel.innerHTML = '';
  hits.forEach((t, idx) => {
    const opt = document.createElement('option');
    opt.value = String(idx);
    opt.textContent = targetLabel(t);
    sel.appendChild(opt);
  });
  sel.style.display = hits.length > 1 ? '' : 'none';
  sel.selectedIndex = 0;
  editing = hits[0];
  const inp = $('editorInput');
  inp.value = fmt(targetVal(editing));
  const el = $('editor');
  el.classList.add('show');
  el.style.left = Math.min(window.innerWidth - 210, Math.max(8, clientX + 12)) + 'px';
  el.style.top = Math.min(window.innerHeight - 150, Math.max(8, clientY + 12)) + 'px';
  inp.focus(); inp.select();
}

function closeEditor() { $('editor').classList.remove('show'); editing = null; }

function confirmEdit() {
  if (!editing) return;
  const v = parseFloat($('editorInput').value);
  const val = isNaN(v) ? 0 : v;
  if (editing.kind === 'input') inputVals[editing.pr - state.padT][editing.pc - state.padL] = val;
  else kernelVals[editing.ki][editing.kj] = val;
  closeEditor();
  rerunAll();
}

$('editorTarget').addEventListener('change', () => {
  const idx = parseInt($('editorTarget').value, 10);
  const hits = window.__lastEditorHits || [];
  editing = hits[idx];
  $('editorInput').value = fmt(targetVal(editing));
});

canvas.addEventListener('pointerdown', (e) => {
  dragStart = { x: e.clientX, y: e.clientY };
});
canvas.addEventListener('pointerup', (e) => {
  if (!dragStart) return;
  const moved = Math.hypot(e.clientX - dragStart.x, e.clientY - dragStart.y);
  dragStart = null;
  if (moved > 6 || e.button !== 0) return;
  if (isAnimating || isPlaying) return;
  const hits = pickHits(e.clientX, e.clientY);
  window.__lastEditorHits = hits;
  openEditorAt(hits, e.clientX, e.clientY);
});
canvas.addEventListener('pointermove', (e) => {
  const hits = pickHits(e.clientX, e.clientY);
  canvas.style.cursor = hits.length ? 'pointer' : 'default';
});

$('editorOk').addEventListener('click', confirmEdit);
$('editorCancel').addEventListener('click', closeEditor);
$('editorInput').addEventListener('keydown', (e) => { if (e.key === 'Enter') confirmEdit(); if (e.key === 'Escape') closeEditor(); });

/* ---------------- 控制面板事件 ---------------- */
$('btnPlay').addEventListener('click', () => {
  if (isPlaying) {
    isPlaying = false;
    updatePlayBtn();
    updateControls();
    return;
  }
  if (currentIndex >= positions.length) resetAnimation();
  isPlaying = true;
  isAnimating = false;
  updatePlayBtn();
  updateControls();
  playLoop(animGen);
});
$('btnNext').addEventListener('click', stepForward);
$('btnPrev').addEventListener('click', stepBackward);
$('btnReset').addEventListener('click', resetAnimation);
$('btnRandom').addEventListener('click', () => {
  if (isAnimating || isPlaying) return;
  randomizeValues();
  rerunAll();
});
$('btnClear').addEventListener('click', () => {
  if (isAnimating || isPlaying) return;
  for (let r = 0; r < inputVals.length; r++) for (let c = 0; c < inputVals[r].length; c++) inputVals[r][c] = 0;
  for (let r = 0; r < kernelVals.length; r++) for (let c = 0; c < kernelVals[r].length; c++) kernelVals[r][c] = 0;
  rerunAll();
});

['inM', 'inN', 'kerS', 'kerT', 'stride', 'padPH', 'padPW'].forEach((id) => {
  $(id).addEventListener('change', () => rebuild());
});
document.querySelectorAll('input[name="pad"]').forEach((radio) => {
  radio.addEventListener('change', () => {
    state.padMode = document.querySelector('input[name="pad"]:checked').value;
    $('customPadRow').style.display = state.padMode === 'custom' ? 'flex' : 'none';
    rebuild();
  });
});
$('speed').addEventListener('input', (e) => { state.speed = parseInt(e.target.value, 10); });

/* ---------------- GIF 导出 ---------------- */
function downloadGif(bytes) {
  const blob = new Blob([bytes], { type: 'image/gif' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'convolution-animation.gif';
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

async function exportGif() {
  const g = window.__gifenc;
  if (!g) { updateStatus('GIF 编码器未就绪，请刷新重试'); return; }
  if (isPlaying || isAnimating) return;

  const btn = $('btnGif');
  const prevSpeed = state.speed;
  btn.disabled = true;
  btn.textContent = '导出中…';
  updateStatus('正在录制动画并生成 GIF…');
  try {
    state.speed = parseInt($('gifSpeed').value, 10) || 100;
    resetAnimation();

    // 录制尺寸：裁剪到内容包围盒，去除大片空白背景，再按画质限制缩放
    const CAP = parseInt($('gifQuality').value, 10) || 760;
    const FRAME_MS = parseInt($('gifFps').value, 10) || 150;
    const box = contentScreenBox();
    const m = 8; // 内容边距
    const cssW = canvas.width / dpr;
    const cssH = canvas.height / dpr;
    const left = Math.max(0, box.left - m);
    const top = Math.max(0, box.top - m);
    const right = Math.min(cssW, box.left + box.width + m);
    const bottom = Math.min(cssH, box.top + box.height + m);
    const srcX = left * dpr, srcY = top * dpr;
    const srcW = Math.max(1, (right - left) * dpr);
    const srcH = Math.max(1, (bottom - top) * dpr);
    const k = Math.min(1, CAP / Math.max(srcW, srcH));
    const w = Math.max(1, Math.round(srcW * k));
    const h = Math.max(1, Math.round(srcH * k));
    const off = document.createElement('canvas');
    off.width = w;
    off.height = h;
    const offCtx = off.getContext('2d');
    const gif = g.GIFEncoder();
    let first = true;

    const capture = () => {
      offCtx.drawImage(canvas, srcX, srcY, srcW, srcH, 0, 0, w, h);
      const data = new Uint8Array(offCtx.getImageData(0, 0, w, h).data);
      const palette = g.quantize(data, 64);
      const index = g.applyPalette(data, palette);
      gif.writeFrame(index, w, h, { palette, delay: FRAME_MS, repeat: 0, first });
      first = false;
    };

    // 等待整段动画跑完
    const done = new Promise((resolve) => {
      const checker = setInterval(() => {
        if (!isAnimating && currentIndex >= positions.length) {
          clearInterval(checker);
          resolve();
        }
      }, 60);
    });

    isPlaying = true;
    isAnimating = false;
    updatePlayBtn();
    playLoop(animGen);

    const capTimer = setInterval(capture, FRAME_MS);
    await done;
    clearInterval(capTimer);
    capture(); // 补录最后一帧（完成态）

    gif.finish();
    downloadGif(gif.bytes());
    updateStatus('GIF 已导出（共 ' + currentIndex + ' 步）');
  } catch (err) {
    updateStatus('GIF 导出失败：' + err.message);
  } finally {
    state.speed = prevSpeed;
    btn.disabled = false;
    btn.textContent = '⬇ 导出 GIF';
    updateControls();
  }
}

$('btnGif').addEventListener('click', exportGif);

/* ---------------- 尺寸与渲染循环 ---------------- */
function onResize() {
  W = container.clientWidth;
  H = container.clientHeight;
  dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.style.width = W + 'px';
  canvas.style.height = H + 'px';
  canvas.width = Math.round(W * dpr);
  canvas.height = Math.round(H * dpr);
  fitView();
}
window.addEventListener('resize', onResize);

function renderLoop() {
  requestAnimationFrame(renderLoop);
  render();
}

/* ---------------- 启动 ---------------- */
onResize();
rebuild();
renderLoop();
