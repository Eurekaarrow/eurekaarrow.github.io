/* ══════════════════════════════════════════════════════════════
   Loss landscape
   ──────────────────────────────────────────────────────────────
   An analytic loss surface rendered as level sets, with a
   momentum-SGD trajectory rolling from initialization toward a
   minimum. Contours are level sets of one scalar field, so they
   can never cross. The cursor adds a soft bump to the surface;
   scrolling advances the optimizer.
   ══════════════════════════════════════════════════════════════ */

(function () {
  var host = document.querySelector('.field');
  if (!host) return;

  var cv = document.createElement('canvas');
  host.appendChild(cv);
  var ctx = cv.getContext('2d', { alpha: true });

  var calm  = matchMedia('(prefers-reduced-motion: reduce)').matches;
  var hover = matchMedia('(hover: hover)').matches;

  /* ── domain ─────────────────────────────────────────────── */
  var X0 = -2.6, X1 = 3.4, Y0 = -4.0, Y1 = 4.0;
  var COLS = 92, ROWS = 130, LEVELS = 38;

  var W = 0, H = 0, dpr = 1;
  var vals = new Float32Array(COLS * ROWS);
  var levels = [];

  /* ── the surface ────────────────────────────────────────── */
  /* One minimum, its basin shaped by angular harmonics so the level sets
     stay nested and organic instead of pinching into saddles, plus a single
     shallow basin off to the side. Everything sits in the right half of the
     domain, since the container is masked from the right.                  */

  var MX = 2.45, MY = 0.95, YSQ = 0.72;

  /* the first-order term is what makes the level sets egg-shaped rather
     than elliptical: broad at the top, tapering below, flattening into
     near-parallel lines out on the left flank                          */
  var HARM = [ [1, 0.200, 1.5], [2, 0.140, 1.1], [3, 0.075, 2.6], [5, 0.032, 4.9] ];
  var SIDE = [ 1.00, -1.95, 1.00, 0.90, 0.24 ];   // cx cy sx sy depth

  var bx = 0, by = 0, bA = 0;                     // cursor bump

  function loss(x, y) {
    var dx = x - MX, dy = (y - MY) * YSQ;
    var r2 = dx * dx + dy * dy;
    var th = Math.atan2(dy, dx);
    var m = 1;
    for (var i = 0; i < HARM.length; i++) m += HARM[i][1] * Math.sin(HARM[i][0] * th + HARM[i][2]);
    var v = 0.17 * r2 * m;

    var ux = x - SIDE[0], uy = y - SIDE[1];
    v -= SIDE[4] * Math.exp(-(ux * ux / (2 * SIDE[2] * SIDE[2]) + uy * uy / (2 * SIDE[3] * SIDE[3])));

    if (bA !== 0) {
      var px = x - bx, py = y - by;
      v += bA * Math.exp(-(px * px + py * py) / 1.45);
    }
    return v;
  }

  function grad(x, y, g) {
    var h = 0.012;
    g[0] = (loss(x + h, y) - loss(x - h, y)) / (2 * h);
    g[1] = (loss(x, y + h) - loss(x, y - h)) / (2 * h);
  }

  /* ── sample the field onto the grid ─────────────────────── */
  function sample() {
    var lo = Infinity, hi = -Infinity, k = 0;
    for (var j = 0; j < ROWS; j++) {
      var y = Y0 + (Y1 - Y0) * j / (ROWS - 1);
      for (var i = 0; i < COLS; i++) {
        var x = X0 + (X1 - X0) * i / (COLS - 1);
        var v = loss(x, y);
        vals[k++] = v;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
    if (levels.length === 0) {
      for (var n = 0; n < LEVELS; n++) {
        levels.push(lo + (hi - lo) * Math.pow((n + 0.5) / LEVELS, 1.50));  // denser near the minimum
      }
    }
  }

  /* ── marching squares ───────────────────────────────────── */
  function march(level, seg) {
    seg.length = 0;
    var W1 = COLS - 1, H1 = ROWS - 1;
    for (var j = 0; j < H1; j++) {
      for (var i = 0; i < W1; i++) {
        var o = j * COLS + i;
        var a = vals[o], b = vals[o + 1], c = vals[o + COLS + 1], d = vals[o + COLS];
        var m = (a > level ? 8 : 0) | (b > level ? 4 : 0) | (c > level ? 2 : 0) | (d > level ? 1 : 0);
        if (m === 0 || m === 15) continue;

        var tx = i + (level - a) / (b - a),      ty = j;
        var rx = i + 1,                          ry = j + (level - b) / (c - b);
        var bxp = i + (level - d) / (c - d),     byp = j + 1;
        var lx = i,                              ly = j + (level - a) / (d - a);

        switch (m) {
          case 1:  case 14: seg.push(lx, ly, bxp, byp); break;
          case 2:  case 13: seg.push(bxp, byp, rx, ry); break;
          case 3:  case 12: seg.push(lx, ly, rx, ry);   break;
          case 4:  case 11: seg.push(tx, ty, rx, ry);   break;
          case 6:  case 9:  seg.push(tx, ty, bxp, byp); break;
          case 7:  case 8:  seg.push(lx, ly, tx, ty);   break;
          case 5:
            if ((a + b + c + d) / 4 > level) { seg.push(lx, ly, tx, ty); seg.push(bxp, byp, rx, ry); }
            else                             { seg.push(lx, ly, bxp, byp); seg.push(tx, ty, rx, ry); }
            break;
          case 10:
            if ((a + b + c + d) / 4 > level) { seg.push(lx, ly, bxp, byp); seg.push(tx, ty, rx, ry); }
            else                             { seg.push(lx, ly, tx, ty); seg.push(bxp, byp, rx, ry); }
            break;
        }
      }
    }
  }

  /* ── momentum SGD with a cosine schedule ────────────────── */
  /* The decaying step size is what makes it settle instead of orbiting the
     minimum forever, so the tail carries almost no arc length. The path is
     then resampled by distance travelled, which is what scroll progress
     tracks — the head keeps moving all the way down the page.             */

  var STEPS = 330;                        // render points, evenly spaced by distance
  var RAW   = 220;                        // raw integration steps
  var WARM  = 110;                        // steps over which the step size decays to zero
  var LR0   = 0.058, MOM = 0.885;

  var path = new Float32Array(STEPS * 2);
  var rawX = new Float64Array(RAW);
  var rawY = new Float64Array(RAW);
  var cum  = new Float64Array(RAW);
  var g    = [0, 0];

  function descend() {
    var x = 1.10, y = -3.55, vx = 0, vy = 0, s;

    for (s = 0; s < RAW; s++) {
      rawX[s] = x; rawY[s] = y;
      var lr = LR0 * 0.5 * (1 + Math.cos(Math.PI * Math.min(1, s / WARM)));
      grad(x, y, g);
      vx = MOM * vx - lr * g[0];
      vy = MOM * vy - lr * g[1];
      x += vx; y += vy;
    }

    cum[0] = 0;
    for (s = 1; s < RAW; s++) {
      var dx = rawX[s] - rawX[s-1], dy = rawY[s] - rawY[s-1];
      cum[s] = cum[s-1] + Math.sqrt(dx * dx + dy * dy);
    }
    var total = cum[RAW-1];

    if (total < 1e-6) {
      for (var r = 0; r < STEPS; r++) { path[r*2] = rawX[0]; path[r*2+1] = rawY[0]; }
      return;
    }

    var step = total / (STEPS - 1), j = 0;
    for (var k = 0; k < STEPS; k++) {
      var target = step * k;
      while (j < RAW - 2 && cum[j + 1] < target) j++;
      var seg = cum[j + 1] - cum[j];
      var f = seg > 1e-9 ? (target - cum[j]) / seg : 0;
      path[k*2]     = rawX[j] + (rawX[j+1] - rawX[j]) * f;
      path[k*2 + 1] = rawY[j] + (rawY[j+1] - rawY[j]) * f;
    }
  }

  /* ── draw ───────────────────────────────────────────────── */
  var seg = [];
  var progress = 0;

  function sx(gx) { return gx / (COLS - 1) * W; }
  function sy(gy) { return gy / (ROWS - 1) * H; }
  function px(x)  { return (x - X0) / (X1 - X0) * W; }
  function py(y)  { return (y - Y0) / (Y1 - Y0) * H; }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    ctx.lineCap = 'round';

    for (var n = 0; n < levels.length; n++) {
      march(levels[n], seg);
      var t = n / (levels.length - 1);
      ctx.strokeStyle = 'rgba(123,90,166,' + (0.60 - 0.38 * t).toFixed(3) + ')';
      ctx.lineWidth = 0.95 - 0.33 * t;
      ctx.beginPath();
      for (var k = 0; k < seg.length; k += 4) {
        ctx.moveTo(sx(seg[k]),     sy(seg[k + 1]));
        ctx.lineTo(sx(seg[k + 2]), sy(seg[k + 3]));
      }
      ctx.stroke();
    }

    /* trajectory — drawn as far as the reader has scrolled */
    var end = Math.max(2, Math.round(STEPS * progress));
    ctx.strokeStyle = 'rgba(78,42,132,.52)';
    ctx.lineWidth = 1.6;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(px(path[0]), py(path[1]));
    for (var s = 1; s < end; s++) ctx.lineTo(px(path[s * 2]), py(path[s * 2 + 1]));
    ctx.stroke();

    var hx = px(path[(end - 1) * 2]), hy = py(path[(end - 1) * 2 + 1]);
    ctx.fillStyle = 'rgba(78,42,132,.78)';
    ctx.beginPath();
    ctx.arc(hx, hy, 3.1, 0, 6.2832);
    ctx.fill();
  }

  /* ── loop ───────────────────────────────────────────────── */
  var tbx = 0, tby = 0, tbA = 0, running = false;

  function frame() {
    var moved = false;
    bx += (tbx - bx) * 0.055;
    by += (tby - by) * 0.055;
    bA += (tbA - bA) * 0.055;
    if (Math.abs(tbx - bx) > 0.004 || Math.abs(tby - by) > 0.004 || Math.abs(tbA - bA) > 0.002) moved = true;

    sample(); descend(); draw();

    if (moved) requestAnimationFrame(frame);
    else running = false;
  }
  function kick() { if (!running) { running = true; requestAnimationFrame(frame); } }

  function resize() {
    var r = host.getBoundingClientRect();
    W = Math.max(1, r.width); H = Math.max(1, r.height);
    dpr = Math.min(2, window.devicePixelRatio || 1);
    cv.width = W * dpr; cv.height = H * dpr;
    cv.style.width = W + 'px'; cv.style.height = H + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    sample(); descend(); draw();
  }

  function scroll() {
    var max = document.documentElement.scrollHeight - innerHeight;
    var p = max > 0 ? Math.min(1, Math.max(0, scrollY / max)) : 1;
    if (Math.abs(p - progress) < 0.002) return;
    progress = p;
    if (!running) draw();
  }

  if (hover && !calm) {
    addEventListener('pointermove', function (e) {
      var r = host.getBoundingClientRect();
      tbx = X0 + (X1 - X0) * ((e.clientX - r.left) / r.width);
      tby = Y0 + (Y1 - Y0) * ((e.clientY - r.top)  / r.height);
      tbA = 0.44;
      kick();
    }, { passive: true });

    addEventListener('pointerleave', function () { tbA = 0; kick(); }, { passive: true });
  }

  addEventListener('resize', resize, { passive: true });
  addEventListener('scroll', scroll, { passive: true });

  progress = calm ? 1 : 0.06;
  resize();
  scroll();
})();
