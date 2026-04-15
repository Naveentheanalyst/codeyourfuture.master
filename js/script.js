document.addEventListener('DOMContentLoaded', () => {
    // Mobile menu toggle
    const hamburger = document.querySelector('.hamburger');
    const navLinks = document.querySelector('.nav-links');

    hamburger.addEventListener('click', () => {
        if (navLinks.style.display === 'flex') {
            navLinks.style.display = 'none';
        } else {
            navLinks.style.display = 'flex';
            navLinks.style.flexDirection = 'column';
            navLinks.style.position = 'absolute';
            navLinks.style.top = '70px';
            navLinks.style.left = '0';
            navLinks.style.width = '100%';
            navLinks.style.background = 'rgba(10, 10, 10, 0.95)';
            navLinks.style.padding = '20px';
            navLinks.style.borderRadius = '12px';
            navLinks.style.border = '1px solid var(--border-color)';
        }
    });

    // Reset inline styles on window resize
    window.addEventListener('resize', () => {
        if (window.innerWidth > 992) {
            navLinks.style.display = '';
            navLinks.style.flexDirection = '';
            navLinks.style.position = '';
            navLinks.style.background = '';
            navLinks.style.padding = '';
            navLinks.style.border = '';
        } else {
            navLinks.style.display = 'none';
        }
    });

    // Smooth scroll for anchor links — offset for fixed navbar
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const href = this.getAttribute('href');
            if (href === '#') return;
            e.preventDefault();
            const target = document.querySelector(href);
            if (target) {
                const navbarHeight = document.querySelector('.navbar')?.offsetHeight || 80;
                const top = target.getBoundingClientRect().top + window.scrollY - navbarHeight - 20;
                window.scrollTo({ top, behavior: 'smooth' });
                // Close mobile menu on click
                if (window.innerWidth <= 992) {
                    navLinks.style.display = 'none';
                }
            }
        });
    });
});

// Curriculum Tab Switcher
function switchTab(btn, tabId) {
    // Remove active from all buttons and panels
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));

    // Activate clicked
    btn.classList.add('active');
    document.getElementById('tab-' + tabId).classList.add('active');
}

// ─── Hero Canvas: Neural Net + Gradient Descent + Linear Regression + Bayes ─
(function () {
    const canvas = document.getElementById('neural-bg');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    let W, H, tick = 0;

    /* ── colours ── */
    const C = {
        accent : 'rgba(234,255,0,',
        cyan   : 'rgba(0,240,255,',
        purple : 'rgba(168,85,247,',
        white  : 'rgba(255,255,255,'
    };

    /* ═══════════════════════════════════════
       1. NEURAL NETWORK  (full canvas)
    ═══════════════════════════════════════ */
    const NODES = 52, CONN_DIST = 150;
    let nodes = [];

    function initNodes() {
        nodes = [];
        const cols = [C.accent, C.cyan, C.purple];
        for (let i = 0; i < NODES; i++) {
            nodes.push({
                x: Math.random() * W, y: Math.random() * H,
                vx: (Math.random() - 0.5) * 0.4,
                vy: (Math.random() - 0.5) * 0.4,
                r: Math.random() * 2 + 1.2,
                col: cols[Math.floor(Math.random() * 3)],
                phase: Math.random() * Math.PI * 2
            });
        }
    }

    function drawNeural() {
        // Connections
        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                const a = nodes[i], b = nodes[j];
                const d = Math.hypot(a.x - b.x, a.y - b.y);
                if (d < CONN_DIST) {
                    const alp = (1 - d / CONN_DIST) * 0.22;
                    const g = ctx.createLinearGradient(a.x, a.y, b.x, b.y);
                    g.addColorStop(0, a.col + alp + ')');
                    g.addColorStop(1, b.col + alp + ')');
                    ctx.beginPath();
                    ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y);
                    ctx.strokeStyle = g; ctx.lineWidth = 0.7; ctx.stroke();
                }
            }
        }
        // Nodes
        nodes.forEach(n => {
            n.phase += 0.007;
            const glow = 0.5 + 0.5 * Math.sin(n.phase);
            const rg = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, n.r * 5);
            rg.addColorStop(0, n.col + glow * 0.4 + ')');
            rg.addColorStop(1, n.col + '0)');
            ctx.beginPath(); ctx.arc(n.x, n.y, n.r * 5, 0, Math.PI * 2);
            ctx.fillStyle = rg; ctx.fill();
            ctx.beginPath(); ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
            ctx.fillStyle = n.col + glow + ')'; ctx.fill();
            n.x += n.vx; n.y += n.vy;
            if (n.x < 0 || n.x > W) n.vx *= -1;
            if (n.y < 0 || n.y > H) n.vy *= -1;
        });
    }

    /* ═══════════════════════════════════════
       2. GRADIENT DESCENT  (bottom-left zone)
    ═══════════════════════════════════════ */
    let theta = 2.5;
    const LR   = 0.018;
    const jFn  = t => t * t;
    const dJ   = t => 2 * t;
    const gdHistory = [];

    // Map θ and J into the bottom-left quadrant (30% W, bottom 38% H)
    function gdX(t)  { return (t / 3.5 + 1) * 0.5 * W * 0.3 + W * 0.02; }
    function gdY(j)  { return H - 0.06 * H - (j / 12.25) * H * 0.30; }

    function drawGD() {
        const qW = W * 0.30, qH = H * 0.38;
        const ox = W * 0.02, oy = H - qH - H * 0.04;

        // Axis labels
        ctx.fillStyle = C.white + '0.18)';
        ctx.font = `${Math.max(10, W * 0.011)}px monospace`;
        ctx.fillText('J(θ) = θ²', ox + 4, oy + 14);
        ctx.fillText('Gradient Descent', ox + 4, oy + 28);

        // Parabola
        ctx.beginPath();
        for (let t = -3.5; t <= 3.5; t += 0.06) {
            const x = gdX(t), y = gdY(jFn(t));
            t === -3.5 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.strokeStyle = C.accent + '0.3)'; ctx.lineWidth = 1.5; ctx.stroke();

        // Trail
        if (gdHistory.length > 1) {
            ctx.beginPath();
            gdHistory.forEach((th, i) => {
                i === 0
                    ? ctx.moveTo(gdX(th), gdY(jFn(th)))
                    : ctx.lineTo(gdX(th), gdY(jFn(th)));
            });
            ctx.strokeStyle = C.accent + '0.1)'; ctx.lineWidth = 1; ctx.stroke();
        }

        // Gradient arrow
        const g = dJ(theta);
        const x1 = gdX(theta), y1 = gdY(jFn(theta));
        const x2 = x1 - g * 18, y2 = y1 + g * g * 2.5;
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
        ctx.strokeStyle = C.cyan + '0.35)'; ctx.lineWidth = 1.2; ctx.stroke();
        const ang = Math.atan2(y2 - y1, x2 - x1);
        ctx.beginPath();
        ctx.moveTo(x2, y2);
        ctx.lineTo(x2 - 6 * Math.cos(ang - 0.4), y2 - 6 * Math.sin(ang - 0.4));
        ctx.lineTo(x2 - 6 * Math.cos(ang + 0.4), y2 - 6 * Math.sin(ang + 0.4));
        ctx.closePath(); ctx.fillStyle = C.cyan + '0.35)'; ctx.fill();

        // Ball
        const bx = gdX(theta), by = gdY(jFn(theta));
        const rg = ctx.createRadialGradient(bx, by, 0, bx, by, 10);
        rg.addColorStop(0, C.accent + '0.7)'); rg.addColorStop(1, C.accent + '0)');
        ctx.beginPath(); ctx.arc(bx, by, 10, 0, Math.PI * 2);
        ctx.fillStyle = rg; ctx.fill();
        ctx.beginPath(); ctx.arc(bx, by, 3.5, 0, Math.PI * 2);
        ctx.fillStyle = C.accent + '1)'; ctx.fill();

        // Live readout
        ctx.fillStyle = C.white + '0.12)';
        ctx.font = `${Math.max(9, W * 0.009)}px monospace`;
        ctx.fillText(`θ=${theta.toFixed(3)}  ∇=${dJ(theta).toFixed(3)}`, ox + 4, H - H * 0.015);

        // Update
        if (Math.abs(theta) < 0.04) {
            theta = (Math.random() > 0.5 ? 1 : -1) * (2.1 + Math.random());
            gdHistory.length = 0;
        }
        theta -= LR * dJ(theta);
        gdHistory.push(theta);
        if (gdHistory.length > 55) gdHistory.shift();
    }

    /* ═══════════════════════════════════════
       3. LINEAR REGRESSION  y = mx + c  (bottom-right)
    ═══════════════════════════════════════ */
    let lrPoints = [], lrM = 0, lrB = 0.5, lrPhase = 0, lrTick = 0;
    const tM = 1.5, tB = 0.08;

    function makeLRPoints() {
        lrPoints = [];
        for (let i = 0; i < 36; i++) {
            const x = Math.random();
            lrPoints.push({ x, y: tM * x + tB + (Math.random() - 0.5) * 0.38 });
        }
    }

    // Map [0,1] into bottom-right quadrant (right 32%, bottom 40%)
    function lrCx(v) { return W * 0.662 + v * W * 0.31; }
    function lrCy(v) { return H - 0.06 * H - v * H * 0.32; }

    function drawLR() {
        // Labels
        ctx.fillStyle = C.white + '0.18)';
        ctx.font = `${Math.max(10, W * 0.011)}px monospace`;
        ctx.fillText('ŷ = mx + c', W * 0.668, H - H * 0.38);
        ctx.fillText('Linear Regression', W * 0.668, H - H * 0.365);

        // Points + residuals
        lrPoints.forEach(p => {
            const predY = lrM * p.x + lrB;
            ctx.beginPath();
            ctx.moveTo(lrCx(p.x), lrCy(p.y));
            ctx.lineTo(lrCx(p.x), lrCy(predY));
            ctx.strokeStyle = C.purple + '0.18)'; ctx.lineWidth = 0.9; ctx.stroke();

            ctx.beginPath();
            ctx.arc(lrCx(p.x), lrCy(p.y), 2.8, 0, Math.PI * 2);
            ctx.fillStyle = C.cyan + '0.55)'; ctx.fill();
        });

        // Regression line
        ctx.beginPath();
        ctx.moveTo(lrCx(0), lrCy(lrM * 0 + lrB));
        ctx.lineTo(lrCx(1), lrCy(lrM * 1 + lrB));
        ctx.strokeStyle = C.accent + '0.45)'; ctx.lineWidth = 2; ctx.stroke();

        // Live values
        ctx.fillStyle = C.white + '0.12)';
        ctx.font = `${Math.max(9, W * 0.009)}px monospace`;
        ctx.fillText(`m=${lrM.toFixed(2)}  c=${lrB.toFixed(2)}`, W * 0.668, H - H * 0.018);

        // Update phase
        if (lrPhase === 0) {
            lrM += (tM - lrM) * 0.014; lrB += (tB - lrB) * 0.014;
            if (Math.abs(tM - lrM) < 0.006) { lrPhase = 1; lrTick = 0; }
        } else if (lrPhase === 1) {
            if (++lrTick > 130) lrPhase = 2;
        } else {
            makeLRPoints();
            lrM = (Math.random() - 0.5) * 2.4; lrB = Math.random() * 1.4;
            lrPhase = 0;
        }
    }

    /* ═══════════════════════════════════════
       4. BAYES' THEOREM  (top-right corner)
    ═══════════════════════════════════════ */
    let bayesFade = 0, bayesDir = 1, bayesPhase = 0;

    function drawBayes() {
        // Fade cycle
        bayesFade += 0.004 * bayesDir;
        if (bayesFade >= 1) { bayesFade = 1; bayesDir = -1; }
        if (bayesFade <= 0) { bayesFade = 0; bayesDir = 1; }

        const alpha = bayesFade * 0.28;
        const fs = Math.max(11, W * 0.012);
        const ox = W * 0.62, oy = H * 0.07;

        // Title
        ctx.fillStyle = C.white + (alpha * 0.85) + ')';
        ctx.font = `bold ${fs}px Outfit, sans-serif`;
        ctx.fillText("Bayes' Theorem", ox, oy);

        // Numerator: P(B|A) · P(A)
        ctx.fillStyle = C.cyan + alpha + ')';
        ctx.font = `${fs * 1.1}px serif`;
        ctx.fillText('P(B|A) · P(A)', ox + 8, oy + fs * 2.2);

        // Fraction bar
        ctx.beginPath();
        ctx.moveTo(ox + 4, oy + fs * 2.5);
        ctx.lineTo(ox + fs * 9.5, oy + fs * 2.5);
        ctx.strokeStyle = C.accent + alpha + ')'; ctx.lineWidth = 1.3; ctx.stroke();

        // Denominator: P(B)
        ctx.fillStyle = C.purple + alpha + ')';
        ctx.font = `${fs * 1.1}px serif`;
        ctx.fillText('P(B)', ox + fs * 3.2, oy + fs * 3.8);

        // Equals sign and result
        ctx.fillStyle = C.white + (alpha * 0.9) + ')';
        ctx.font = `${fs * 1.15}px serif`;
        ctx.fillText('P(A|B) =', ox - fs * 5.8, oy + fs * 2.9);

        // Annotation
        ctx.fillStyle = C.white + (alpha * 0.4) + ')';
        ctx.font = `${Math.max(9, fs * 0.78)}px monospace`;
        ctx.fillText('posterior  ∝  likelihood × prior', ox - fs * 5.5, oy + fs * 5.2);
    }

    /* ═══════════════════════════════════════
       5. FLOATING EQUATIONS  (scattered)
    ═══════════════════════════════════════ */
    const equations = [
        { text: 'y = mx + c',        x: 0.08,  y: 0.12,  speed: 0.0011, phase: 0.0 },
        { text: 'σ(z) = 1/(1+e⁻ᶻ)', x: 0.72, y: 0.52,  speed: 0.0014, phase: 1.2 },
        { text: '∇J(θ) = XᵀXθ − Xᵀy', x: 0.04, y: 0.58,  speed: 0.0009, phase: 2.4 },
        { text: 'RSS = Σ(yᵢ − ŷᵢ)²', x: 0.35,  y: 0.08,  speed: 0.0013, phase: 0.8 },
        { text: 'R² = 1 − SS_res/SS_tot', x: 0.55, y: 0.78, speed: 0.0010, phase: 3.1 },
    ];

    function drawEquations() {
        equations.forEach(eq => {
            eq.phase += eq.speed * 55;
            const alpha = (0.5 + 0.5 * Math.sin(eq.phase)) * 0.15;
            ctx.fillStyle = C.white + alpha + ')';
            ctx.font = `${Math.max(10, W * 0.011)}px monospace`;
            ctx.fillText(eq.text, eq.x * W, eq.y * H);
        });
    }

    /* ═══════════════════════════════════════
       MAIN LOOP
    ═══════════════════════════════════════ */
    function resize() {
        W = canvas.width  = canvas.offsetWidth;
        H = canvas.height = canvas.offsetHeight;
        initNodes(); makeLRPoints();
    }

    function loop() {
        ctx.clearRect(0, 0, W, H);
        drawNeural();
        drawGD();
        drawLR();
        drawBayes();
        drawEquations();
        tick++;
        requestAnimationFrame(loop);
    }

    window.addEventListener('resize', resize);
    resize();
    loop();
})();
