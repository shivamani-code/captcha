(function () {
  const API_URL = "https://captcha-2-fix9.onrender.com/verify";

  const slider = document.getElementById("slider");
  const resultText = document.getElementById("result");

  let positions = [];
  let dragStartTime = null;
  let pageLoadTime = Date.now();
  let lastMoveTime = null;
  let idleTime = 0;
  let isDragging = false;

  slider.addEventListener("mousedown", () => {
    positions = [];
    idleTime = 0;
    dragStartTime = Date.now();
    lastMoveTime = dragStartTime;
    isDragging = true;
  });

  document.addEventListener("mousemove", (e) => {
    if (!isDragging) return;

    const now = Date.now();

    if (now - lastMoveTime > 120) {
      idleTime += now - lastMoveTime;
    }

    positions.push({
      x: e.clientX,
      y: e.clientY,
      t: now
    });

    lastMoveTime = now;
  });

  document.addEventListener("mouseup", async () => {
    if (!isDragging) return;
    isDragging = false;

    if (positions.length < 5) {
      resultText.innerText = "Verification failed. Try again.";
      return;
    }

    const payload = computeBehavioralFeatures();

    console.log("SmartCAPTCHA payload:", payload);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const data = await response.json();

      if (data.decision === "human") {
        window.location.href = "/success.html";
      } else {
        resultText.innerText = "Verification failed. Try again.";
      }

    } catch (err) {
      resultText.innerText = "Network error. Please retry.";
    }
  });

  function computeBehavioralFeatures() {
    let distance = 0;
    let directionChanges = 0;

    for (let i = 1; i < positions.length; i++) {
      const dx = positions[i].x - positions[i - 1].x;
      const dy = positions[i].y - positions[i - 1].y;
      distance += Math.sqrt(dx * dx + dy * dy);

      if (i > 1) {
        const pdx = positions[i - 1].x - positions[i - 2].x;
        const pdy = positions[i - 1].y - positions[i - 2].y;
        if (Math.sign(dx) !== Math.sign(pdx) || Math.sign(dy) !== Math.sign(pdy)) {
          directionChanges++;
        }
      }
    }

    const totalTime = (positions.at(-1).t - positions[0].t) / 1000;

    return {
      avg_mouse_speed: distance / totalTime,
      mouse_path_entropy: directionChanges / positions.length,
      click_delay: (positions[0].t - pageLoadTime) / 1000,
      task_completion_time: totalTime,
      idle_time: idleTime / 1000
    };
  }
})();
