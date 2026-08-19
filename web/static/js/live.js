"use strict";

(function () {
  const $ = (sel) => document.querySelector(sel);

  let ws = null;
  let mode = "idle";
  let lastRequest = null;
  let webcamStream = null;
  let sendTimer = null;
  let reconnecting = false;
  let reconnectAttempts = 0;
  const sendQueue = [];

  const video = $("#webcam-video");
  const canvas = $("#webcam-canvas");
  const liveImage = $("#live-image");
  const liveHud = $("#live-hud");
  const liveStage = $("#live-stage");
  const status = $("#live-status");
  const liveConf = $("#live-conf");
  const liveConfVal = $("#live-conf-val");

  const btnWebcamStart = $("#webcam-start");
  const btnWebcamStop = $("#webcam-stop");
  const btnDcStart = $("#dc-start");
  const btnDcStop = $("#dc-stop");

  function setStatus(msg, cls) {
    status.className = "status" + (cls ? " " + cls : "");
    status.textContent = msg;
  }

  function setButtons(disabled) {
    [btnWebcamStart, btnDcStart].forEach((b) => (b.disabled = disabled));
  }

  function openSocket() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const sock = new WebSocket(`${proto}://${location.host}/ws/live`);
    sock.binaryType = "blob";
    sock.onopen = () => {
      ws = sock;
      reconnecting = false;
      reconnectAttempts = 0;
      setButtons(false);
      setStatus(mode !== "idle" ? "Connected." : "Connected. Ready.");
      while (sendQueue.length) sock.send(sendQueue.shift());
      if (lastRequest) send(lastRequest);
    };
    sock.onmessage = onMessage;
    sock.onclose = () => {
      if (ws === sock) ws = null;
      stopWebcamStream();
      if (mode !== "idle" && reconnectAttempts < 1) {
        reconnectAttempts += 1;
        reconnecting = true;
        setStatus("Connection lost — reconnecting…", "err");
        setTimeout(openSocket, 500);
      } else {
        reconnectAttempts = 0;
        mode = "idle";
        lastRequest = null;
        resetUI();
        setStatus("Disconnected.", "err");
      }
    };
    sock.onerror = () => {
      setStatus("WebSocket error.", "err");
    };
  }

  function ensureSocket() {
    if (ws && ws.readyState === WebSocket.OPEN) return;
    if (ws && (ws.readyState === WebSocket.CONNECTING || reconnecting)) return;
    setButtons(true);
    setStatus("Connecting…");
    openSocket();
  }

  function onMessage(event) {
    if (typeof event.data === "string") {
      const msg = JSON.parse(event.data);
      if (msg.type === "detections") {
        liveHud.textContent = msg.count
          ? `${msg.count} object(s) detected`
          : "no detections";
      } else if (msg.type === "error") {
        setStatus(msg.message, "err");
      } else if (msg.type === "mode") {
        mode = msg.mode;
      }
      return;
    }
    const url = URL.createObjectURL(event.data);
    liveImage.src = url;
    const old = liveImage.dataset.url;
    if (old) URL.revokeObjectURL(old);
    liveImage.dataset.url = url;
  }

  function send(data) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(data);
    } else if (typeof data === "string") {
      sendQueue.push(data);
    }
  }

  function sendJSON(obj) {
    send(JSON.stringify(obj));
  }

  function stopWebcamStream() {
    if (sendTimer) { clearInterval(sendTimer); sendTimer = null; }
    if (webcamStream) {
      webcamStream.getTracks().forEach((t) => t.stop());
      webcamStream = null;
    }
  }

  function resetUI() {
    stopWebcamStream();
    btnWebcamStart.classList.remove("hidden");
    btnWebcamStop.classList.add("hidden");
    btnDcStart.classList.remove("hidden");
    btnDcStop.classList.add("hidden");
    setButtons(false);
  }

  async function startWebcam() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatus("Camera unavailable — use https or localhost.", "err");
      return;
    }
    ensureSocket();
    try {
      webcamStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640 },
        audio: false,
      });
    } catch (err) {
      setStatus("Camera access denied or unavailable.", "err");
      return;
    }
    video.srcObject = webcamStream;
    await video.play();
    lastRequest = { type: "webcam" };
    sendJSON(lastRequest);
    mode = "webcam";
    btnWebcamStart.classList.add("hidden");
    btnWebcamStop.classList.remove("hidden");
    liveStage.classList.remove("hidden");
    setStatus("Webcam streaming…");

    const WIDTH = 640;
    sendTimer = setInterval(() => {
      if (video.readyState < 2) return;
      const scale = WIDTH / video.videoWidth;
      canvas.width = WIDTH;
      canvas.height = Math.round(video.videoHeight * scale);
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(
        (blob) => {
          if (blob && ws && ws.readyState === WebSocket.OPEN) ws.send(blob);
        },
        "image/jpeg",
        0.7
      );
    }, 100);
  }

  function stopWebcam() {
    lastRequest = null;
    sendJSON({ type: "stop" });
    mode = "idle";
    resetUI();
    liveStage.classList.add("hidden");
    setStatus("Stopped.");
  }

  function startDroidcam() {
    const ip = $("#dc-ip").value.trim();
    const port = $("#dc-port").value.trim() || "4747";
    if (!ip) {
      setStatus("Enter the phone IP address.", "err");
      return;
    }
    ensureSocket();
    lastRequest = { type: "droidcam", ip, port: Number(port) };
    sendJSON(lastRequest);
    mode = "droidcam";
    btnDcStart.classList.add("hidden");
    btnDcStop.classList.remove("hidden");
    liveStage.classList.remove("hidden");
    setStatus(`Connecting to DroidCam at ${ip}:${port}…`);
  }

  function stopDroidcam() {
    lastRequest = null;
    sendJSON({ type: "stop" });
    mode = "idle";
    resetUI();
    liveStage.classList.add("hidden");
    setStatus("Stopped.");
  }

  liveConf.addEventListener("input", () => {
    liveConfVal.textContent = Number(liveConf.value).toFixed(2);
    sendJSON({ type: "params", conf: Number(liveConf.value) });
  });

  btnWebcamStart.addEventListener("click", startWebcam);
  btnWebcamStop.addEventListener("click", stopWebcam);
  btnDcStart.addEventListener("click", startDroidcam);
  btnDcStop.addEventListener("click", stopDroidcam);

  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      const m = tab.dataset.mode;
      $("#panel-webcam").classList.toggle("hidden", m !== "webcam");
      $("#panel-droidcam").classList.toggle("hidden", m !== "droidcam");
    });
  });
})();
