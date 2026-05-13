```javascript id="9vr0ot"
// ======================================================================
// AI DETEKSI PULPEN & PENSIL
// ======================================================================

const CONFIG = {

    // File model ONNX
    modelPath: "./best.onnx",

    // Label HARUS sama dengan Roboflow
    labels: ["pulpen", "pensil"],

    // Confidence
    threshold: 0.45,

    // IoU
    iouThreshold: 0.4
};

// ======================================================================
// AMBIL ELEMEN HTML
// ======================================================================

const video = document.getElementById("webcam");

const overlay = document.getElementById("overlay");
const ctxOverlay = overlay.getContext("2d");

const processor = document.getElementById("processor");
const ctxProcessor = processor.getContext("2d", {
    willReadFrequently: true
});

const status = document.getElementById("status");
const initBtn = document.getElementById("btn-init");

let session;

const TARGET_SIZE = 640;

// ======================================================================
// TOMBOL AKTIFKAN AI
// ======================================================================

initBtn.addEventListener("click", async () => {

    try {

        initBtn.disabled = true;

        initBtn.innerText = "MEMUAT AI...";

        status.innerText = "STATUS : MEMUAT MODEL AI";

        // Load ONNX Runtime
        ort.env.wasm.wasmPaths =
            "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

        // Load model
        session = await ort.InferenceSession.create(
            CONFIG.modelPath,
            {
                executionProviders: ["webgl", "wasm"]
            }
        );

        status.innerText =
            "STATUS : MODEL BERHASIL DIMUAT";

        // Jalankan kamera
        await startCamera();

    } catch (error) {

        console.error(error);

        status.innerText =
            "ERROR : MODEL GAGAL DIMUAT";

        alert(
            "Model AI gagal dimuat. Pastikan best.onnx ada."
        );

        initBtn.disabled = false;
    }
});

// ======================================================================
// MENYALAKAN KAMERA
// ======================================================================

async function startCamera() {

    try {

        const stream =
            await navigator.mediaDevices.getUserMedia({
                video: true,
                audio: false
            });

        video.srcObject = stream;

        video.onloadedmetadata = () => {

            video.play();

            status.innerText =
                "STATUS : AI AKTIF";

            initBtn.style.display = "none";

            requestAnimationFrame(processFrame);
        };

    } catch (error) {

        console.error(error);

        status.innerText =
            "ERROR : KAMERA TIDAK BISA DIAKSES";

        alert(
            "Izinkan akses kamera terlebih dahulu."
        );
    }
}

// ======================================================================
// PROSES FRAME
// ======================================================================

async function processFrame() {

    if (!session) return;

    // Ambil gambar kamera
    ctxProcessor.drawImage(
        video,
        0,
        0,
        TARGET_SIZE,
        TARGET_SIZE
    );

    // Ambil data pixel
    const imageData =
        ctxProcessor.getImageData(
            0,
            0,
            TARGET_SIZE,
            TARGET_SIZE
        ).data;

    // Float32
    const float32Data =
        new Float32Array(
            3 * TARGET_SIZE * TARGET_SIZE
        );

    // RGB
    for (let i = 0; i < TARGET_SIZE * TARGET_SIZE; i++) {

        float32Data[i] =
            imageData[i * 4] / 255;

        float32Data[i + TARGET_SIZE * TARGET_SIZE] =
            imageData[i * 4 + 1] / 255;

        float32Data[i + 2 * TARGET_SIZE * TARGET_SIZE] =
            imageData[i * 4 + 2] / 255;
    }

    // Tensor
    const inputTensor =
        new ort.Tensor(
            "float32",
            float32Data,
            [1, 3, TARGET_SIZE, TARGET_SIZE]
        );

    // Run AI
    const results =
        await session.run({
            [session.inputNames[0]]: inputTensor
        });

    // Output
    const output =
        results[session.outputNames[0]].data;

    // Parsing
    const boxes = processOutput(output);

    // Draw
    drawBoxes(boxes);

    requestAnimationFrame(processFrame);
}

// ======================================================================
// MEMBACA OUTPUT YOLO
// ======================================================================

function processOutput(output) {

    const boxes = [];

    const numClasses =
        CONFIG.labels.length;

    const elements = 8400;

    for (let i = 0; i < elements; i++) {

        let maxScore = 0;
        let classId = 0;

        for (let c = 0; c < numClasses; c++) {

            const score =
                output[i + (4 + c) * elements];

            if (score > maxScore) {

                maxScore = score;
                classId = c;
            }
        }

        // Threshold
        if (maxScore > CONFIG.threshold) {

            let x = output[i];
            let y = output[i + elements];
            let w = output[i + elements * 2];
            let h = output[i + elements * 3];

            // Scale
            if (w <= 1.5) {

                x *= TARGET_SIZE;
                y *= TARGET_SIZE;
                w *= TARGET_SIZE;
                h *= TARGET_SIZE;
            }

            boxes.push({

                x: x - w / 2,
                y: y - h / 2,
                w: w,
                h: h,

                score: maxScore,

                classId: classId
            });
        }
    }

    return nonMaxSuppression(
        boxes,
        CONFIG.iouThreshold
    );
}

// ======================================================================
// IoU
// ======================================================================

function calculateIoU(box1, box2) {

    const x1 =
        Math.max(box1.x, box2.x);

    const y1 =
        Math.max(box1.y, box2.y);

    const x2 =
        Math.min(
            box1.x + box1.w,
            box2.x + box2.w
        );

    const y2 =
        Math.min(
            box1.y + box1.h,
            box2.y + box2.h
        );

    const intersection =
        Math.max(0, x2 - x1) *
        Math.max(0, y2 - y1);

    const union =
        (box1.w * box1.h) +
        (box2.w * box2.h) -
        intersection;

    return intersection / union;
}

// ======================================================================
// NMS
// ======================================================================

function nonMaxSuppression(boxes, threshold) {

    boxes.sort((a, b) =>
        b.score - a.score
    );

    const result = [];

    while (boxes.length > 0) {

        const current = boxes.shift();

        result.push(current);

        boxes = boxes.filter(box =>
            calculateIoU(current, box) < threshold
        );
    }

    return result;
}

// ======================================================================
// GAMBAR HASIL
// ======================================================================

function drawBoxes(boxes) {

    ctxOverlay.clearRect(
        0,
        0,
        overlay.width,
        overlay.height
    );

    boxes.forEach(box => {

        const scaleX =
            overlay.width / TARGET_SIZE;

        const scaleY =
            overlay.height / TARGET_SIZE;

        const x = box.x * scaleX;
        const y = box.y * scaleY;
        const w = box.w * scaleX;
        const h = box.h * scaleY;

        // Warna
        const color =
            box.classId === 0
                ? "#00E5FF"
                : "#FFD600";

        // Kotak
        ctxOverlay.strokeStyle = color;
        ctxOverlay.lineWidth = 3;

        ctxOverlay.strokeRect(
            x,
            y,
            w,
            h
        );

        // Background text
        ctxOverlay.fillStyle = color;

        ctxOverlay.fillRect(
            x,
            y - 28,
            150,
            28
        );

        // Text
        ctxOverlay.fillStyle = "#000";

        ctxOverlay.font =
            "bold 16px Arial";

        ctxOverlay.fillText(
            `${CONFIG.labels[box.classId]} ${(box.score * 100).toFixed(0)}%`,
            x + 5,
            y - 8
        );
    });
}
```
