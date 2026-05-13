```javascript id="working-ai-pulpen-pensil"
// ======================================================
// AI DETEKSI PULPEN & PENSIL
// FILE : app.js
// ======================================================

// ======================
// KONFIGURASI
// ======================

const CONFIG = {
    modelPath: "best.onnx",

    // HARUS SAMA DENGAN ROBOFLOW
    labels: ["pulpen", "pensil"],

    confidence: 0.45
};

// ======================
// ELEMEN HTML
// ======================

const video = document.getElementById("webcam");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");

const processor = document.getElementById("processor");
const pctx = processor.getContext("2d");

const statusText = document.getElementById("status");
const startBtn = document.getElementById("btn-init");

let session = null;

const SIZE = 640;

// ======================================================
// TOMBOL START
// ======================================================

startBtn.addEventListener("click", async () => {

    try {

        statusText.innerText = "MEMUAT MODEL AI...";

        startBtn.disabled = true;

        // LOAD ONNX
        session = await ort.InferenceSession.create(
            CONFIG.modelPath
        );

        statusText.innerText =
            "MODEL BERHASIL DIMUAT";

        // START CAMERA
        await startCamera();

    } catch (err) {

        console.error(err);

        statusText.innerText =
            "GAGAL MEMUAT MODEL";

        alert(
            "best.onnx tidak ditemukan!"
        );
    }
});

// ======================================================
// START CAMERA
// ======================================================

async function startCamera() {

    try {

        const stream =
            await navigator.mediaDevices.getUserMedia({
                video: true,
                audio: false
            });

        video.srcObject = stream;

        await video.play();

        statusText.innerText =
            "AI AKTIF";

        startBtn.style.display = "none";

        detect();

    } catch (err) {

        console.error(err);

        statusText.innerText =
            "KAMERA GAGAL DIAKSES";

        alert(
            "Izinkan akses kamera!"
        );
    }
}

// ======================================================
// DETEKSI LOOP
// ======================================================

async function detect() {

    if (!session) return;

    // gambar kamera ke canvas processor
    pctx.drawImage(video, 0, 0, SIZE, SIZE);

    const imageData =
        pctx.getImageData(0, 0, SIZE, SIZE);

    // preprocess
    const input = preprocess(imageData);

    // tensor
    const tensor = new ort.Tensor(
        "float32",
        input,
        [1, 3, SIZE, SIZE]
    );

    // inference
    const output =
        await session.run({
            images: tensor
        });

    // ambil output pertama
    const outputName =
        session.outputNames[0];

    const data =
        output[outputName].data;

    // parse hasil
    const boxes =
        parseYOLOOutput(data);

    // gambar
    draw(boxes);

    requestAnimationFrame(detect);
}

// ======================================================
// PREPROCESS
// ======================================================

function preprocess(imageData) {

    const pixels = imageData.data;

    const input =
        new Float32Array(3 * SIZE * SIZE);

    for (let i = 0; i < SIZE * SIZE; i++) {

        input[i] =
            pixels[i * 4] / 255;

        input[i + SIZE * SIZE] =
            pixels[i * 4 + 1] / 255;

        input[i + SIZE * SIZE * 2] =
            pixels[i * 4 + 2] / 255;
    }

    return input;
}

// ======================================================
// PARSE OUTPUT YOLO
// ======================================================

function parseYOLOOutput(data) {

    const boxes = [];

    const rows = 8400;

    for (let i = 0; i < rows; i++) {

        const x = data[i];
        const y = data[i + rows];
        const w = data[i + rows * 2];
        const h = data[i + rows * 3];

        // class scores
        const scorePulpen =
            data[i + rows * 4];

        const scorePensil =
            data[i + rows * 5];

        let score = scorePulpen;
        let classId = 0;

        if (scorePensil > score) {

            score = scorePensil;
            classId = 1;
        }

        // threshold
        if (score > CONFIG.confidence) {

            boxes.push({

                x: x - w / 2,
                y: y - h / 2,
                w: w,
                h: h,

                score: score,

                classId: classId
            });
        }
    }

    return boxes;
}

// ======================================================
// DRAW BOX
// ======================================================

function draw(boxes) {

    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );

    const scaleX =
        canvas.width / SIZE;

    const scaleY =
        canvas.height / SIZE;

    boxes.forEach(box => {

        const x = box.x * scaleX;
        const y = box.y * scaleY;
        const w = box.w * scaleX;
        const h = box.h * scaleY;

        // warna
        const color =
            box.classId === 0
                ? "#00E5FF"
                : "#FFD600";

        // kotak
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;

        ctx.strokeRect(x, y, w, h);

        // label bg
        ctx.fillStyle = color;

        ctx.fillRect(
            x,
            y - 30,
            160,
            30
        );

        // text
        ctx.fillStyle = "#000";

        ctx.font =
            "bold 18px Arial";

        ctx.fillText(
            `${CONFIG.labels[box.classId]} ${(box.score * 100).toFixed(0)}%`,
            x + 5,
            y - 8
        );
    });
}
```
