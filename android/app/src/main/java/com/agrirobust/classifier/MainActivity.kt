package com.agrirobust.classifier

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.exifinterface.media.ExifInterface
import java.io.File
import java.io.InputStream

class MainActivity : AppCompatActivity() {

    private lateinit var classifier: AgriClassifier
    private lateinit var imageView: ImageView
    private lateinit var placeholderContainer: View
    private lateinit var tvResultClass: TextView
    private lateinit var tvConfidence: TextView
    private lateinit var tvAbstentionStatus: TextView
    private lateinit var tvLatency: TextView
    private lateinit var tvTopK: TextView

    private var currentPhotoUri: Uri? = null

    // Gallery Picker Launcher
    private val selectImageLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let { handleImageUri(it) }
    }

    // Camera Capture Launcher
    private val takePictureLauncher = registerForActivityResult(ActivityResultContracts.TakePicture()) { success: Boolean ->
        if (success) {
            currentPhotoUri?.let { handleImageUri(it) }
        }
    }

    // Camera Permission Launcher
    private val requestCameraPermissionLauncher = registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted: Boolean ->
        if (isGranted) {
            launchCamera()
        } else {
            Toast.makeText(this, "Camera permission is required to take photos", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        classifier = AgriClassifier(this)

        imageView = findViewById(R.id.imageView)
        placeholderContainer = findViewById(R.id.placeholderContainer)
        tvResultClass = findViewById(R.id.tvResultClass)
        tvConfidence = findViewById(R.id.tvConfidence)
        tvAbstentionStatus = findViewById(R.id.tvAbstentionStatus)
        tvLatency = findViewById(R.id.tvLatency)
        tvTopK = findViewById(R.id.tvTopK)

        findViewById<Button>(R.id.btnTakePhoto).setOnClickListener {
            requestCameraPermissionLauncher.launch(android.Manifest.permission.CAMERA)
        }

        findViewById<Button>(R.id.btnSelectImage).setOnClickListener {
            selectImageLauncher.launch("image/*")
        }

        findViewById<Button>(R.id.btnRunBenchmark).setOnClickListener {
            runOnDeviceBenchmark()
        }
    }

    private fun launchCamera() {
        try {
            val photoFile = File(cacheDir, "temp_capture_${System.currentTimeMillis()}.jpg")
            currentPhotoUri = FileProvider.getUriForFile(
                this,
                "${applicationContext.packageName}.fileprovider",
                photoFile
            )
            takePictureLauncher.launch(currentPhotoUri)
        } catch (e: Exception) {
            Toast.makeText(this, "Error starting camera: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun handleImageUri(uri: Uri) {
        try {
            val inputStream: InputStream? = contentResolver.openInputStream(uri)
            val rawBitmap = BitmapFactory.decodeStream(inputStream)
            inputStream?.close()

            if (rawBitmap != null) {
                // Correct EXIF orientation so rotated camera photos infer correctly
                val correctedBitmap = correctOrientation(uri, rawBitmap)

                placeholderContainer.visibility = View.GONE
                imageView.setImageBitmap(correctedBitmap)

                // Run classification on background thread for smooth UI responsiveness
                Thread {
                    val result = classifier.classify(correctedBitmap)
                    runOnUiThread {
                        displayResult(result)
                    }
                }.start()
            }
        } catch (e: Exception) {
            Toast.makeText(this, "Failed to load image: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun correctOrientation(uri: Uri, bitmap: Bitmap): Bitmap {
        return try {
            contentResolver.openInputStream(uri)?.use { stream ->
                val exif = ExifInterface(stream)
                val orientation = exif.getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_NORMAL
                )
                val matrix = Matrix()
                when (orientation) {
                    ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
                    ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
                    ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
                    else -> return bitmap
                }
                Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            } ?: bitmap
        } catch (e: Exception) {
            bitmap
        }
    }

    private fun runOnDeviceBenchmark() {
        tvResultClass.text = "Running On-Device Benchmark..."
        tvConfidence.text = "Evaluating 10 warmup + 50 timed iterations..."
        tvAbstentionStatus.text = "In progress..."
        tvAbstentionStatus.setTextColor(getColor(R.color.brand_accent))

        Thread {
            val sampleBitmap = Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888)
            // 10 warmup passes
            for (i in 0 until 10) {
                classifier.classify(sampleBitmap)
            }

            // 50 timed runs
            val latencies = LongArray(50)
            var lastResult: ClassificationResult? = null
            for (i in 0 until 50) {
                val res = classifier.classify(sampleBitmap)
                latencies[i] = res.inferenceTimeMs
                lastResult = res
            }

            latencies.sort()
            val meanLat = latencies.average()
            val p50 = latencies[25]
            val p90 = latencies[45]
            val p95 = latencies[47]
            val minLat = latencies.first()
            val maxLat = latencies.last()

            android.util.Log.i("AgriBenchmark", "ON_DEVICE_BENCHMARK_RESULT: mean=%.2f ms, p50=%d ms, p90=%d ms, p95=%d ms, min=%d ms, max=%d ms".format(meanLat, p50, p90, p95, minLat, maxLat))

            runOnUiThread {
                tvResultClass.text = "On-Device Benchmark Complete!"
                tvConfidence.text = "Mean: %.2f ms | Median: %d ms | P90: %d ms".format(meanLat, p50, p90)
                tvLatency.text = "Min: %d ms | Max: %d ms | P95: %d ms".format(minLat, maxLat, p95)
                tvAbstentionStatus.text = "Samsung Exynos 1330"
                tvAbstentionStatus.setTextColor(getColor(R.color.brand_accent))
                lastResult?.let { displayResult(it) }
            }
        }.start()
    }

    private fun displayResult(result: ClassificationResult) {
        // Format human-friendly class label (e.g. Tomato Early Blight)
        val formattedName = result.topClassName.replace("___", " — ").replace("__", " — ").replace("_", " ")

        tvResultClass.text = formattedName
        tvConfidence.text = String.format("Calibrated Confidence: %.1f%%", result.confidence * 100)
        tvLatency.text = String.format("Inference Latency: %d ms (Total: %d ms)", result.inferenceTimeMs, result.totalTimeMs)

        if (result.isAccepted) {
            tvAbstentionStatus.text = "ACCEPTED (High Confidence)"
            tvAbstentionStatus.setTextColor(getColor(R.color.status_accepted))
        } else {
            tvAbstentionStatus.text = "ABSTAINED (Low Confidence)"
            tvAbstentionStatus.setTextColor(getColor(R.color.status_abstained))
        }

        val topKText = StringBuilder()
        for ((name, prob) in result.topKPredictions) {
            val friendly = name.replace("___", " — ").replace("__", " — ").replace("_", " ")
            topKText.append(String.format("• %s: %.1f%%\n", friendly, prob * 100))
        }
        tvTopK.text = topKText.toString().trimEnd()
    }
}

