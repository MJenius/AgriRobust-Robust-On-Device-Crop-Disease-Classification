package com.agrirobust.classifier

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import java.io.InputStream

class MainActivity : AppCompatActivity() {

    private lateinit var classifier: AgriClassifier
    private lateinit var imageView: ImageView
    private lateinit var tvResultClass: TextView
    private lateinit var tvConfidence: TextView
    private lateinit var tvAbstentionStatus: TextView
    private lateinit var tvLatency: TextView
    private lateinit var tvTopK: TextView

    private val selectImageLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let { handleImageUri(it) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        classifier = AgriClassifier(this)

        imageView = findViewById(R.id.imageView)
        tvResultClass = findViewById(R.id.tvResultClass)
        tvConfidence = findViewById(R.id.tvConfidence)
        tvAbstentionStatus = findViewById(R.id.tvAbstentionStatus)
        tvLatency = findViewById(R.id.tvLatency)
        tvTopK = findViewById(R.id.tvTopK)

        findViewById<Button>(R.id.btnSelectImage).setOnClickListener {
            selectImageLauncher.launch("image/*")
        }
    }

    private fun handleImageUri(uri: Uri) {
        val inputStream: InputStream? = contentResolver.openInputStream(uri)
        val bitmap = BitmapFactory.decodeStream(inputStream)
        inputStream?.close()

        if (bitmap != null) {
            imageView.setImageBitmap(bitmap)
            val result = classifier.classify(bitmap)
            displayResult(result)
        }
    }

    private fun displayResult(result: ClassificationResult) {
        tvResultClass.text = result.topClassName
        tvConfidence.text = String.format("Confidence: %.1f%%", result.confidence * 100)
        tvLatency.text = String.format("Inference: %d ms | Total: %d ms", result.inferenceTimeMs, result.totalTimeMs)

        if (result.isAccepted) {
            tvAbstentionStatus.text = "ACCEPTED (High Confidence)"
            tvAbstentionStatus.setTextColor(getColor(android.R.color.holo_green_dark))
        } else {
            tvAbstentionStatus.text = "ABSTAINED (Low Confidence / Uncertain Prediction)"
            tvAbstentionStatus.setTextColor(getColor(android.R.color.holo_red_dark))
        }

        val topKText = StringBuilder("Top Predictions:\n")
        for ((name, prob) in result.topKPredictions) {
            topKText.append(String.format("• %s: %.1f%%\n", name, prob * 100))
        }
        tvTopK.text = topKText.toString()
    }
}
