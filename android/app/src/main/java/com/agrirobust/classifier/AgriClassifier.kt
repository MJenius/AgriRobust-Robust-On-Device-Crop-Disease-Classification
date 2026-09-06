package com.agrirobust.classifier

import android.content.Context
import android.graphics.Bitmap
import org.json.JSONObject
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import kotlin.math.exp

/**
 * Result data class for AgriRobust plant disease classification.
 */
data class ClassificationResult(
    val topClassName: String,
    val topClassIndex: Int,
    val confidence: Float,
    val isAccepted: Boolean,
    val rawLogit: Float,
    val topKPredictions: List<Pair<String, Float>>,
    val inferenceTimeMs: Long,
    val totalTimeMs: Long
)

/**
 * AgriRobust mobile on-device classification engine.
 *
 * Implements:
 * 1. MobileNetV3-Small on-device forward pass.
 * 2. Deterministic ImageNet RGB normalization.
 * 3. Frozen Phase 6 temperature scaling (T = 0.5406).
 * 4. Frozen Phase 6 selective abstention threshold (tau = 0.8143).
 */
class AgriClassifier(private val context: Context) {

    private var module: Module? = null
    private val classLabels = mutableListOf<String>()
    
    // Strict frozen parameters from Phase 6
    val calibrationTemperature: Float = 0.5406f
    val abstentionThreshold: Float = 0.8143f

    init {
        loadModelAndLabels()
    }

    private fun assetFilePath(assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }
        context.assets.open(assetName).use { inputStream ->
            FileOutputStream(file).use { outputStream ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
                outputStream.flush()
            }
        }
        return file.absolutePath
    }

    private fun loadModelAndLabels() {
        try {
            // Load labels.json
            val labelsJsonStr = context.assets.open("labels.json").bufferedReader().use { it.readText() }
            val jsonObject = JSONObject(labelsJsonStr)
            val classesArray = jsonObject.getJSONArray("classes")
            classLabels.clear()
            for (i in 0 until classesArray.length()) {
                classLabels.add(classesArray.getString(i))
            }

            // Load PyTorch Mobile model artifact
            // Tries dynamic-int8 artifact first; falls back to fp32 if unavailable
            val modelPath = try {
                assetFilePath("model_int8_dynamic.pt")
            } catch (e: IOException) {
                assetFilePath("model_fp32.pt")
            }

            module = LiteModuleLoader.load(modelPath)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    /**
     * Run deterministic inference on input bitmap.
     */
    fun classify(bitmap: Bitmap): ClassificationResult {
        val t0 = System.currentTimeMillis()

        // 1. Resize and Center-Crop to 224x224
        val resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)

        // 2. Preprocess: CHW RGB float tensor normalized with ImageNet mean/std
        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
            resized,
            floatArrayOf(0.485f, 0.456f, 0.406f),
            floatArrayOf(0.229f, 0.224f, 0.225f)
        )

        // 3. Model Inference
        val tInfStart = System.currentTimeMillis()
        val outputTensor = module!!.forward(IValue.from(inputTensor)).toTensor()
        val tInfEnd = System.currentTimeMillis()
        val logits = outputTensor.dataAsFloatArray

        // 4. Apply Frozen Temperature Scaling: z_scaled = z / 0.5406
        val scaledLogits = FloatArray(logits.size) { i -> logits[i] / calibrationTemperature }

        // 5. Stable Softmax
        var maxLogit = Float.NEGATIVE_INFINITY
        for (v in scaledLogits) {
            if (v > maxLogit) maxLogit = v
        }

        var sumExp = 0.0
        val expValues = DoubleArray(scaledLogits.size)
        for (i in scaledLogits.indices) {
            expValues[i] = exp((scaledLogits[i] - maxLogit).toDouble())
            sumExp += expValues[i]
        }

        val probabilities = FloatArray(scaledLogits.size) { i -> (expValues[i] / sumExp).toFloat() }

        // 6. Find Top-1 & Top-K predictions
        var topIndex = 0
        var topProb = probabilities[0]
        for (i in 1 until probabilities.size) {
            if (probabilities[i] > topProb) {
                topProb = probabilities[i]
                topIndex = i
            }
        }

        // Top-K ranking
        val sortedIndices = probabilities.indices.sortedByDescending { probabilities[it] }
        val topK = sortedIndices.take(5).map { idx ->
            val name = if (idx < classLabels.size) classLabels[idx] else "Class_$idx"
            Pair(name, probabilities[idx])
        }

        // 7. Selective Abstention Rule
        val isAccepted = topProb >= abstentionThreshold
        val tFinal = System.currentTimeMillis()

        val topClassName = if (topIndex < classLabels.size) classLabels[topIndex] else "Class_$topIndex"

        return ClassificationResult(
            topClassName = topClassName,
            topClassIndex = topIndex,
            confidence = topProb,
            isAccepted = isAccepted,
            rawLogit = logits[topIndex],
            topKPredictions = topK,
            inferenceTimeMs = tInfEnd - tInfStart,
            totalTimeMs = tFinal - t0
        )
    }
}
