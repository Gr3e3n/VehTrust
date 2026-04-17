package com.example.vehtrust.trace

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL

object OpenAiAnalysisApi {

    private const val BASE_URL = "http://10.0.2.2:8080"
    private const val ENDPOINT = "/api/accident/analyze"
    private const val TIMEOUT_MS = 30_000

    suspend fun analyzeAccident(bundle: AccidentDetailBundle): AiAccidentAnalysisResult =
        withContext(Dispatchers.IO) {
            try {
                val response = post(bundle.toAiRequestJson())
                parseResponse(response)
            } catch (e: Exception) {
                fallbackAiAnalysis(bundle, remoteError = simplifyError(e.message))
            }
        }

    private fun post(jsonBody: String): String {
        val connection = (URL("$BASE_URL$ENDPOINT").openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = TIMEOUT_MS
            readTimeout = TIMEOUT_MS
            doOutput = true
            setRequestProperty("Content-Type", "application/json; charset=utf-8")
            setRequestProperty("Accept", "application/json")
        }
        OutputStreamWriter(connection.outputStream, Charsets.UTF_8).use {
            it.write(jsonBody)
            it.flush()
        }
        val stream = if (connection.responseCode in 200..299) {
            connection.inputStream
        } else {
            connection.errorStream
        }
        val raw = stream.bufferedReader(Charsets.UTF_8).use { it.readText() }
        if (connection.responseCode !in 200..299) {
            throw IllegalStateException("HTTP ${connection.responseCode}: $raw")
        }
        return raw
    }

    private fun parseResponse(raw: String): AiAccidentAnalysisResult {
        val obj = JSONObject(raw)
        if (!obj.optBoolean("success", true)) {
            throw IllegalStateException(obj.optString("message", "AI 分析失败"))
        }

        val data = obj.optJSONObject("data") ?: obj
        return AiAccidentAnalysisResult(
            summary = data.readText("summary", "未返回事故摘要"),
            rootCause = data.readText("rootCause", "未返回根因判断"),
            comprehensiveAnalysis = data.readText("comprehensiveAnalysis"),
            scenarioReconstruction = data.readText("scenarioReconstruction"),
            confidenceStatement = data.readText("confidenceStatement"),
            evidencePoints = data.optJSONArray("evidencePoints").toStringList(),
            suggestions = data.optJSONArray("suggestions").toStringList(),
            modelHint = data.readText("modelHint", "OpenAI 分析").sanitizeModelHint(),
            rawText = data.readText("rawText"),
            remoteError = null,
        )
    }

    private fun JSONArray?.toStringList(): List<String> {
        if (this == null) return emptyList()
        return buildList {
            for (index in 0 until length()) {
                add(optString(index))
            }
        }.filter { it.isNotBlank() }
    }

    private fun JSONObject.readText(key: String, fallback: String = ""): String {
        val value = opt(key)
        val normalized = when (value) {
            null, JSONObject.NULL -> ""
            is JSONArray -> value.toStringList().joinToString("\n\n")
            is JSONObject -> value.toReadableParagraphs()
            is String -> value.normalizeStructuredText()
            else -> value.toString()
        }
        return normalized.ifBlank { fallback }
    }

    private fun JSONObject.toReadableParagraphs(): String {
        val ordered = listOf(
            "driving" to "驾驶方面",
            "system" to "系统方面",
            "environment" to "环境方面",
        )

        val knownParts = ordered.mapNotNull { (key, label) ->
            optString(key).trim().takeIf { it.isNotBlank() }?.let { "$label：$it" }
        }
        if (knownParts.isNotEmpty()) return knownParts.joinToString("\n\n")

        val fallbackParts = mutableListOf<String>()
        val keys = keys()
        while (keys.hasNext()) {
            val key = keys.next()
            val value = optString(key).trim()
            if (value.isNotBlank()) fallbackParts += "$key：$value"
        }
        return fallbackParts.joinToString("\n\n")
    }

    private fun String.normalizeStructuredText(): String {
        val raw = trim()
        if (!(raw.startsWith("{") && raw.endsWith("}"))) return raw

        val parsed = runCatching { JSONObject(raw) }
            .recoverCatching { JSONObject(raw.replace('\'', '"')) }
            .getOrNull()

        return parsed?.toReadableParagraphs() ?: raw
    }

    private fun String.sanitizeModelHint(): String {
        val cleaned = this
            .replace(Regex("^\\s*Backend\\s*\\+\\s*", RegexOption.IGNORE_CASE), "")
            .replace(Regex("^\\s*Backend\\s*", RegexOption.IGNORE_CASE), "")
            .trim()
            .ifBlank { "OpenAI 分析" }

        return if (Regex("^模型说明[:：]", RegexOption.IGNORE_CASE).containsMatchIn(cleaned)) {
            cleaned
        } else {
            "模型说明：$cleaned"
        }
    }

    private fun simplifyError(raw: String?): String {
        if (raw.isNullOrBlank()) return "后端服务不可用"
        return when {
            raw.contains("Connection refused", ignoreCase = true) -> "后端未启动，请先运行 8080 端口服务"
            raw.contains("failed to connect", ignoreCase = true) -> "无法连接后端，请确认电脑与模拟器网络互通"
            raw.contains("404") -> "后端接口不存在，请检查 /api/accident/analyze"
            raw.contains("504") || raw.contains("timed out", ignoreCase = true) -> "请求 OpenAI 超时，请稍后重试或配置可用中转网关"
            raw.contains("502") -> "后端到 OpenAI 的链路异常，请检查网络/代理或中转配置"
            raw.contains("500") -> "后端 AI 服务内部错误，请检查服务日志"
            else -> raw.take(200)
        }
    }
}
