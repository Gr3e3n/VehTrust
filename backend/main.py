from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import json
import os

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

app = FastAPI(title="VehTrust AI Backend", version="1.0.0")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


class ResponsibilityPayload(BaseModel):
    driverFactor: int
    systemFactor: int
    environmentFactor: int
    conclusion: str
    reasons: list[str] = Field(default_factory=list)


class EnvironmentPayload(BaseModel):
    weather: str
    road: str
    obstacle: str
    laneMarking: str


class DecisionTracePayload(BaseModel):
    sensorInput: str
    perception: str
    planning: str
    control: str


class TelemetryPayload(BaseModel):
    tMs: int
    speedKph: float
    axMS2: float
    brake: int
    steerDeg: float


class AccidentAnalyzeRequest(BaseModel):
    eventId: str
    eventType: str
    timeMillis: int
    location: str
    summary: str
    triggerReasons: list[str] = Field(default_factory=list)
    severity: str
    autoDrivingState: str
    telemetry: list[TelemetryPayload] = Field(default_factory=list)
    responsibility: ResponsibilityPayload
    environment: EnvironmentPayload | None = None
    decisionTrace: DecisionTracePayload | None = None


def build_prompt(payload: AccidentAnalyzeRequest) -> str:
    telemetry_summary = "\n".join(
        f"t={point.tMs}ms, speed={point.speedKph:.2f}km/h, ax={point.axMS2:.2f}m/s², brake={point.brake}%, steer={point.steerDeg:.2f}°"
        for point in payload.telemetry[:20]
    )

    return f"""
请基于以下事故数据生成“结构化专家报告”JSON，字段固定为：
summary, rootCause, comprehensiveAnalysis, scenarioReconstruction, confidenceStatement, evidencePoints, suggestions, modelHint, rawText。

输出要求：
1) 只输出 JSON，不要 markdown，不要代码块。
2) 严格基于输入数据，不得虚构未提供的证据或结论。
3) summary：1 段，80~160 字，概括事件本质与主要风险。
4) rootCause：1 段，60~120 字，明确“主因 + 次因”。
5) comprehensiveAnalysis：2~3 段，分别从驾驶、系统、环境角度解释因果链。
6) scenarioReconstruction：1~2 段，按时间顺序复盘风险形成与碰撞过程。
7) confidenceStatement：1 段，说明置信度（高/中/低）及不确定性来源。
8) evidencePoints：3~6 条，简短证据点列表。
9) suggestions：3~5 条，可执行改进建议或补充取证建议。
10) rawText：3~5 段“增量叙述”，每段都必须提供前文未出现的新信息。
11) rawText 必须优先覆盖以下维度中未被前文覆盖的项：
   - 时间线细节（具体到阶段/先后关系）
   - 证据链闭环（某证据如何支持某结论）
   - 反事实或替代解释（若关键条件变化，结果可能如何变化）
   - 待补充取证点（缺失什么数据会影响结论）
12) rawText 严禁复述 summary/rootCause/comprehensiveAnalysis/scenarioReconstruction 的句子或近义改写。”。
13) 若输入证据不足，rawText 只写“信息不足点 + 建议补采数据”等内容，不要重复已有判断。

【事件】
id={payload.eventId}
type={payload.eventType}
time={payload.timeMillis}
location={payload.location}
summary={payload.summary}
triggers={'、'.join(payload.triggerReasons)}
severity={payload.severity}
autoDrivingState={payload.autoDrivingState}

【责任分析】
conclusion={payload.responsibility.conclusion}
driver={payload.responsibility.driverFactor}%
system={payload.responsibility.systemFactor}%
environment={payload.responsibility.environmentFactor}%
reasons={' | '.join(payload.responsibility.reasons)}

【环境】
{payload.environment.model_dump_json() if payload.environment else 'null'}

【决策链】
{payload.decisionTrace.model_dump_json() if payload.decisionTrace else 'null'}

【事故前关键遥测】
{telemetry_summary}
""".strip()


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL}


@app.post("/api/accident/analyze")
def analyze_accident(payload: AccidentAnalyzeRequest):
    if not OPENAI_API_KEY or client is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY 未配置")

    try:
        completion = client.chat.completions.create(
            model=MODEL,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "你是资深车载事故分析专家。请严格基于输入数据进行专业、谨慎、可解释的因果分析，不要虚构事实，输出结构化 JSON。若发现后文与前文语义重复，必须优先改写为新增证据、新增视角或新增不确定性说明。",
                },
                {
                    "role": "user",
                    "content": build_prompt(payload),
                },
            ],
        )
        content = completion.choices[0].message.content or "{}"
        data = json.loads(content)
        return {
            "success": True,
            "data": {
                "summary": data.get("summary", "未返回事故摘要"),
                "rootCause": data.get("rootCause", "未返回根因判断"),
                "comprehensiveAnalysis": data.get("comprehensiveAnalysis", ""),
                "scenarioReconstruction": data.get("scenarioReconstruction", ""),
                "confidenceStatement": data.get("confidenceStatement", ""),
                "evidencePoints": data.get("evidencePoints", []),
                "suggestions": data.get("suggestions", []),
                "modelHint": f"模型说明：{MODEL}",
                "rawText": data.get("rawText", content),
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
