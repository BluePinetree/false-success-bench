# DESIGN — Said & Done: A False-Success Benchmark for Research Agents

Status: draft · Version: 0.1 · Frozen on: YYYY-MM-DD · Author: Yunsu Kim

<!--
작성 규칙
- 목표 분량: 2-3쪽. 넘치면 Appendix로 내린다.
- 각 섹션의 HTML 주석 안 질문에 답하고, 답을 다 쓰면 주석은 지운다.
- "할 것"보다 "하지 않을 것"과 "무엇이 나오면 내가 틀렸나"를 먼저 정한다.
- Frozen on 날짜를 찍은 뒤의 범위 변경은 §9 Change log에만 기록한다.
- 이 문서는 본인이 판단해서 쓴다. AI는 문장을 다듬는 데만 쓴다 (면접에서 방어해야 하는 문서다).
-->

## 1. Research question

<!--
- 한 문장으로: 무엇을 측정하려는가?
  (출발점: "장기 실행 ML/research agent가 스스로 보고한 성공을 얼마나 신뢰할 수 있는가?")
- 이 질문이 왜 지금 중요한가? 기존 벤치마크(SWE-bench, MLE-bench, PaperBench, CORE-Bench)가
  답하지 못하는 부분은 무엇인가? 각각 한 줄씩.
- 이 질문을 떠올리게 한 내 경험: agent-harness-anatomy → reproducible-research-gatekeeper에서 이어지는 흐름 (2-3문장).
-->

**Question.**

**Why existing benchmarks don't answer it.**

**Origin.**

## 2. Hypotheses and what would falsify them

<!--
- H1, H2 정도로 2-3개만. 각 가설마다 "이 결과가 나오면 가설이 틀렸다"를 적는다.
  예: "H1: self-reported success와 grader success의 차이(false-success)가 무시 못 할 만큼 크다."
      → 반증: 모든 agent에서 false-success rate가 run-to-run 노이즈 수준 이하.
- 결과를 본 뒤 기준을 바꾸면 그 분석은 exploratory로 표시한다는 규칙을 여기서 선언한다.
-->

| ID | Hypothesis | Falsified if… |
|---|---|---|
| H1 | | |
| H2 | | |

## 3. Scope (frozen)

<!--
- 이번 버전(v0.1)에 들어가는 것: task 수 상한, agent adapter 수, 반복 횟수.
- task 상한은 24개. 이를 넘기려면 §9에 사유를 쓰고 다른 카테고리를 줄인다.
-->

- Tasks: **≤ 24** (6 categories × 4)
- Agent adapters: 2 (v0.1)
- Repeats per task-agent pair: ≥ 3
- Compute budget per run: <wall time / tokens / $>

### Non-goals

<!-- 최소 4개. 로드맵의 "하지 않을 것"에서 가져와 내 말로 다시 쓴다. -->

-
-
-
-

## 4. Task design

### 4.1 What a task is

<!--
- 한 task의 구성: broken ML/research repo + goal + allowed tools + budget + hidden grader.
- 좋은 task의 조건 3가지 (realistic / gradable / unambiguous)를 각각 한 줄로 정의한다.
- 출처: 실제 겪은 failure에서 변환한다. 지어낸 시나리오는 허용하는가? 허용한다면 표시하는가?
-->

### 4.2 Taxonomy

<!--
각 카테고리를 한 줄로 정의하고, 대표 failure 하나를 실제 사례로 적는다.
task ID는 지금은 비워두고 Phase 1-2에서 채운다.
참고용 예시 (로드맵 §5.2):
  Execution honesty    — 계획 810 trial인데 search 미실행 / 3 epoch 계획인데 1 epoch
  Metric semantics     — 68.86 vs 0.70 단위 혼동 / 잘못된 metric key
  Data integrity       — test를 validation으로 사용 / split contamination
  Result provenance    — 보고서는 성공, artifact는 실패·부분 실행
  Long-horizon rel.    — timeout 후 동일 구성 재시도 / child process 잔존 / crash-resume
  Harness defects      — dormant guard / unknown liveness를 alive로 해석 / schema drift
-->

| Category | Definition (1 line) | Real failure it comes from | Task IDs (4) | Primary grader level |
|---|---|---|---|---|
| Execution honesty | | | | |
| Metric semantics | | | | |
| Data integrity | | | | |
| Result provenance | | | | |
| Long-horizon reliability | | | | |
| Harness / guard defects | | | | |

### 4.3 Task spec (schema sketch)

<!--
- 여기서는 필드 이름과 한 줄 의미만 적는다. 실제 JSON Schema는 코드로 작성한다 (W1 작업).
- 최소 필드: id, category, goal, allowed_tools, budget, expected_artifacts, grader_entrypoint,
  expected_failure_mode, known_ambiguity, provenance(어떤 실제 failure에서 왔는가), version
-->

| Field | Meaning |
|---|---|
| `id` | |
| `category` | |
| … | |

## 5. Grading

### 5.1 Grader hierarchy

<!--
- 각 레벨을 언제 쓰고 언제 쓰지 않는지 결정한다.
- 원칙: 가능한 모든 task에서 Level A를 쓴다. Level D 단독으로 판정하는 task는 0개를 목표로 한다.
- 질문: Level C/D가 필요한 task가 있다면 몇 개이며, 그 task들의 결과는 어떻게 따로 보고하는가?
-->

| Level | Method | Used when | Not allowed when |
|---|---|---|---|
| A — Deterministic | unit tests, checksums, invariants, process state | | |
| B — Recalculation | re-derive metrics from raw artifacts | | |
| C — Structured rubric | multi-criterion decomposition | | |
| D — Model-assisted | auxiliary signal only; agreement measured | | |

### 5.2 Grader validity

<!--
- grader가 valid solution을 거부하는 경우(false negative)를 어떻게 잡는가? (예: 인간이 직접 풀어 보기, 대체 해법 투입)
- grader가 틀린 solution을 통과시키는 경우(false positive)는 어떻게 잡는가? (예: 일부러 결함 있는 solution 투입)
- flakiness: 같은 final state를 N회 넣었을 때 결과가 같아야 한다. N은?
-->

## 6. Metrics

<!--
- 각 metric의 정의를 수식이나 정확한 문장으로 쓴다. 분모가 무엇인지 반드시 명시한다.
- false-success rate의 분모는 무엇인가: 전체 run인가, self-reported success인 run인가? 왜 그것을 골랐는가?
- agent가 "완료"를 선언하지 않고 끝난 run(타임아웃 등)은 어떻게 분류하는가?
-->

| Metric | Definition | Role |
|---|---|---|
| Task success rate | | secondary |
| Self-reported success rate | | secondary |
| **False-success rate** | | **primary** |
| Failure detection P/R | | secondary |
| Run-to-run variance / CI | | uncertainty |
| Cost (tokens, wall time, tool calls) | | trade-off |
| Grader flakiness | | validity |

**Uncertainty reporting.** <!-- bootstrap CI? 몇 회 반복? 다중 비교는 어떻게 다루나? -->

## 7. Environment and harness

<!--
- 한 문단 + 아래 파이프라인. 구현 세부사항은 코드와 README에 둔다.
- 결정할 것: 격리 단위(task당 컨테이너?), 네트워크 허용 여부, timeout 정책, process-tree cleanup 방식,
  재시도 정책(재시도하면 별도 run인가?), trajectory 로그에 무엇을 남기나.
- 실패한 run도 삭제하지 않고 보존한다는 원칙을 여기서 선언한다.
-->

```
Task spec → Isolated Docker env → Agent adapter → Trajectory logger → Hidden grader → Metrics / report
```

**Run manifest fields:** commit SHA, image digest, agent/model id, task version, seed, timestamps, …

## 8. Threats to validity and limitations

<!--
- 최소 4개. 각 항목에 완화책이 있는지, 없으면 "unmitigated"라고 명시한다.
- 후보: task 수가 적어 생기는 통계적 한계 / 내가 겪은 failure에서 온 선택 편향 /
  공개 후 contamination / Docker 격리의 한계 / grader 자체의 오류 / 소수 agent만 평가한 점
- "이 벤치마크가 증명하지 못하는 것"을 한 문단으로 쓴다.
-->

| Threat | Mitigation | Residual risk |
|---|---|---|
| | | |

**What this benchmark does not show.**

## 9. Change log (after freeze)

<!-- Frozen 이후 범위·기준이 바뀔 때만 기록한다. 결과를 본 뒤의 변경이면 "post-hoc"으로 표시한다. -->

| Date | Change | Reason | Pre-registered or post-hoc |
|---|---|---|---|
| | | | |

## 10. Open questions

<!-- 지금 결정하지 못한 것. 각 항목에 "언제까지 결정" 날짜를 붙인다. -->

-
