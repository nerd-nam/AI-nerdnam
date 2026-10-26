"""로컬 real 모드 /restructure 확인: SPEC fixtures 프로젝트 102로 initial → split → episode를 차례로 호출한다.
사용: AI_URL=http://localhost:8000 python evals/smoke_restructure.py ../SPEC-nerdnam/fixtures/projects.json"""
import json, os, sys, time, urllib.error, urllib.request

AI = os.getenv("AI_URL", "http://localhost:8000")
P = next(p for p in json.load(open(sys.argv[1]))["projects"] if p["project_id"] == 102)
RAW = P["raw_text"]


def post(body):
    t = time.time()
    req = urllib.request.Request(AI + "/restructure", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=300))
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read().decode()[:300]); return None
    print(f"--- {body['mode']}  {time.time() - t:.1f}s  model={r['model']} prompt={r['prompt_version']}  사례 {len(r['episodes'])}개")
    for i, e in enumerate(r["episodes"], 1):
        print(f"[{i}] {e['title']} | 역할: {e['role']}")
        print("    문제:", e["problem"]); print("    해결:", e["solution"]); print("    결과:", e["result"])
        print("    기술:", e["skills"], "| 수치:", [(m["name"], m["before"], m["after"], m["source"]) for m in e["metrics"]])
        print("    spans:", e["source_spans"], [RAW[a:b][:25] + "…" for a, b in e["source_spans"]])
        for c in e["clarifications"]:
            print(f"    ? {c['key']} [{c['field']}] {c['question']}  ({c['example']})")
    return r


r = post({"mode": "initial", "title": P["title"], "source_text": RAW, "skills": P["skills"]})
if r:
    ep = r["episodes"][0]
    post({"mode": "split", "title": P["title"], "source_text": RAW, "skills": P["skills"],
          "current_episodes": r["episodes"], "feedback": "두 사례를 하나로 합쳐 주세요"})
    post({"mode": "episode", "title": P["title"], "source_text": RAW, "skills": P["skills"], "target_episode": ep,
          "answer": {"question": "피크 처리량은 어땠나요?", "answer": P["answers"][0]["answer"]}})
