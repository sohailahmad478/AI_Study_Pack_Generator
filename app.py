import os, json, time
import streamlit as st
from groq import Groq

MODEL = "openai/gpt-oss-20b"
TOKENS = {"Planning":1200,"Content Generation":1900,"Assessment":1600,"Review":1000,"Refine":2000}

st.set_page_config(page_title="AI Study Pack Generator", page_icon="📚", layout="wide")

SYSTEM = {
"Planning": """You are the Planning Agent. Return ONLY a JSON object with keys learner_profile, objectives, topic_map, difficulty_progression, weekly_plan, content_strategy. Create a concise personalized plan. Do not create questions.""",
"Content Generation": """You are the Content Generation Agent. Return ONLY JSON with keys summary, learning_objectives, key_concepts, flashcards. A concept has term, explanation, example. A flashcard has question, answer. Respect requested counts.""",
"Assessment": """You are the Assessment Agent. Return ONLY JSON with keys mcqs, short_answer_questions, long_answer_questions. MCQ fields: question, options, correct_answer, explanation. Short fields: question, answer_points. Long fields: question, answer_outline.""",
"Review": """You are the Review/Quality-Control Agent. Return ONLY JSON with keys passed, quality_score, strengths, issues. Each issue has severity, section, problem, recommended_fix. Check accuracy, alignment, difficulty, duplicates, ambiguity, missing answers, clarity and personalization. Do not rewrite the pack.""",
"Refine": """You are the Refinement Agent. Return ONLY JSON containing the final study pack with keys summary, learning_objectives, key_concepts, flashcards, mcqs, short_answer_questions, long_answer_questions, study_plan, exam_tips. Preserve good content and fix review issues. No commentary outside JSON."""
}

LIMITS = {"Planning":9000,"Content Generation":9000,"Assessment":11000,"Review":13000,"Refine":15000}

def key():
    try:
        if st.secrets.get("GROQ_API_KEY"): return str(st.secrets["GROQ_API_KEY"])
    except Exception: pass
    return os.getenv("GROQ_API_KEY", "")

@st.cache_resource
def get_client():
    k=key()
    if not k: raise RuntimeError("GROQ_API_KEY is missing. Add it in Streamlit Cloud → App Settings → Secrets.")
    return Groq(api_key=k)

def compact(x,n):
    s=json.dumps(x,ensure_ascii=False,separators=(",",":"))
    return s if len(s)<=n else s[:n]+"[TRUNCATED]"

def parse_json(s):
    s=(s or "").strip()
    if s.startswith("```"):
        a=s.splitlines()[1:]
        if a and a[-1].strip()=="```": a=a[:-1]
        s="\n".join(a).strip()
    try: v=json.loads(s)
    except json.JSONDecodeError:
        a,b=s.find("{"),s.rfind("}")
        if a<0 or b<=a: raise ValueError("The AI returned invalid JSON.")
        v=json.loads(s[a:b+1])
    if not isinstance(v,dict): raise ValueError("The AI returned invalid JSON.")
    return v

def call(stage,prompt):
    c=get_client(); last=None
    for attempt in range(3):
        try:
            r=c.chat.completions.create(model=MODEL,messages=[{"role":"system","content":SYSTEM[stage]},{"role":"user","content":prompt[:LIMITS[stage]]}],temperature=0.2,max_tokens=TOKENS[stage],response_format={"type":"json_object"})
            return parse_json(r.choices[0].message.content)
        except Exception as e:
            last=e; text=str(e).lower()
            if "413" in text or "tokens per minute" in text: raise RuntimeError(f"{stage} exceeded Groq's token limit. Reduce pack size and try again.") from e
            if attempt<2: time.sleep(1.5*(attempt+1))
    raise RuntimeError(f"{stage} failed after 3 attempts: {last}")

def run_workflow(topic,level,weeks,goal,language,counts,notify):
    ctx={"input":{"topic":topic,"level":level,"weeks":weeks,"goal":goal,"language":language,"counts":counts},"outputs":{},"status":{},"errors":[]}
    def stage(name,prompt):
        ctx["status"][name]="running"; notify(name,"running")
        try:
            ctx["outputs"][name]=call(name,prompt); ctx["status"][name]="completed"; notify(name,"completed")
        except Exception as e:
            ctx["status"][name]="failed"; ctx["errors"].append({"stage":name,"error":str(e)}); notify(name,"failed"); raise
    stage("Planning",f"Learner: {compact(ctx['input'],3500)}\nCreate a {weeks}-week plan for the topic and goal.")
    stage("Content Generation",f"Learner: {compact(ctx['input'],3000)}\nPlan: {compact(ctx['outputs']['Planning'],5000)}\nCounts: {json.dumps(counts)}\nGenerate concise content in {language}.")
    stage("Assessment",f"Learner: {compact(ctx['input'],2500)}\nObjectives: {compact(ctx['outputs']['Planning'].get('objectives',[]),2500)}\nConcepts: {compact(ctx['outputs']['Content Generation'].get('key_concepts',[]),6500)}\nCounts: {json.dumps(counts)}")
    stage("Review",f"Learner: {compact(ctx['input'],2000)}\nContent: {compact(ctx['outputs']['Content Generation'],5500)}\nAssessment: {compact(ctx['outputs']['Assessment'],5500)}\nReview concisely.")
    draft={"summary":ctx["outputs"]["Content Generation"].get("summary",""),"learning_objectives":ctx["outputs"]["Content Generation"].get("learning_objectives",[]),"key_concepts":ctx["outputs"]["Content Generation"].get("key_concepts",[]),"flashcards":ctx["outputs"]["Content Generation"].get("flashcards",[]),"mcqs":ctx["outputs"]["Assessment"].get("mcqs",[]),"short_answer_questions":ctx["outputs"]["Assessment"].get("short_answer_questions",[]),"long_answer_questions":ctx["outputs"]["Assessment"].get("long_answer_questions",[])}
    stage("Refine",f"Learner: {compact(ctx['input'],2200)}\nDraft: {compact(draft,9000)}\nReview: {compact(ctx['outputs']['Review'],3500)}\nWeekly plan: {compact(ctx['outputs']['Planning'].get('weekly_plan',[]),2200)}\nReturn the final pack in {language}.")
    return ctx

def md(pack,topic):
    L=[f"# AI Study Pack — {topic}","","## Summary",pack.get("summary",""),"","## Learning Objectives"]
    L += [f"- {x}" for x in pack.get("learning_objectives",[])] + ["","## Key Concepts"]
    for x in pack.get("key_concepts",[]): L += [f"### {x.get('term','')}",x.get("explanation",""),f"**Example:** {x.get('example','')}",""]
    L += ["## Flashcards"]
    for i,x in enumerate(pack.get("flashcards",[]),1): L += [f"**{i}. Q:** {x.get('question','')}",f"**A:** {x.get('answer','')}",""]
    L += ["## MCQs"]
    for i,x in enumerate(pack.get("mcqs",[]),1): L += [f"**{i}. {x.get('question','')}**"]+[f"- {o}" for o in x.get("options",[])]+[f"**Correct:** {x.get('correct_answer','')}",f"**Explanation:** {x.get('explanation','')}",""]
    L += ["## Short-Answer Questions"]
    for i,x in enumerate(pack.get("short_answer_questions",[]),1): L += [f"**{i}. {x.get('question','')}**"]+[f"- {p}" for p in x.get("answer_points",[])]+[""]
    L += ["## Long-Answer Questions"]
    for i,x in enumerate(pack.get("long_answer_questions",[]),1): L += [f"**{i}. {x.get('question','')}**"]+[f"- {p}" for p in x.get("answer_outline",[])]+[""]
    L += ["## Study Plan"]
    for x in pack.get("study_plan",[]): L += [f"### Week {x.get('week',x.get('day',''))}: {x.get('focus','')}"]+[f"- {t}" for t in x.get("tasks",[])]+[""]
    L += ["## Exam Tips"]+[f"- {x}" for x in pack.get("exam_tips",[])]
    return "\n".join(L)

st.title("📚 AI Study Pack Generator")
st.caption("Planning → Content Generation → Assessment → Review → Refine")
with st.sidebar:
    st.header("🎯 Personalization")
    topic=st.text_input("Topic",placeholder="e.g. Python OOP")
    level=st.selectbox("Student level",["Beginner","School","High School","College / University","Professional"])
    weeks=st.number_input("Study duration (weeks)",1,12,4)
    goal=st.text_area("Learning goal",placeholder="Example: Prepare for my university exam and understand the topic deeply.")
    language=st.selectbox("Output language",["English","Urdu","Roman Urdu","Arabic","Spanish","French"])
    st.subheader("Study pack size")
    counts={"concepts":st.slider("Key concepts",3,10,5),"flashcards":st.slider("Flashcards",5,20,10),"mcqs":st.slider("MCQs",5,15,8),"short_questions":st.slider("Short-answer questions",2,8,4),"long_questions":st.slider("Long-answer questions",1,5,2)}
    st.info(f"Open-weight model: {MODEL}\n\nNo file upload required.")
if st.button("🚀 Generate Study Pack",type="primary",use_container_width=True):
    if not topic.strip() or not goal.strip(): st.warning("Enter both a topic and learning goal."); st.stop()
    try:
        bar=st.progress(0); box=st.empty(); order=["Planning","Content Generation","Assessment","Review","Refine"]
        def notify(name,state):
            i=order.index(name)
            if state=="running": box.info(f"🔄 Stage {i+1}/5: {name}"); bar.progress(i/5)
            elif state=="completed": box.success(f"✅ Stage {i+1}/5 completed: {name}"); bar.progress((i+1)/5)
            else: box.error(f"❌ Stage failed: {name}")
        st.session_state.ctx=run_workflow(topic.strip(),level,int(weeks),goal.strip(),language,counts,notify)
        st.success("🎉 Study pack generated successfully.")
    except Exception as e: st.error(f"❌ Workflow failed: {e}")
if "ctx" in st.session_state:
    ctx=st.session_state.ctx; pack=ctx["outputs"].get("Refine",{})
    if pack:
        st.header("🎉 Final Personalized Study Pack")
        a,b,c,d=st.columns(4); a.metric("Concepts",len(pack.get("key_concepts",[]))); b.metric("Flashcards",len(pack.get("flashcards",[]))); c.metric("MCQs",len(pack.get("mcqs",[]))); d.metric("Weeks",len(pack.get("study_plan",[])))
        tabs=st.tabs(["Summary","Concepts","Flashcards","MCQs","Questions","Study Plan","Exam Tips","AI Review","Workflow"])
        with tabs[0]:
            st.markdown(pack.get("summary","")); st.subheader("Learning Objectives"); [st.markdown(f"- {x}") for x in pack.get("learning_objectives",[])]
        with tabs[1]:
            for x in pack.get("key_concepts",[]):
                with st.expander(x.get("term","Concept")): st.write(x.get("explanation","")); x.get("example") and st.info(f"Example: {x['example']}")
        with tabs[2]:
            for i,x in enumerate(pack.get("flashcards",[]),1):
                with st.expander(f"Card {i}: {x.get('question','')}"): st.write(x.get("answer",""))
        with tabs[3]:
            for i,x in enumerate(pack.get("mcqs",[]),1):
                st.markdown(f"**{i}. {x.get('question','')}**"); opts=x.get("options",[])
                if opts:
                    ans=st.radio("Choose an answer:",opts,key=f"q{i}")
                    if st.button(f"Check {i}",key=f"c{i}"): st.success("Correct! 🎉") if ans==x.get("correct_answer","") else st.error(f"Correct answer: {x.get('correct_answer','')}")
        with tabs[4]:
            st.subheader("Short-Answer Questions")
            for i,x in enumerate(pack.get("short_answer_questions",[]),1):
                st.markdown(f"**{i}. {x.get('question','')}**");
                with st.expander("Answer points"): [st.markdown(f"- {p}") for p in x.get("answer_points",[])]
            st.subheader("Long-Answer Questions")
            for i,x in enumerate(pack.get("long_answer_questions",[]),1):
                st.markdown(f"**{i}. {x.get('question','')}**");
                with st.expander("Answer outline"): [st.markdown(f"- {p}") for p in x.get("answer_outline",[])]
        with tabs[5]:
            for x in pack.get("study_plan",[]): st.markdown(f"### Week {x.get('week',x.get('day',''))}: {x.get('focus','')}"); [st.checkbox(t,key=f"{x.get('week',x.get('day',''))}-{t}") for t in x.get("tasks",[])]
        with tabs[6]: [st.markdown(f"- {x}") for x in pack.get("exam_tips",[])]
        with tabs[7]:
            r=ctx["outputs"].get("Review",{}); st.metric("AI Quality Score",r.get("quality_score","N/A")); st.success("Review passed.") if r.get("passed") else st.warning("Review identified improvements.")
            [st.markdown(f"✅ {x}") for x in r.get("strengths",[])]; [st.warning(f"**{x.get('severity','').upper()} — {x.get('section','')}**\n\n{x.get('problem','')}\n\nFix: {x.get('recommended_fix','')}") for x in r.get("issues",[])]
        with tabs[8]:
            for s in ["Planning","Content Generation","Assessment","Review","Refine"]:
                v=ctx["status"].get(s,"unknown"); st.success(f"✅ {s}: completed") if v=="completed" else st.error(f"❌ {s}: failed") if v=="failed" else st.info(f"ℹ️ {s}: {v}")
            st.write("Context is passed between stages. Refine receives only a compact draft and review feedback, preventing oversized Groq requests.")
        st.subheader("⬇️ Download")
        x,y=st.columns(2); x.download_button("Download Markdown",md(pack,ctx["input"]["topic"]),"ai_study_pack.md","text/markdown",use_container_width=True); y.download_button("Download JSON",json.dumps(pack,ensure_ascii=False,indent=2),"ai_study_pack.json","application/json",use_container_width=True)
