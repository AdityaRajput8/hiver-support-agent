import streamlit as st
import yaml
from src.pipeline import SupportPipeline

st.set_page_config(page_title="Uber Support AI Triage Agent", layout="wide")

@st.cache_resource
def load_pipeline():
    return SupportPipeline()

pipeline = load_pipeline()

st.title("🚗 @Uber_Support AI Triage & Resolution Agent")
st.markdown(
    "Auditable, grounded support pipeline with deterministic escalation policies and safety gates."
)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Incoming Customer Tweet")
    sample_options = [
        "Select an example...",
        "Why was I charged $35 cancellation fee when the driver canceled the trip himself??",
        "The driver locked the doors and refused to let me exit the car. I had to threaten calling the police.",
        "Left my red wallet and office keys in the back seat of a white Camry 20 mins ago.",
        "My app keeps crashing whenever I try to log in on my new phone.",
        "How do I update my business account invoice email address?"
    ]
    selected_sample = st.selectbox("Preset Scenarios", sample_options)
    
    default_text = "" if selected_sample == "Select an example..." else selected_sample
    user_input = st.text_area("Customer Tweet Input:", value=default_text, height=120)
    submit_btn = st.button("Run Triage Pipeline", type="primary")

if submit_btn and user_input.strip():
    with st.spinner("Processing through Intent -> Retrieval -> Drafting -> Critic -> Policy..."):
        res = pipeline.process(user_input.strip())

    with col2:
        st.subheader("Triage Audit Summary")
        decision = res["decision"]
        
        if decision == "AUTO_HANDLE":
            st.success(f"### DECISION: {decision}")
        else:
            st.error(f"### DECISION: {decision}")
            
        st.write(f"**Escalation Reason Code:** `{res['escalation_reason']}`")
        st.write(f"**Policy Detail:** {res['escalation_detail']}")

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Intent", res["intent"])
    c2.metric("Intent Confidence", f"{res['intent_confidence'] * 100:.1f}%")
    c3.metric("Faithfulness Score", f"{res['groundedness_score'] * 100:.1f}%")

    st.markdown("---")
    tab1, tab2, tab3 = st.columns([1.2, 1, 1])

    with tab1:
        st.subheader("Proposed Tweet Reply")
        st.info(res["draft_reply"])
        st.caption(f"Character Count: {len(res['draft_reply'])} / 280")

    with tab2:
        st.subheader("Top Historical Precedents")
        for i, item in enumerate(res["retrieved_precedents"]):
            st.write(f"**Precedent {i+1} (Cosine Sim: {item['score']}):**")
            st.caption(item["reply"])

    with tab3:
        st.subheader("Internal Verification")
        st.write(f"**Intent Rationale:** {res['intent_reasoning']}")
        st.write(f"**Critic Notes:** {res['critic_reason']}")