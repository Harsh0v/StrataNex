
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from pathlib import Path

st.set_page_config(page_title="StrataX", page_icon="⛰️", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.block-container{max-width:760px;padding:0.8rem 0.7rem 5rem}
h1{font-size:1.65rem!important;margin:.2rem 0}
h2{font-size:1.25rem!important}
[data-testid="stMetric"]{border:1px solid rgba(128,128,128,.22);padding:10px;border-radius:16px}
[data-testid="stMetricValue"]{font-size:1.25rem}
.stButton>button{width:100%;min-height:48px;border-radius:14px;font-weight:650}
div[data-baseweb="select"]{min-height:48px}
.card{border:1px solid rgba(128,128,128,.22);border-radius:18px;padding:14px;margin:8px 0}
.badge{display:inline-block;padding:5px 10px;border-radius:20px;border:1px solid rgba(128,128,128,.3);font-size:.8rem}
.small{opacity:.72;font-size:.84rem}
@media(max-width:600px){
 .block-container{padding:.55rem .55rem 5rem}
 [data-testid="column"]{min-width:0!important}
}
</style>
""", unsafe_allow_html=True)

DATA=Path(__file__).parent/"pilot_segments.csv"

@st.cache_data
def load(): return pd.read_csv(DATA)

def risk_engine(x,rain,soil,tilt):
    d=x.copy()
    d["risk"]=(.30*np.clip(rain/200,0,1)+.25*np.clip(d.slope_deg/60,0,1)+
               .20*np.clip(d.historical_density/10,0,1)+.15*np.clip(soil/100,0,1)+
               .10*np.clip(tilt/10,0,1)).clip(0,1)
    d["class"]=pd.cut(d.risk,[-.01,.30,.50,.70,1.01],labels=["Low","Medium","High","Critical"])
    d["confidence"]=(.62+.30*np.abs(d.risk-.5)*2).clip(.62,.92)
    return d

if "logged" not in st.session_state: st.session_state.logged=False
if "role" not in st.session_state: st.session_state.role="Citizen"
if "page" not in st.session_state: st.session_state.page="Home"

# Login screen
if not st.session_state.logged:
    st.title("⛰️ STRATAX")
    st.subheader("Landslide Safety • NER")
    st.markdown('<div class="card"><b>Predict • Explain • Visualise • Act</b><br><span class="small">Mobile decision-support prototype for landslide risk monitoring.</span></div>',unsafe_allow_html=True)
    role=st.selectbox("Continue as",["Citizen","Field Officer","Administrator"])
    name=st.text_input("Name",placeholder="Enter your name")
    if st.button("Continue to StrataX"):
        st.session_state.logged=True
        st.session_state.role=role
        st.session_state.user=name.strip() or "Demo User"
        st.rerun()
    st.caption("Academic prototype • Not an official emergency-warning service")
    st.stop()

# Prototype inputs
rain=95; soil=67; tilt=3.2
df=risk_engine(load(),rain,soil,tilt)
danger=df[df["class"].astype(str).isin(["High","Critical"])]
avg=float(df.risk.mean())

st.title("⛰️ STRATAX")
st.caption(f"{st.session_state.role} • {st.session_state.get('user','Demo User')}")

# Mobile navigation
nav=st.radio("Menu",["Home","Live Map","Prediction","SafeRoute","Emergency","Profile"],horizontal=True,label_visibility="collapsed")
st.session_state.page=nav

if nav=="Home":
    st.subheader("Good to see you")
    c1,c2=st.columns(2)
    c1.metric("Corridor Risk",f"{avg:.0%}")
    c2.metric("System","ALERT" if len(danger) else "NORMAL")
    if len(danger):
        st.warning(f"⚠️ {len(danger)} monitored segment(s) currently show High/Critical prototype risk.")
    else: st.success("✓ No High/Critical prototype risk.")
    st.markdown('<div class="card"><b>Quick Safety</b><br>Check the live risk map before travelling through monitored hill corridors. Follow official GSI, IMD and local authority instructions.</div>',unsafe_allow_html=True)
    st.subheader("Current conditions")
    a,b,c=st.columns(3)
    a.metric("Rain",f"{rain} mm")
    b.metric("Soil",f"{soil}%")
    c.metric("Tilt",f"{tilt:.1f}")

elif nav=="Live Map":
    st.subheader("Live Risk Map")
    m=df.copy()
    colors={"Low":[22,163,74,190],"Medium":[234,179,8,200],"High":[249,115,22,220],"Critical":[220,38,38,235]}
    m["color"]=m["class"].astype(str).map(colors)
    st.pydeck_chart(pdk.Deck(
      layers=[pdk.Layer("ScatterplotLayer",m,get_position="[lon,lat]",get_fill_color="color",get_radius=350,pickable=True)],
      initial_view_state=pdk.ViewState(latitude=m.lat.mean(),longitude=m.lon.mean(),zoom=6.7),
      tooltip={"text":"{segment}\n{class} • {risk}"}))
    seg=st.selectbox("Corridor details",df.segment)
    r=df[df.segment==seg].iloc[0]
    st.markdown(f'<div class="card"><b>{seg}</b><br>Risk: <b>{r["class"]}</b> • Probability {r.risk:.0%}<br>Confidence {r.confidence:.0%}</div>',unsafe_allow_html=True)

elif nav=="Prediction":
    st.subheader("AI Risk Prediction")
    seg=st.selectbox("Select monitored slope",df.segment,key="pred")
    r=df[df.segment==seg].iloc[0]
    st.metric("Risk Level",str(r["class"]))
    st.progress(float(r.risk))
    st.write(f"Probability **{r.risk:.1%}** · Confidence **{r.confidence:.1%}**")
    drivers=pd.DataFrame({"Factor":["Rainfall","Slope","Landslide history","Soil moisture","Tilt"],
       "Impact":[.30*min(rain/200,1),.25*min(r.slope_deg/60,1),.20*min(r.historical_density/10,1),.15*min(soil/100,1),.10*min(tilt/10,1)]})
    st.subheader("Why this prediction?")
    st.bar_chart(drivers.set_index("Factor"))
    st.caption("Transparent prototype scoring. Final pilot should replace this with validated RF/XGBoost.")

elif nav=="SafeRoute":
    st.subheader("🛣 SafeRoute")
    routes=pd.DataFrame({"Route":["Direct Route","Alternate Route","Emergency Diversion"],
                         "ETA":[52,66,78],
                         "Risk":[min(.98,avg*1.20),max(.04,avg*.70),max(.03,avg*.55)]})
    routes["Status"]=routes.Risk.apply(lambda x:"AVOID" if x>=.70 else ("CAUTION" if x>=.45 else "PREFERRED"))
    best=routes.sort_values(["Risk","ETA"]).iloc[0]
    st.success(f"✓ Recommended\n\n**{best.Route}** • {best.ETA} min • Risk {best.Risk:.0%}")
    for _,r in routes.iterrows():
        st.markdown(f'<div class="card"><b>{r.Route}</b> <span class="badge">{r.Status}</span><br>ETA {r.ETA} min • Risk {r.Risk:.0%}</div>',unsafe_allow_html=True)

elif nav=="Emergency":
    st.subheader("🔔 Emergency & Alerts")
    if danger.empty: st.success("No High/Critical alerts.")
    else:
        for _,r in danger.sort_values("risk",ascending=False).iterrows():
            st.error(f"⚠️ {r.segment}\n\n{r['class']} risk • {r.risk:.0%}")
    st.subheader("Safety message")
    lang=st.selectbox("Language",["English","Hindi"])
    if lang=="Hindi":
        st.warning("सावधान: पहाड़ी मार्ग पर भूस्खलन का जोखिम बढ़ा है। सुरक्षित मार्ग चुनें और आधिकारिक प्रशासनिक सलाह का पालन करें।")
    else:
        st.warning("Caution: landslide risk is elevated on monitored hill sections. Use a safer route and follow official authority guidance.")
    if st.button("📞 Emergency Call — 112"):
        st.info("On a deployed mobile app, this control should open the phone dialler for India's emergency number 112.")
    st.caption("Prototype alerts do not replace official emergency warnings.")

elif nav=="Profile":
    st.subheader("Profile / Admin")
    st.write(f"**User:** {st.session_state.get('user')}")
    st.write(f"**Role:** {st.session_state.role}")
    with st.expander("Data & system status"):
        st.write("GSI/NLFC: planned authoritative landslide source")
        st.write("IMD: planned rainfall/forecast source")
        st.write("NRSC/ISRO: planned historical/geospatial source")
        st.write("IoT: phased priority-slope deployment")
    if st.session_state.role in ["Field Officer","Administrator"]:
        st.subheader("Field verification")
        event=st.selectbox("Observed condition",["No event","Minor debris","Slope movement","Landslide"])
        note=st.text_area("Field note")
        if st.button("Submit verification"):
            st.success("Demo verification captured for threshold/model feedback.")
    if st.button("Sign out"):
        st.session_state.logged=False
        st.rerun()

st.divider()
st.caption("STRATAX • Predict → Explain → Visualise → Act → Verify → Improve")
