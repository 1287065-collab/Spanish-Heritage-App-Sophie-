import io, math, time
from datetime import date
import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

st.set_page_config(page_title="Hispanic Heritage Historical Data Explorer", layout="wide")
st.title("Hispanic Heritage Historical Data Explorer")
st.caption("Hispanic Heritage Data Lab · World Bank historical indicators + editable modeling")

COUNTRIES = {"Mexico":"MX","Brazil (Latin American comparison)":"BR","Argentina":"AR","Colombia":"CO","Chile":"CL","Peru":"PE","Venezuela":"VE","Ecuador":"EC","Uruguay":"UY"}
INDICATORS = {
"Life expectancy at birth":"SP.DYN.LE00.IN", "Unemployment, national estimate (% of labor force)":"SL.UEM.TOTL.NE.ZS",
"GDP per capita (current US$)":"NY.GDP.PCAP.CD", "Crude birth rate (per 1,000 people)":"SP.DYN.CBRT.IN",
"Net migration (people)":"SM.POP.NETM", "Intentional homicide rate (per 100,000 people)":"VC.IHR.PSRC.P5",
"Consumer price inflation (%)":"FP.CPI.TOTL.ZG", "Central government debt (% of GDP)":"GC.DOD.TOTL.GD.ZS",
"Total reserves including gold (current US$)":"FI.RES.TOTL.CD", "External balance on goods/services (% of GDP)":"NE.RSB.GNFS.ZS"}
UNITS = {"SP.DYN.LE00.IN":"years","SL.UEM.TOTL.NE.ZS":"% of labor force","NY.GDP.PCAP.CD":"current US$ per person","SP.DYN.CBRT.IN":"births per 1,000 people","SM.POP.NETM":"people","VC.IHR.PSRC.P5":"per 100,000 people","FP.CPI.TOTL.ZG":"%","GC.DOD.TOTL.GD.ZS":"% of GDP","FI.RES.TOTL.CD":"current US$","NE.RSB.GNFS.ZS":"% of GDP"}
@st.cache_data(ttl=3600, show_spinner=False)
def fetch(country, indicator):
    end=date.today().year-1; start=date.today().year-70
    url=f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}?date={start}:{end}&format=json&per_page=200"
    r=requests.get(url,timeout=25); r.raise_for_status(); payload=r.json()
    if not isinstance(payload,list) or len(payload)<2 or not payload[1]: return pd.DataFrame(),url
    rows=[{"Year":int(x["date"]),"Value":x["value"]} for x in payload[1] if x.get("value") is not None]
    return pd.DataFrame(rows).sort_values("Year"),url

with st.sidebar:
    st.header("Data and model")
    country_name=st.selectbox("Country",list(COUNTRIES))
    indicator_name=st.selectbox("Indicator",list(INDICATORS))
    degree=st.slider("Polynomial degree",3,8,3)
    sample_step=st.slider("Graph sample increment (years)",1,10,2)
    horizon=st.slider("Forward display horizon (years)",0,50,10)
    comparison=st.selectbox("Comparison country",["None"]+list(COUNTRIES))
    printer=st.checkbox("Printer-friendly report view")
country=COUNTRIES[country_name]; indicator=INDICATORS[indicator_name]; unit=UNITS[indicator]
try:
    source,url=fetch(country,indicator)
except Exception as e:
    st.error(f"World Bank request failed: {e}. Check connection, API availability, and logs."); st.stop()
if source.empty:
    st.warning("The API returned no observations for this country/indicator/window. No values were fabricated."); st.stop()
st.caption(f"Source: World Bank API · retrieved {date.today().isoformat()} · requested {date.today().year-70}–{date.today().year-1}.")
st.caption(f"Coverage: {source.Year.min()}–{source.Year.max()} · {len(source)} non-missing observations. GDP per capita is not household income or average wealth; crude birth rate is a rate, not a count; net migration is not an emigration count.")
if country=="BR": st.info("Brazil is included as a Latin American comparison; it is not a Spanish-speaking country.")

if "working" not in st.session_state or st.session_state.get("data_key")!=(country,indicator):
    st.session_state.working=source.copy(); st.session_state.data_key=(country,indicator)
st.subheader("Editable working data")
edited=st.data_editor(st.session_state.working,num_rows="dynamic",use_container_width=True,key="editor")
if st.button("Reset to source"):
    st.session_state.working=source.copy(); st.rerun()
st.session_state.working=edited.copy()
valid=edited.copy()
try:
    valid["Year"]=pd.to_numeric(valid["Year"],errors="raise")
    valid["Value"]=pd.to_numeric(valid["Value"],errors="raise")
    if valid.isna().any().any() or not np.all(valid.Year==valid.Year.astype(int)): raise ValueError("Years must be whole numbers and values cannot be blank.")
    valid["Year"]=valid.Year.astype(int)
    if valid.Year.duplicated().any(): raise ValueError("Duplicate years are not allowed.")
    if indicator in ("SP.DYN.LE00.IN","SP.DYN.CBRT.IN","VC.IHR.PSRC.P5","FI.RES.TOTL.CD") and (valid.Value<0).any(): raise ValueError("This indicator cannot be negative.")
except Exception as e:
    st.error(f"Working data invalid: {e}"); st.stop()
modified=not valid.equals(source)
if modified: st.warning("User-modified scenario: results and exports use edited values.")
valid=valid.sort_values("Year")
if len(valid)<degree+1: st.error(f"Need at least {degree+1} distinct valid years for degree {degree}."); st.stop()
base=int(valid.Year.min()); X=(valid.Year.to_numpy()-base).reshape(-1,1); y=valid.Value.to_numpy()
poly=PolynomialFeatures(degree=degree,include_bias=False); Xp=poly.fit_transform(X)
model=LinearRegression().fit(Xp,y); fitted=model.predict(Xp)
r2=r2_score(y,fitted); mae=mean_absolute_error(y,fitted)
def predict(years):
    return model.predict(poly.transform((np.asarray(years,dtype=float).reshape(-1,1)-base)))
coef=[model.intercept_]+list(model.coef_)
equation=f"y = {coef[0]:.8g}"+ "".join(f" {v:+.8g}(year−{base})^{i}" for i,v in enumerate(coef[1:],1))
st.subheader("Model summary")
st.write(f"**Centered-time equation:** `{equation}`")
st.write(f"Base year: {base} · degree {degree} · in-sample R²: {r2:.4f} · in-sample MAE: {mae:.4g} {unit} · observations: {len(valid)} · historical mean: {y.mean():.4g} {unit}")
st.caption("Polynomial fits describe observed data; extrapolation can become implausible. MAE/residuals describe in-sample error, not a validated prediction interval.")
left,right=st.columns(2)
with left:
    yr=st.number_input("Single year (1950–2100)",1950,2100, int(valid.Year.max()))
    if st.button("Calculate Value"):
        st.session_state.single=(int(yr),float(predict([yr])[0]),(country,indicator,degree,tuple(valid.Year),tuple(valid.Value)))
    if "single" in st.session_state:
        sy,sv,sk=st.session_state.single
        if sk==(country,indicator,degree,tuple(valid.Year),tuple(valid.Value)):
            st.success(f"{sy}: {sv:.6g} {unit} · {'within' if valid.Year.min()<=sy<=valid.Year.max() else 'outside'} fitted range")
        else: st.info("Previous single-year result invalidated by changed model/data.")
with right:
    mode=st.radio("Batch input",["Year range","Custom years"],horizontal=True)
    if mode=="Year range":
        a,b=st.slider("Year range",1950,2100,(int(valid.Year.min()),int(valid.Year.max())))
        step=st.slider("Range step",1,10,1)
        years=list(range(a,b+1,step))
    else:
        raw=st.text_input("Comma-separated years","2000,2010,2020")
        try: years=sorted(set(int(s.strip()) for s in raw.split(",") if s.strip()))
        except ValueError: years=[]; st.error("Enter comma-separated whole-number years.")
    years=[z for z in years if 1950<=z<=2100]
    results=pd.DataFrame({"Year":years,"Estimate":predict(years) if years else []})
    st.dataframe(results,use_container_width=True)

fig,ax=plt.subplots(figsize=(10,4.8))
ax.scatter(valid.Year,valid.Value,label="Observed / working data",color="black")
grid=np.linspace(valid.Year.min(),valid.Year.max()+horizon,500)
ax.plot(grid,predict(grid),label="Polynomial fit (smooth)",color="#2878b5")
sample=valid.iloc[::sample_step]
ax.scatter(sample.Year,predict(sample.Year),label=f"Fit markers every {sample_step} year(s)",s=24)
ax.axvline(valid.Year.max(),linestyle="--",color="gray",label="Historical boundary")
if years: ax.scatter(years,predict(years),marker="x",label="Batch estimates",color="red")
ax.set(xlabel="Year",ylabel=unit,title=f"{country_name} · {indicator_name}"); ax.grid(alpha=.2); ax.legend()
st.pyplot(fig)
if comparison!="None":
    try:
        comp,_=fetch(COUNTRIES[comparison],indicator)
        if not comp.empty:
            st.subheader("Same-indicator comparison")
            fig2,ax2=plt.subplots(figsize=(9,3.5))
            ax2.plot(valid.Year,valid.Value,label=country_name); ax2.plot(comp.Year,comp.Value,label=comparison,alpha=.8)
            ax2.set(xlabel="Year",ylabel=unit); ax2.legend(); ax2.grid(alpha=.2); st.pyplot(fig2)
            st.caption(f"Coverage differs: {country_name} {valid.Year.min()}–{valid.Year.max()}, {comparison} {comp.Year.min()}–{comp.Year.max()}. Missing years are not imputed.")
    except Exception as e: st.warning(f"Comparison unavailable: {e}")

# Context/function lab: safe expression parsing via SymPy, no eval.
st.subheader("Function lab and calculus")
course=st.selectbox("Mathematics pathway",["Honors Algebra 2","Precalculus","Calculus"])
context=st.selectbox("Context",["Business","Engineering","Medical","Sports science","Biology","Bioengineering","Economic forecasting","Weather","Custom"])
suggestions={"Business":"a*x+b","Engineering":"a*x**2+b*x+c","Medical":"a*exp(-b*x)","Sports science":"a*sin(b*x)","Biology":"a*exp(b*x)","Bioengineering":"a/(1+b*x)","Economic forecasting":"a*x**2+b*x+c","Weather":"a*sin(b*x+c)","Custom":"a*x+b"}
import sympy as sp
x=sp.symbols("x",real=True)
expr_text=st.text_input("Function f(x) (use x, + - * / **, sin, cos, exp, log)",suggestions[context])
a=st.number_input("Parameter a",value=2.0); b=st.number_input("Parameter b",value=0.5); c=st.number_input("Parameter c",value=0.0)
try:
    expr=sp.sympify(expr_text,locals={"x":x,"a":sp.Float(a),"b":sp.Float(b),"c":sp.Float(c)})
    if expr.free_symbols-{x}: raise ValueError("Use only x and parameters a, b, c.")
    f=sp.lambdify(x,expr,"numpy"); d1=sp.diff(expr,x); d2=sp.diff(d1,x)
    lo,hi=st.slider("Analysis interval",min_value=-20.0,max_value=20.0,value=(0.0,10.0))
    point=st.slider("Current x",float(lo),float(hi),float((lo+hi)/2))
    av1,av2=st.slider("Average-rate interval",float(lo),float(hi),(float(lo),float(hi)))
    st.latex(sp.latex(sp.Eq(sp.Symbol("f(x)"),expr)))
    if course=="Calculus":
        st.latex(r"f'(x)="+sp.latex(d1)); st.latex(r"f''(x)="+sp.latex(d2))
    val=float(f(point)); slope=float(sp.lambdify(x,d1,"numpy")(point))
    st.write(f"At x={point:.3g}: f(x)={val:.5g}; instantaneous rate f′(x)={slope:.5g}.")
    if av1!=av2: st.write(f"Average rate on [{av1:g}, {av2:g}] = {(float(f(av2))-float(f(av1)))/(av2-av1):.5g}.")
    xx=np.linspace(lo,hi,500); yy=np.asarray(f(xx),dtype=float)
    if yy.ndim==0: yy=np.full_like(xx,yy)
    yy=np.where(np.isfinite(yy),yy,np.nan)
    fig3,ax3=plt.subplots(figsize=(9,4)); ax3.plot(xx,yy,label="f(x)")
    ax3.scatter([point],[val],label="Current point",color="red")
    ax3.plot(xx,val+slope*(xx-point),"--",label="Tangent")
    if av1!=av2: ax3.plot([av1,av2],[float(f(av1)),float(f(av2))],"o-",label="Secant")
    ax3.legend(); ax3.grid(alpha=.2); st.pyplot(fig3)
    st.caption("Numerical display; domain gaps may appear as breaks. Interpret units from the chosen context; the derivative has dependent-units per independent-unit.")
except Exception as e: st.error(f"Function could not be evaluated: {e}. Check syntax and domain (for example, log(x) requires x > 0).")

st.subheader("Export")
st.download_button("Download working data CSV",valid.to_csv(index=False).encode(), "working_data.csv","text/csv")
xlsx=io.BytesIO()
with pd.ExcelWriter(xlsx,engine="xlsxwriter") as writer:
    pd.DataFrame({"Item":["Country","Indicator","Unit","Degree","Base year","Equation","R² (in-sample)","MAE (in-sample)","Edited scenario"],"Value":[country_name,indicator_name,unit,degree,base,equation,r2,mae,modified]}).to_excel(writer,sheet_name="Summary",index=False)
    valid.to_excel(writer,sheet_name="Working data",index=False); results.to_excel(writer,sheet_name="Calculated years",index=False)
st.download_button("Download Excel workbook",xlsx.getvalue(),"explorer.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
pdf=io.BytesIO(); cpdf=canvas.Canvas(pdf,pagesize=letter)
cpdf.setFont("Helvetica-Bold",15); cpdf.drawString(45,750,"Hispanic Heritage Historical Data Explorer")
cpdf.setFont("Helvetica",9)
lines=[f"Country: {country_name} | Indicator: {indicator_name}",f"Unit: {unit} | Retrieved: {date.today().isoformat()} | Edited scenario: {modified}",f"Coverage: {valid.Year.min()}–{valid.Year.max()} | n={len(valid)} | degree={degree} | base={base}",equation,f"In-sample R2={r2:.5f}; MAE={mae:.5g}", "Polynomial extrapolation is uncertain; no prediction interval is asserted."]
for i,line in enumerate(lines): cpdf.drawString(45,720-i*18,line[:115])
cpdf.save()
st.download_button("Download PDF report",pdf.getvalue(),"explorer_report.pdf","application/pdf")
