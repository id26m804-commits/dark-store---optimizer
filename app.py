import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial.distance import cdist

st.set_page_config(page_title="Dark Store Location Optimizer", page_icon="📍", layout="wide")
st.title("📍 Dark Store Location Optimization — Chennai")
st.caption("Facility-location decision support with a maximum customer-to-store distance constraint.")

with st.sidebar:
    st.header("Model inputs")
    n_candidates = st.number_input("Dark store candidate locations", min_value=5, max_value=200, value=50, step=5)
    n_customers = st.number_input("Number of customers", min_value=100, max_value=50000, value=10000, step=500)
    max_distance = st.number_input("Maximum customer-to-store distance (km)", min_value=0.5, max_value=30.0, value=5.0, step=0.5)
    max_open = st.slider("Maximum stores allowed to open", min_value=1, max_value=min(25, int(n_candidates)), value=min(10, int(n_candidates)))
    seed = st.number_input("Random seed (reproducible)", min_value=0, max_value=99999, value=42, step=1)
    st.divider()
    st.subheader("Economic assumptions")
    margin = st.number_input("Contribution per served customer per day (₹)", min_value=1, max_value=2000, value=120, step=10)
    fixed_cost = st.number_input("Fixed cost per open store per day (₹)", min_value=0, max_value=200000, value=25000, step=1000)
    delivery_cost = st.number_input("Delivery cost per customer-km (₹)", min_value=0.0, max_value=100.0, value=8.0, step=1.0)
    st.caption("Illustrative assumptions; replace with business data when available.")

# Approximate Chennai study-area bounds; coordinates are synthetic samples within the rectangle.
LAT_MIN, LAT_MAX = 12.82, 13.20
LON_MIN, LON_MAX = 80.10, 80.32

def haversine_matrix(customers, stores):
    r = 6371.0088
    c_lat = np.radians(customers[:, 0])[:, None]
    s_lat = np.radians(stores[:, 0])[None, :]
    dlat = s_lat - c_lat
    dlon = np.radians(stores[:, 1])[None, :] - np.radians(customers[:, 1])[:, None]
    a = np.sin(dlat / 2)**2 + np.cos(c_lat) * np.cos(s_lat) * np.sin(dlon / 2)**2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

@st.cache_data(show_spinner=False)
def make_data(nc, nu, seed):
    rng = np.random.default_rng(int(seed))
    centers = np.array([[13.0827,80.2707],[13.0569,80.2425],[13.0108,80.2350],
                        [13.1067,80.2206],[12.9249,80.1000],[13.0358,80.2610],
                        [12.9716,80.2210],[13.1350,80.2300]])
    weights = np.array([.22,.16,.12,.12,.08,.12,.10,.08])
    idx = rng.choice(len(centers), size=int(nu), p=weights)
    cust = centers[idx] + rng.normal(0, [0.018, 0.020], size=(int(nu),2))
    cust[:,0] = np.clip(cust[:,0], LAT_MIN, LAT_MAX)
    cust[:,1] = np.clip(cust[:,1], LON_MIN, LON_MAX)
    stores = np.column_stack([rng.uniform(LAT_MIN,LAT_MAX,int(nc)),
                              rng.uniform(LON_MIN,LON_MAX,int(nc))])
    return cust, stores

def evaluate(open_ids, distances, max_dist, margin, fixed_cost, delivery_cost):
    if not open_ids:
        return {"profit": 0.0, "served": 0, "coverage": 0.0, "stores": 0,
                "avg_distance": 0.0, "assignments": np.full(distances.shape[0], -1)}
    ds = distances[:, open_ids]
    local_idx = np.argmin(ds, axis=1)
    nearest = ds[np.arange(ds.shape[0]), local_idx]
    served_mask = nearest <= max_dist
    assignments = np.full(distances.shape[0], -1, dtype=int)
    assignments[served_mask] = np.asarray(open_ids)[local_idx[served_mask]]
    served_dist = nearest[served_mask]
    profit = served_mask.sum()*margin - len(open_ids)*fixed_cost - served_dist.sum()*delivery_cost
    return {"profit": float(profit), "served": int(served_mask.sum()),
            "coverage": float(served_mask.mean()*100), "stores": len(open_ids),
            "avg_distance": float(served_dist.mean()) if len(served_dist) else 0.0,
            "assignments": assignments}

def greedy_optimize(distances, max_dist, max_stores, margin, fixed_cost, delivery_cost):
    nstores = distances.shape[1]
    opened = []
    current = evaluate(opened, distances, max_dist, margin, fixed_cost, delivery_cost)
    history = []
    for step in range(min(max_stores, nstores)):
        best_id, best_result, best_gain = None, None, 0.0
        for j in range(nstores):
            if j in opened:
                continue
            result = evaluate(opened + [j], distances, max_dist, margin, fixed_cost, delivery_cost)
            gain = result["profit"] - current["profit"]
            if gain > best_gain:
                best_id, best_result, best_gain = j, result, gain
        if best_id is None:
            break
        opened.append(best_id)
        current = best_result
        history.append({k: current[k] for k in ["stores","served","coverage","avg_distance","profit"]})
    return opened, current, pd.DataFrame(history)

run = st.button("Run optimization", type="primary", use_container_width=True)
if run or "run_results" not in st.session_state:
    with st.spinner("Generating synthetic Chennai data and optimizing store locations…"):
        customers, candidates = make_data(int(n_candidates), int(n_customers), int(seed))
        distances = haversine_matrix(customers, candidates)
        opened, solution, history = greedy_optimize(distances, float(max_distance), int(max_open),
                                                    float(margin), float(fixed_cost), float(delivery_cost))
        st.session_state["run_results"] = (customers, candidates, distances, opened, solution, history,
                                           (int(n_candidates),int(n_customers),float(max_distance),int(max_open),
                                            int(seed),float(margin),float(fixed_cost),float(delivery_cost)))
else:
    customers, candidates, distances, opened, solution, history, old_inputs = st.session_state["run_results"]
    new_inputs=(int(n_candidates),int(n_customers),float(max_distance),int(max_open),int(seed),
                float(margin),float(fixed_cost),float(delivery_cost))
    if new_inputs != old_inputs:
        with st.spinner("Inputs changed — recalculating optimization…"):
            customers, candidates = make_data(int(n_candidates), int(n_customers), int(seed))
            distances = haversine_matrix(customers, candidates)
            opened, solution, history = greedy_optimize(distances, float(max_distance), int(max_open),
                                                        float(margin), float(fixed_cost), float(delivery_cost))
            st.session_state["run_results"]=(customers,candidates,distances,opened,solution,history,new_inputs)

customers, candidates, distances, opened, solution, history, old_inputs = st.session_state["run_results"]
st.info("The model uses synthetic customer and candidate coordinates within an approximate Chennai study area.")

if not opened:
    st.warning("No store has positive marginal profit under these assumptions.")
else:
    assigned = solution["assignments"]
    kpis = pd.DataFrame([
        ["Candidate locations", len(candidates)],
        ["Customers", len(customers)],
        ["Stores opened", len(opened)],
        ["Customers served", f'{solution["served"]:,}'],
        ["Coverage", f'{solution["coverage"]:.1f}%'],
        ["Average assigned distance", f'{solution["avg_distance"]:.2f} km'],
        ["Maximum allowed distance", f'{max_distance:.2f} km'],
        ["Estimated daily profit", f'₹{solution["profit"]:,.0f}'],
    ], columns=["KPI","Value"])
    st.subheader("1. Key performance indicators")
    st.dataframe(kpis, hide_index=True, use_container_width=True)

    st.subheader("2. Detailed solution map")
    map_df = pd.DataFrame(customers, columns=["lat","lon"])
    map_df["status"] = np.where(assigned >= 0, "Served customer", "Unserved customer")
    map_df["store"] = np.where(assigned >= 0, assigned.astype(str), "Unassigned")
    store_df = pd.DataFrame(candidates, columns=["lat","lon"])
    store_df["candidate"] = np.arange(len(store_df))
    store_df["status"] = np.where(store_df["candidate"].isin(opened), "Opened store", "Candidate (not opened)")
    
    fig = go.Figure()
    for status, color, size, opacity in [
        ("Served customer","#2A9D8F",4,0.45),("Unserved customer","#E76F51",4,0.55)]:
        sub=map_df[map_df.status==status]
        fig.add_trace(go.Scattermap(lat=sub.lat,lon=sub.lon,mode="markers",
            marker=dict(size=size,color=color,opacity=opacity),name=status,
            hovertemplate="Customer<br>Lat %{lat:.4f}<br>Lon %{lon:.4f}<extra></extra>"))
            
    opened_df=store_df[store_df.status=="Opened store"]
    other_df=store_df[store_df.status!="Opened store"]
    
    fig.add_trace(go.Scattermap(lat=other_df.lat,lon=other_df.lon,mode="markers",
        marker=dict(size=8,color="#888888",symbol="circle"),name="Candidate (not opened)",
        text=other_df.candidate,hovertemplate="Candidate %{text}<extra></extra>"))
        
    fig.add_trace(go.Scattermap(lat=opened_df.lat,lon=opened_df.lon,mode="markers",
        marker=dict(size=15,color="#264653",symbol="star"),name="Opened store",
        text=opened_df.candidate,hovertemplate="Opened store candidate %{text}<extra></extra>"))
        
    fig.update_layout(
        map_style="open-street-map",
        map=dict(center=dict(lat=13.05, lon=80.23), zoom=9.5),
        height=650,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=0.01)
    )
        
    st.plotly_chart(fig,use_container_width=True)

st.divider()
st.subheader("3. Profit for requested store limits (2, 5, 10, 20, 25 stores)")
st.caption("Re-optimizing for store limits 2, 5, 10, 20, and 25 using the current geography and economic inputs.")

compare_limits = [2, 5, 10, 20, 25]
compare_limits = [k for k in compare_limits if k <= len(candidates)]
rows = []
for k in compare_limits:
    op, sol, hist = greedy_optimize(distances, float(max_distance), k, float(margin), float(fixed_cost), float(delivery_cost))
    rows.append({
        "Maximum stores allowed": k, 
        "Stores opened": sol["stores"], 
        "Customers served": sol["served"],
        "Coverage (%)": round(sol["coverage"], 2), 
        "Estimated daily profit (₹)": round(sol["profit"], 2)
    })
comparison = pd.DataFrame(rows)

if len(comparison):
    chart = px.line(comparison, x="Maximum stores allowed", y="Estimated daily profit (₹)", markers=True,
                  title="Estimated daily profit vs. maximum number of stores allowed")
    chart.update_layout(height=400)
    st.plotly_chart(chart, use_container_width=True)
    st.dataframe(comparison, hide_index=True, use_container_width=True)

with st.expander("Model formulation and assumptions"):
    st.markdown("""
**Decision variables & constraints summary:**  
- Maximizes daily profit (revenue from served customers minus fixed store costs and delivery costs).  
- Enforces the strict maximum customer-to-store distance constraint.  
- Solved using a greedy heuristic approach suitable for large-scale facility location problems.
""")