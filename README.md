# Finished-Vehicle Transporter Routing Problem (FVTRP) Decision Support System

> **Prototype Web-Based Decision Support System for Automotive Outbound Logistics**  
> Based on the research paper:  
> *Hu, Z.-H., Zhao, Y., Tao, S., & Sheng, Z.-H. (2015). "Finished-vehicle transporter routing problem solved by loading pattern discovery." Annals of Operations Research, 234(1), 37–56.*

---

## 🎯 Scope Alignment (Step 3 Prototype DSS)

| Step | Objective | Implementation |
|---|---|---|
| **Step 1: Real-life Case** | FVL Outbound Logistics | Modeled on **Anji Logistics** (Shanghai RVDC, Anting Auto City) |
| **Step 2: Documentation** | Business Requirements, Flowcharts, Test Cases | Full BRD, Prins-Split DAG pseudocode, Table 1-3 validation suite |
| **Step 3: Web-based Prototype DSS** | Data Input, Running the Model, Visualizing Results | **Interactive Streamlit Web Dashboard with Heuristics & Map Studio** |

---

## 🚀 Key Capabilities

### 1. 🔬 Heuristics Optimization Studio (Before vs. After Comparison)
Allows users to compare unoptimized baseline solutions against **4 distinct heuristics** (plus a multi-heuristic pipeline):
1. **Heuristic 1: 2-Opt Edge Uncrossing (Intra-Route)** — Reverses subsegments to eliminate route intersections.
2. **Heuristic 2: Clarke & Wright Savings (Route Merge)** — Merges direct delivery routes based on travel savings while verifying multi-type vehicle deck loading patterns.
3. **Heuristic 3: Inter-Route Customer Relocate (Shift)** — Re-allocates dealers between transporter routes to remove detours.
4. **Heuristic 4: Cross-Route Customer Swap (Osman)** — Swaps pairs of dealers across carriers to reduce joint travel costs and improve deck stowage.
5. **Heuristic 5: Full Multi-Heuristic Pipeline** — Sequentially runs Relocate $\to$ Swap $\to$ 2-Opt.

**Comparative Visuals:**
- Side-by-side KPI cards with time saved ($\Delta$ min) and percentage reduction.
- Dual-map view showing initial vs. heuristic-improved routes.
- Step-by-step heuristic optimization trajectory chart.
- Per-transporter delta table comparing sequence, travel time, and deck utilization.

### 2. 🚚 Dynamic Transporter Carrier Capacity Controls
- Fleet capacity presets (Standard 13-16 cars, High-Capacity 18-24 cars, Compact 8-12 cars).
- Global capacity scale slider ($0.5\times$ to $2.5\times$).
- Editable capacity table for Type 1 (Sedans), Type 2 (SUVs), and Type 3 (MPVs) per carrier pattern.

### 3. 🗺️ Multi-Engine Map Visualization (100% Free, Zero API Keys)
- **Folium Map** with OpenStreetMap, Esri World Street, and Esri Satellite Imagery.
- **Plotly Native Map** running token-free via WebGL.
- **2D Planar Coordinate Grid** reproducing Figures 4 & 5 from the paper.

---

## 🛠️ Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```
Open your browser at **`http://localhost:8501`**.
