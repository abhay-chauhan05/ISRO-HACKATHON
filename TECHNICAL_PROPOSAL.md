# ThermaSense AI — Written Technical Proposal
### A Physics-Informed Geospatial AI/ML Framework for Urban Heat Diagnosis and Cooling Intervention Optimization

*Submission for the Idea Submission Phase — ISRO National Hackathon*
*Problem Statement: Optimizing Urban Heat Mitigation and Cooling Strategies via AIML*

---

## 1. Technical Abstract

Rising land surface temperatures (LST) and intensifying Urban Heat Island (UHI) effects pose an escalating public-health, energy, and climate-resilience burden across Indian cities, where surface UHI intensities of 4–10 °C are routinely observed. Existing decision-support approaches are inadequate: purely statistical machine-learning models overfit and violate physical conservation laws when extrapolated to unseen urban morphologies or hypothetical mitigation scenarios, while high-fidelity microclimate simulators (e.g., ENVI-met) are computationally intractable at city scale. This proposal presents **ThermaSense AI**, an end-to-end geospatial AI/ML framework that couples satellite Earth Observation with the physics of the land-surface energy balance to diagnose heat, attribute its drivers, and prescribe spatially explicit, budget-aware cooling interventions.

The system fuses multi-source geospatial data on a harmonized analysis grid. **Land surface temperature** is retrieved from **Landsat 8/9 TIRS** (30–100 m) via a Split-Window Algorithm with NDVI-threshold emissivity correction, and from **ECOSTRESS** (~70 m, sub-daily) to resolve the diurnal — critically the nocturnal — thermal cycle. **Sentinel-2** (10–20 m) provides land-use/land-cover (LULC) and spectral indices (NDVI, NDBI, NDWI); **ERA5** reanalysis and **CPCB** stations supply meteorological forcing and validation; **OSM** and **GHSL/UT-GLOBUS** furnish three-dimensional urban form (building height, footprint, Sky-View Factor, Frontal Area Index, canyon aspect ratio).

The core scientific innovation is the integration of the **surface energy balance equation** — `Rn = H + LE + G`, with `Rn = (1−α)S↓ + εL↓ − εσT⁴` — directly into the loss function of a **Physics-Informed Neural Network (PINN)**. Trained on a composite objective combining a data-fidelity term (satellite LST mismatch) and a physics-residual term (energy-balance violation), the PINN learns a thermodynamically consistent mapping between LST and its contributing factors (albedo, vegetation fraction, morphology, meteorology). This yields a **fast, physics-constrained surrogate model** that generalizes to counterfactual scenarios and executes ~10³–10⁴× faster than full microclimate solvers.

Driver quantification is achieved via **SHAP attribution** and **Geographically Weighted Regression (GWR)**, producing a per-hotspot "heat driver fingerprint." Cooling scenarios — urban greening, cool/green roofs, high-albedo pavements, and blue infrastructure — are simulated by perturbing surface parameters through the surrogate, anchored and validated against **SOLWEIG** (radiation geometry, Mean Radiant Temperature) and the **InVEST Urban Cooling** model (ecosystem-service cooling). Finally, a multi-objective **genetic algorithm (NSGA-II)** solves the spatial resource-allocation problem, maximizing exposure-weighted temperature reduction under budget and land-use constraints.

**Deliverables** comprise validated, high-resolution day/night heat-stress maps; quantified, spatially explicit driver assessments; a PINN validated against held-out LST and CPCB records (R², RMSE in °C, plus a physics-consistency metric); and an optimization engine that outputs, for each intervention, its **type** (e.g., green roofs, albedo change), its **exact spatial placement**, and the **estimated temperature reduction (°C)** — delivered through an interactive planner dashboard aligned with the Smart Cities Mission and NDMA Heat Action Plans.

---

## 2. Complete Technical Workflow & System Architecture (Methodology)

The system is architected as a four-phase, modular pipeline. Each phase produces validated intermediate artifacts consumed by the next, enabling independent testing and de-risked development.

```
[ Phase 1 ]            [ Phase 2 ]                 [ Phase 3 ]                [ Phase 4 ]
Data Ingestion   ->    Hotspot ID &         ->     Physics-Informed    ->     Scenario Simulation
& Preprocessing        Driver Analysis             ML (PINN Surrogate)        & Optimization
     |                      |                            |                          |
Harmonized Grid       Heat-Stress Maps +         Physics-consistent         Optimal Intervention
(analysis-ready)      Driver Fingerprints        LST surrogate f(·)         Plan (type/place/°C)
```

---

### Phase 1 — Data Ingestion & Preprocessing

**Objective:** Convert heterogeneous, multi-resolution, multi-temporal geospatial sources into a single analysis-ready data cube on a common spatial grid and temporal reference.

**1.1 Multi-Source Acquisition**

- **Remote Sensing — Thermal:**
  - **Landsat 8/9 (TIRS + OLI):** Thermal infrared bands (B10/B11) for LST; optical bands for surface classification. Native thermal 100 m (resampled to 30 m); 16-day revisit.
  - **ECOSTRESS (ISS):** ~70 m LST at variable sub-daily overpass times — the key source for **diurnal and nocturnal** thermal dynamics that fixed-time Landsat cannot capture.
- **Remote Sensing — Optical / LULC:**
  - **Sentinel-2 (MSI):** 10–20 m multispectral for LULC classification and spectral indices.
- **Meteorological Forcing:**
  - **ERA5 reanalysis:** Air temperature (2 m), relative humidity, wind speed/direction, incoming shortwave/longwave radiation (hourly, ~0.25° / ~9 km).
  - **CPCB ground stations:** Point observations of air temperature/humidity for calibration and independent validation.
- **Urban Form (2D + 3D):**
  - **OSM:** Street network, building footprints, land-use polygons.
  - **GHSL / UT-GLOBUS:** Built-up surface, building height and volume for morphological parameterization.

**1.2 Sensor-Specific Preprocessing**

- **Radiometric & atmospheric correction:** Convert Digital Numbers → Top-of-Atmosphere radiance → surface reflectance (Sentinel-2 L2A / Landsat surface-reflectance product).
- **LST retrieval:** Apply the **Split-Window Algorithm** to Landsat TIRS; derive **land surface emissivity (ε)** via the **NDVI-Threshold Method** (NDVI → fractional vegetation cover → ε). Convert brightness temperature → LST.
- **Cloud & shadow masking:** Apply QA/scene-classification bands (e.g., Fmask, Sentinel-2 SCL) to remove contaminated pixels.

**1.3 Spatial Harmonization**

- **Common grid definition:** Reproject all layers to a single projected CRS (e.g., UTM zone) and resample to a **unified target resolution (30 m)**.
- **Resampling strategy:** Continuous fields (LST, NDVI) by bilinear/cubic; categorical fields (LULC) by nearest-neighbor/majority.
- **Downscaling:** Sharpen coarse LST using high-resolution predictors (LULC, NDVI, morphology) so effective resolution matches the analysis grid; **effective resolution is reported transparently**.

**1.4 Temporal Alignment**

- **Temporal windowing:** Composite scenes within matched seasonal windows (e.g., pre-monsoon peak-heat months) to remove phenological/seasonal bias.
- **Meteo interpolation:** Temporally interpolate hourly ERA5 to satellite overpass times; spatially interpolate CPCB points (kriging/IDW) to the grid.
- **Gap-filling:** Reconstruct masked pixels via spatiotemporal interpolation to yield a continuous cube.

**Phase 1 Output:** An **analysis-ready data cube** — co-registered layers of LST (day/night), spectral indices, LULC, 3D morphology metrics, and meteorological fields — on a unified 30 m grid.

---

### Phase 2 — Hotspot Identification & Driver Analysis

**Objective:** Delineate statistically significant, exposure-weighted heat-stress hotspots and quantify the physical drivers responsible for heating at each location.

**2.1 Feature Engineering (Driver Variable Set)**

- **Surface parameters:** Albedo, emissivity (ε), impervious-surface fraction, vegetation fraction (from NDVI), NDBI, NDWI, soil-moisture proxy.
- **3D Urban Morphology:** Building height & density, **Sky-View Factor (SVF)**, **Frontal Area Index (FAI)**, canyon aspect ratio (H/W), anthropogenic-heat proxy.
- **Atmospheric covariates:** Air temperature, relative humidity, wind speed, incoming solar radiation.

**2.2 Hotspot Delineation**

- **Spatial-statistics clustering:** Apply the **Getis-Ord Gi\*** statistic to the LST surface to identify statistically significant hot/cold spatial clusters (rather than isolated hot pixels).
- **Anomaly thresholding:** Flag pixels where `LST > μ + nσ` to complement the cluster analysis.
- **Heat Stress Index (HSI):** Fuse LST, air temperature, and humidity into a composite heat-exposure index (proxy for **UTCI / heat index**); weight by **population exposure** (GHSL) to prioritize where heat harms the most people.
- **Diurnal separation:** Produce independent **day-time** and **night-time** hotspot maps (ECOSTRESS-driven) to capture heat *retention* by thermal mass.

**2.3 Driver Quantification (Explainable Attribution)**

- **Global & local attribution:** Train a gradient-boosted baseline (LST as target, drivers as features) and compute **SHAP (Shapley Additive exPlanations)** values → per-location, per-driver marginal contribution to LST.
- **Spatial non-stationarity:** Apply **Geographically Weighted Regression (GWR)** to demonstrate that driver influence varies across the city (vegetation-limited vs morphology-limited wards).
- **Permutation importance:** Cross-check global driver ranking.

**Phase 2 Output:** Ranked, exposure-weighted **heat-stress hotspot polygons** plus a **quantitative "driver fingerprint"** (% contribution of each factor) for each hotspot — enabling targeted rather than generic interventions.

---

### Phase 3 — Physics-Informed ML Modeling (PINN Surrogate)

**Objective:** Learn a thermodynamically consistent mapping from surface/morphology/meteorology inputs to LST, so the model can be queried as a fast, physics-respecting surrogate for scenario simulation.

**3.1 Governing Physics — The Surface Energy Balance**

The land-surface energy balance dictates that net radiation is partitioned into turbulent and storage fluxes:

- **`Rn = H + LE + G`**
  where `Rn` = net radiation, `H` = sensible heat flux, `LE` = latent heat flux (evapotranspiration), `G` = ground/storage heat flux.
- **Net radiation:** **`Rn = (1 − α)·S↓ + ε·L↓ − ε·σ·T⁴`**
  where `α` = surface albedo, `S↓` = incoming shortwave, `ε` = emissivity, `L↓` = incoming longwave, `σ` = Stefan–Boltzmann constant, `T` = LST.

These equations analytically link **LST (T)** to the very parameters interventions modify — **albedo (α)** for cool roofs/pavements, and **LE** for vegetation/water (evapotranspiration).

**3.2 Network Architecture**

- **Inputs (feature vector per grid cell):** albedo, emissivity, vegetation fraction/NDVI, impervious fraction, SVF, FAI, building height/density, aspect ratio; meteorological forcing (air temperature, humidity, wind, `S↓`, `L↓`); spatial coordinates.
- **Backbone:** A fully connected deep neural network (feed-forward MLP, ~4–8 hidden layers, tanh/Swish activations); optionally a **spatially-aware CNN/Graph-NN** variant to exploit neighborhood context (e.g., shading from adjacent tall buildings).
- **Output:** Predicted **LST (T̂)** per grid cell (extendable to predicted flux terms H, LE, G for full closure).

**3.3 Physics-Constrained Composite Loss**

The network is trained to minimize:

- **`L_total = L_data + λ·L_physics + L_boundary`**
  - **`L_data`** — Mean squared error between predicted `T̂` and satellite-observed LST (data fidelity).
  - **`L_physics`** — The **energy-balance residual**: the network's predicted state is substituted into `Rn − (H + LE + G)` (and the radiative form of `Rn`), and the squared residual is penalized. This forces predictions to **conserve energy**.
  - **`L_boundary`** — Enforces known physical constraints/limits (e.g., non-negativity, plausible flux ranges, boundary conditions).
  - **`λ`** — Adaptive weighting balancing data vs physics terms (self-adaptive / gradient-normalization schemes to prevent one term dominating).

**3.4 Why the Physics Constraint Matters**

- A black-box model interpolates within observed data but **hallucinates under extrapolation** — e.g., predicting an implausible response when roof albedo is raised from 0.15 → 0.60, a value never seen in training.
- The physics-residual term guarantees the learned function **obeys radiation and heat-exchange laws**: raising `α` *must* reduce absorbed shortwave `(1−α)S↓`; increasing vegetation *must* raise latent-heat cooling `LE`. The model **cannot violate thermodynamics**, making its counterfactual predictions scientifically defensible.

**3.5 Training, Validation & Consistency Checks**

- **Data split:** Spatial and temporal hold-out (train on subset of wards/scenes, test on unseen ones) to test true generalization.
- **Metrics:** **R², RMSE (°C), MAE** against held-out satellite LST and independent **CPCB** air-temperature relationships.
- **Physics-consistency metric:** Report the mean energy-balance residual on the test set — a validation dimension unavailable to purely statistical models.

**Phase 3 Output:** A **validated, physics-consistent LST surrogate** `T̂ = f(surface, morphology, meteo)` executing in milliseconds per query.

---

### Phase 4 — Scenario Simulation (SOLWEIG / InVEST) & Optimization

**Objective:** Use the fast surrogate to simulate cooling interventions and solve for the optimal, budget-constrained spatial strategy with quantified °C reduction.

**4.1 Scenario Construction (Parameter Perturbation)**

For a candidate intervention, the relevant land-surface parameters in the input cube are modified, then re-evaluated through the PINN to obtain a predicted **ΔLST** map:

- **🌳 Urban greening / green roofs:** ↑ vegetation fraction / NDVI → ↑ `LE` (evapotranspiration) + shading.
- **🏠 Cool / white roofs:** ↑ roof albedo (`α`: ~0.1 → 0.6) → ↓ absorbed shortwave `(1−α)S↓`.
- **🎨 High-albedo / permeable pavements:** ↑ pavement albedo and ↓ stored ground flux `G`.
- **💧 Blue infrastructure (water bodies, fountains):** ↑ `LE` + thermal-mass buffering.

**4.2 Physics-Model Coupling (Anchoring & Validation)**

- **SOLWEIG:** Simulates 3D shortwave/longwave radiation fluxes, shadow patterns, and **Mean Radiant Temperature (Tmrt)** at street level → grounds the surrogate in radiation geometry and enables **thermal-comfort (UTCI)** outputs, not just LST.
- **InVEST Urban Cooling Model:** Quantifies **ecosystem-service cooling** (shade, evapotranspiration, albedo contribution) at planning scale → validates the greening/blue-infrastructure scenarios.
- **Role:** Selected benchmark scenarios are run in SOLWEIG/InVEST and compared to surrogate predictions (agreement metrics on ΔLST / ΔTmrt) to **certify** the surrogate before it is used inside the optimization loop.

**4.3 Optimization Engine**

- **Problem formulation:** A constrained, multi-objective **spatial resource-allocation / combinatorial optimization** problem over the set of candidate intervention placements.
- **Objective function:**
  - **`maximize  Σ_i ( ΔLST_i × exposure_i )  −  cost_penalty`**
    (exposure-weighted total cooling; may be extended to a second objective for comfort ΔUTCI).
- **Constraints:**
  - Total cost ≤ **budget B**;
  - Per-zone **land-use feasibility** (can't green a highway median beyond capacity);
  - **Minimum intervention area** thresholds;
  - **Equity constraint** — prioritize high-exposure / vulnerable wards.
- **Solver:** **Genetic Algorithm — NSGA-II** for multi-objective search (cooling vs cost), with optional **Bayesian optimization** for continuous parameters. Each candidate solution is scored by rapid PINN surrogate queries, making thousands of evaluations tractable — infeasible with a full simulator.
- **Result:** A **Pareto-optimal frontier** of intervention plans; the planner selects a budget point and receives an explicit plan.

**Phase 4 Output — Final Deliverable:** For the chosen budget, a georeferenced **optimal intervention plan** specifying, per location:
1. **Intervention type** (green roof, high-albedo surface, tree canopy, water body);
2. **Exact spatial placement** (grid-cell / parcel map);
3. **Estimated temperature reduction (°C)** and cost — surfaced through an interactive **web-GIS planner dashboard**.

---

## 3. Mapping to Required Hackathon Outcomes

| Required Outcome | Delivered By | Artifact |
|---|---|---|
| Heat Stress Maps | Phase 2 | Day/night exposure-weighted HSI + LST hotspot maps |
| Quantitative driver assessment | Phase 2 | SHAP driver fingerprints + GWR spatial attribution |
| Validated AI/ML model | Phase 3 | PINN with R²/RMSE(°C) + physics-consistency metric |
| Scenario-based evaluation | Phase 4 | ΔLST / ΔUTCI maps per intervention (SOLWEIG/InVEST-anchored) |
| Optimal intervention strategy | Phase 4 | Pareto-optimal placement map: type + location + °C reduction |
