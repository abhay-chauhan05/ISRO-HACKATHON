# 🛰️ ISRO Hackathon — Idea Submission Pitch Deck
## Optimizing Urban Heat Mitigation & Cooling Strategies via Geospatial AI/ML

> **Format note:** Each slide contains four blocks — **Slide Title**, **Visual Suggestion for Canva**, **Slide Content (on-slide bullets)**, and **Speaker Notes**. Copy the *Slide Content* block directly into Canva text frames; use *Visual Suggestion* to pick your Canva elements; keep *Speaker Notes* off-slide for your verbal pitch.

---

## SLIDE 1 — Title Slide & Team Vision

**🎨 Visual Suggestion for Canva**
- Full-bleed background: a dual-tone satellite thermal image of a city (deep blue → red gradient) faded to 30% opacity.
- Centered project title in bold, large sans-serif (e.g., Montserrat/Poppins).
- Bottom strip: team name, member roles as small icon chips, ISRO/hackathon logos aligned right.
- Accent: a thin animated-looking "scanline" divider to evoke satellite sensing.

**📝 Slide Content**
- **Project Title:** *ThermaSense AI — A Physics-Informed Geospatial Engine for Urban Heat Diagnosis & Cooling Optimization*
- **Tagline:** *"From satellite pixels to street-level cooling — decisions grounded in physics, powered by AI."*
- **Team:** [Team Name] | [Institution]
- **Domain:** Geospatial AI/ML • Remote Sensing • Urban Climate Physics
- **Problem Statement Ref:** Optimizing Urban Heat Mitigation & Cooling Strategies via AIML

**🎤 Speaker Notes**
Open with a one-line hook: "Every summer, Indian cities record surface temperatures 6–10°C hotter than their rural surroundings — and we currently have no scalable, physics-grounded way to decide *where* to intervene." Introduce ThermaSense AI as an end-to-end decision-support engine that doesn't just *map* heat, but *explains its physical drivers* and *prescribes the optimal, budget-aware cooling intervention with a predicted °C reduction*. Emphasize that our differentiator is fusing satellite remote sensing with the energy-balance physics of land surface temperature — not a black-box model.

---

## SLIDE 2 — The Problem: Urban Heat Island (UHI) Effect & Impact

**🎨 Visual Suggestion for Canva**
- Split screen. **Left:** a "surface energy balance" diagram of a city block — arrows for incoming shortwave, reflected (albedo), stored heat flux (concrete), latent heat (missing vegetation), sensible heat rising.
- **Right:** three stacked impact stat-cards with icons (health ❤️, energy ⚡, mortality 📈) each with a bold number.
- Bottom: a small line chart "Rural vs Urban night-time temperature" with a widening gap.

**📝 Slide Content**
- **The UHI mechanism:** Impervious surfaces (asphalt, concrete) have **low albedo + high thermal admittance** → absorb & re-radiate heat; loss of vegetation removes **evapotranspirative (latent heat) cooling**; urban canyon geometry **traps longwave radiation** and reduces sky-view factor.
- **Scale of impact:** Canopy/surface UHI intensity in Indian metros reaches **4–10°C**; heat is now the **deadliest climate hazard**, driving mortality, reduced labor productivity, and **20–30% higher peak cooling energy demand**.
- **The decision gap:** City planners lack **spatially explicit, quantitative** tools to answer *"which intervention, placed where, yields the maximum cooling per rupee?"*
- **Why status-quo tools fail:** Pure statistical models ignore physics (poor generalization); pure microclimate simulators (e.g., ENVI-met) are **computationally intractable at city scale**.

**🎤 Speaker Notes**
Frame the problem as *both* a public-health emergency *and* a decision-science vacuum. Stress the two failure modes of current approaches: statistical ML overfits and can't extrapolate to unseen morphologies or future scenarios, while high-fidelity physics simulators can model one street but never an entire city in reasonable time. Position our physics-informed approach as the bridge that captures the physical laws *and* scales. This slide sets up "why physics + ML" as the inevitable answer.

---

## SLIDE 3 — Our Solution Architecture: A Bird's Eye View

**🎨 Visual Suggestion for Canva**
- A horizontal **4-stage pipeline flow diagram** with connecting arrows: `Data Fusion → Diagnosis → Physics-Informed Modeling → Optimization`.
- Under each stage, 2–3 icon chips (satellite, brain, gears, map-pin).
- Use a distinct color per stage; keep a legend. Place a small "human-in-the-loop / planner dashboard" node at the output.

**📝 Slide Content**
- **Layer 1 — Data Fusion & Preprocessing:** Multi-sensor ingestion, cloud masking, atmospheric correction, spatial harmonization to a common grid.
- **Layer 2 — Hotspot Diagnosis:** LST retrieval, thermal anomaly detection, heat-stress indexing (day + night).
- **Layer 3 — Physics-Informed ML Core (PINN):** Learns the LST ↔ urban-morphology ↔ meteorology mapping, constrained by the **surface energy balance equation**.
- **Layer 4 — Scenario & Optimization Engine:** Perturbs land-surface parameters (albedo, vegetation fraction, water), predicts ΔLST, and solves for the **optimal spatial placement of interventions** under budget/area constraints.
- **Output:** Interactive **planner dashboard** — heat maps, driver attribution, ranked interventions with estimated °C reduction.

**🎤 Speaker Notes**
Give the judges the mental map for the next eight slides. Emphasize modularity: each layer is independently validated and swappable, which de-risks the build. Highlight that the physics-informed core (Layer 3) is the "brain" — it is trained once and then queried thousands of times inside the optimization loop, giving us physics-consistent predictions at a fraction of a simulator's cost. This is the architectural insight that makes city-scale optimization feasible.

---

## SLIDE 4 — Data Ingestion & Preprocessing

**🎨 Visual Suggestion for Canva**
- A **table/grid of dataset cards**, each card: dataset name, variable it provides, native resolution, and a small satellite/sensor icon.
- Right side: a vertical "preprocessing pipeline" ribbon showing steps as connected pills.
- Use a subtle world-grid background to reinforce "geospatial."

**📝 Slide Content**
- **Land Surface Temperature (LST):** **Landsat 8/9 TIRS** (100 m, resampled 30 m) via **Split-Window Algorithm**; **ECOSTRESS** (~70 m, sub-daily) for diurnal thermal dynamics.
- **Land Use / Land Cover (LULC):** **Sentinel-2** (10–20 m) + Landsat → classification into built-up / vegetation / water / bare soil; derive **NDVI, NDBI, NDWI**.
- **Meteorological forcing:** **ERA5 reanalysis** (air temp, humidity, wind, solar radiation) + **CPCB** ground stations for calibration/validation.
- **Urban morphology:** **UT-GLOBUS** & **GHSL** (building height, footprint, volume) + **OSM** (street network) → compute **Sky-View Factor, Frontal Area Index, canyon aspect ratio, building density**.
- **Preprocessing stack:** Cloud/shadow masking → atmospheric correction → **spatiotemporal co-registration to a unified analysis grid** → gap-filling → emissivity correction (NDVI-threshold method) → feature normalization.

**🎤 Speaker Notes**
Demonstrate data literacy — the judges are remote-sensing experts. Explain *why* each dataset is chosen: Landsat gives thermal + spatial detail, ECOSTRESS uniquely captures the *diurnal cycle* (critical because UHI peaks at night), Sentinel-2 provides fine LULC, and UT-GLOBUS/GHSL are the key to *3D urban form* — the variable most models ignore. Stress the hardest engineering problem: harmonizing sensors of different resolutions, revisit times, and projections onto one grid without introducing artifacts. Mention emissivity correction explicitly — it's where naive LST pipelines go wrong.

---

## SLIDE 5 — Phase 1: Identifying Hotspots & Geospatial Mapping

**🎨 Visual Suggestion for Canva**
- **Split screen.** Left: a 3D/2.5D city heat map (blue→red) with hotspot polygons outlined. Right: bullet points + a small legend for the Heat Stress Index tiers.
- Add a "day vs night" toggle mockup to show diurnal analysis.

**📝 Slide Content**
- **LST retrieval & normalization:** Convert TIRS brightness temperature → LST with emissivity correction; normalize across scenes to remove seasonal/acquisition bias.
- **Hotspot detection:** Statistical **Getis-Ord Gi\*** hotspot analysis + threshold anomaly detection (LST > μ + nσ) to delineate **statistically significant thermal clusters**.
- **Heat Stress Index (HSI):** Composite index fusing LST, air temperature, humidity (→ proxy for **UTCI/heat index**) and exposure (population from GHSL).
- **Diurnal analysis:** ECOSTRESS-driven day vs **night-time** hotspot mapping — captures heat *retention* by thermal mass, not just daytime absorption.
- **Output:** Ranked, geo-referenced hotspot polygons prioritized by **severity × human exposure**.

**🎤 Speaker Notes**
Clarify that "hotspot" is not just "hottest pixel" — it is a *statistically validated cluster* weighted by human exposure, so we prioritize where heat actually harms people. Highlight the night-time angle: many tools only use daytime Landsat, but the health burden of UHI is largely nocturnal (bodies can't recover). ECOSTRESS lets us capture that. This makes our prioritization defensibly rigorous and policy-relevant.

---

## SLIDE 6 — Phase 2: Analyzing Drivers of Urban Heating

**🎨 Visual Suggestion for Canva**
- Left: a **horizontal bar chart / SHAP-style plot** showing each driver's contribution to LST (albedo, vegetation fraction, building density, SVF, etc.).
- Right: a small correlation matrix heatmap thumbnail + bullet points.
- Icons for each driver category (leaf, building, road, cloud).

**📝 Slide Content**
- **Driver feature set:**
  - *Surface:* albedo, emissivity, impervious surface fraction, NDVI/vegetation fraction, soil moisture proxy.
  - *Morphology (3D):* building height, density, Sky-View Factor, Frontal Area Index, canyon aspect ratio, anthropogenic heat proxy.
  - *Atmospheric:* air temperature, relative humidity, wind speed, incoming solar radiation.
- **Attribution method:** **SHAP (SHapley Additive exPlanations)** values + permutation importance → *quantitative, per-location* driver contribution to LST.
- **Spatial heterogeneity:** Use **Geographically Weighted Regression (GWR)** to show drivers vary across the city (vegetation dominates in some wards, morphology in others).
- **Deliverable:** A **"heat driver fingerprint"** for every hotspot — the % contribution of each factor, enabling *targeted* (not one-size-fits-all) interventions.

**🎤 Speaker Notes**
This is the "explainability" slide — crucial for winning trust from scientists. We don't just predict heat; we *decompose* it. SHAP gives locally faithful attributions, and GWR proves drivers are spatially non-stationary — a nuance many teams miss. The payoff: for one hotspot the fix might be greening (low vegetation is the culprit), for another it's albedo (dark low-rise rooftops). Emphasize that this driver fingerprint is what makes the downstream optimization *intelligent* rather than generic.

---

## SLIDE 7 — Phase 3: The Core Tech — Physics-Informed ML Modeling ⭐

**🎨 Visual Suggestion for Canva**
- Centerpiece slide. A **diagram of a neural network** on the left whose output feeds into a **physics equation box** (the surface energy balance) on the right, with a feedback arrow labeled "physics residual loss."
- Below: an equation strip showing the total loss = data loss + physics loss.
- Use a glowing/accent treatment — this is the "wow" slide.

**📝 Slide Content**
- **The governing physics — Surface Energy Balance:**
  `Rn = H + LE + G` → Net radiation = Sensible + Latent + Ground heat flux; with `Rn = (1−α)S↓ + εL↓ − εσT⁴` (α = albedo, ε = emissivity, T = LST).
- **Why pure ML fails:** it interpolates within training data but **violates energy conservation** when extrapolating to unseen morphologies or future cooling scenarios → physically implausible predictions.
- **Our Physics-Informed Neural Network (PINN):** LST is modeled as a function of morphology + surface + meteo inputs; the network is trained with a **composite loss**:
  `L = L_data (satellite LST mismatch) + λ·L_physics (energy-balance residual) + L_boundary`
- **What the physics term enforces:** predictions must satisfy the energy-balance residual → e.g., increasing albedo *must* reduce absorbed shortwave; adding vegetation *must* increase latent-heat cooling. **The model cannot "cheat" against thermodynamics.**
- **Result:** A **surrogate model** that is (a) physically consistent, (b) generalizes to *counterfactual* scenarios, and (c) runs ~10³–10⁴× faster than full microclimate simulators.

**🎤 Speaker Notes**
Spend the most time here. The core intellectual claim: we embed the surface energy balance directly into the loss function as a soft constraint (a physics residual). So the network learns from satellite LST data *and* is penalized whenever it violates the physics of radiation and heat exchange. Explain the payoff concretely: when we later simulate "paint these roofs white (raise albedo from 0.15 → 0.6)," a black-box model has never seen that data and would hallucinate — but our PINN respects `Rn = (1−α)S↓ + ...`, so it produces a thermodynamically valid ΔLST. This is *exactly* why physics-informed ML is the right tool for scenario extrapolation, and it directly links LST to urban morphology through the physically meaningful parameters (albedo, SVF, vegetation fraction). This is the slide that separates us from every "we ran XGBoost on LST" team.

---

## SLIDE 8 — Phase 4: Scenario Generation & Cooling Interventions

**🎨 Visual Suggestion for Canva**
- A **before/after slider mockup** of a neighborhood: left = current LST, right = post-intervention LST (cooler).
- Four intervention "cards" with icons: 🌳 Urban Greening, 🏠 Cool Roofs, 🎨 Albedo/Pavement, 💧 Water Bodies.
- Small callouts to SOLWEIG (radiation/thermal comfort) and InVEST (ecosystem cooling) integration.

**📝 Slide Content**
- **Scenario library (parameter perturbations fed to the PINN surrogate):**
  - 🌳 **Urban greening** → ↑ vegetation fraction / NDVI → ↑ evapotranspiration (latent heat) + shading.
  - 🏠 **Cool roofs / white roofs** → ↑ roof albedo (0.1→0.6) → ↓ absorbed shortwave.
  - 🎨 **High-albedo / permeable pavements** → ↓ stored ground heat flux (G).
  - 💧 **Blue infrastructure (water bodies, fountains)** → evaporative + thermal-mass cooling.
- **Physics-model coupling:**
  - **SOLWEIG** → simulates 3D radiation fluxes, shadow patterns & **Mean Radiant Temperature** for thermal-comfort (UTCI) realism at street level.
  - **InVEST Urban Cooling model** → quantifies **ecosystem-service cooling** (shade, evapotranspiration, albedo) at planning scale.
- **Workflow:** For each scenario, modify surface/morphology parameters → query PINN + SOLWEIG/InVEST → produce a **predicted ΔLST and ΔUTCI map**.

**🎤 Speaker Notes**
Explain the elegant loop: instead of physically rebuilding the city or running weeks of ENVI-met simulations, we *perturb the input parameters* of our validated PINN surrogate and instantly get a physically consistent cooling map. SOLWEIG and InVEST aren't replaced — they *anchor and validate* our surrogate for radiation geometry and ecosystem services respectively, giving credibility and human-comfort (UTCI) grounding. Make clear each intervention maps to a specific physical term in the energy balance from Slide 7 — greening → LE, cool roofs → albedo term, pavements → G, water → LE + thermal mass. This coherence is what impresses climate scientists.

---

## SLIDE 9 — Optimization Engine: The Best Intervention, Placed Right

**🎨 Visual Suggestion for Canva**
- A city map with **optimally placed intervention markers** (green patches, blue dots, white roof zones) and a side panel showing a "cost vs cooling" Pareto curve.
- Include a compact objective-function box and a constraints list with checkmark icons.

**📝 Slide Content**
- **Problem framing:** Spatial resource-allocation / combinatorial optimization — *maximize total °C reduction (and exposure-weighted comfort gain) subject to budget, land availability, and feasibility constraints.*
- **Objective function:**
  `max Σ (ΔLST_i × exposure_i) − budget penalty`, evaluated via the fast PINN surrogate.
- **Constraints:** total cost ≤ B; per-zone land-use feasibility; minimum intervention area; equity constraints (prioritize vulnerable/high-exposure wards).
- **Solvers:** **Genetic Algorithm / NSGA-II (multi-objective)** and **Bayesian optimization** to search the enormous intervention-placement space; surrogate speed makes thousands of evaluations tractable.
- **Output:** A **Pareto-optimal set** of intervention plans — planners pick their budget point and get an exact **spatial placement map + estimated °C reduction + cost**.

**🎤 Speaker Notes**
This is where we turn science into an *actionable decision*. Emphasize the exposure weighting (equity — we cool where people suffer most) and the multi-objective framing (cooling vs cost) yielding a Pareto front so decision-makers can choose trade-offs transparently. The key enabler is Slide 7's surrogate: because a PINN query is milliseconds, we can run a genetic algorithm over thousands of candidate placements — impossible with a full simulator. State the deliverable plainly: "Given ₹X crore, place cool roofs *here*, parks *there*, and expect a Y°C average reduction across Z affected residents."

---

## SLIDE 10 — Expected Outcomes & Deliverables

**🎨 Visual Suggestion for Canva**
- A **two-column mapping table:** left = "Hackathon Required Outcome," right = "Our Deliverable" with matching checkmarks.
- Bottom: a row of 4 output thumbnails (heat map, driver chart, model accuracy card, intervention plan).

**📝 Slide Content**
- ✅ **Heat Stress Maps** → high-resolution day/night HSI & LST hotspot maps (exposure-weighted).
- ✅ **Quantitative driver assessment** → per-hotspot SHAP driver fingerprints + GWR spatial attribution.
- ✅ **Validated AI/ML model** → Physics-Informed NN validated against held-out satellite LST & CPCB stations (report **R², RMSE (°C), MAE**; physics-residual consistency check).
- ✅ **Scenario evaluation** → predicted ΔLST/ΔUTCI maps for greening, cool roofs, albedo, water — cross-checked with SOLWEIG/InVEST.
- ✅ **Optimal intervention strategy** → Pareto-optimal, budget-aware **spatial placement map with estimated °C reduction**.
- 🎁 **Bonus:** interactive planner **dashboard** (web GIS) for city officials.

**🎤 Speaker Notes**
Walk down the list *pointing back to the exact problem-statement outcomes* — this signals to judges that we solved *their* brief, not our own. Stress validation rigor: we report standard metrics (R², RMSE in °C) *and* a physics-consistency metric, which no purely statistical team can offer. End on the dashboard — it converts our research into something a municipal corporation could actually deploy.

---

## SLIDE 11 — Scalability, Novelty & Why This Wins

**🎨 Visual Suggestion for Canva**
- Left: three "pillar" cards — **Novelty**, **Scalability**, **Impact** — each with an icon and 2 bullets.
- Right: a small "any Indian city" map montage (Delhi, Mumbai, Ahmedabad, Chennai) to signal transferability.

**📝 Slide Content**
- **Novelty:**
  - First to couple a **Physics-Informed NN surrogate** with **combinatorial spatial optimization** for city-scale cooling.
  - Uses **3D urban form (UT-GLOBUS/GHSL)** — most models stop at 2D LULC.
  - **Explainable** (SHAP + GWR) *and* **physically consistent** — trustworthy for public policy.
- **Scalability:**
  - Cloud-native, built on open data (Landsat, Sentinel, ERA5) → **transferable to any Indian city** with zero new sensors.
  - Surrogate model enables **near-real-time re-optimization** as new imagery arrives.
- **Alignment & Impact:**
  - Directly supports **Smart Cities Mission, Heat Action Plans (NDMA), and climate resilience** goals.
  - Advances **ISRO's vision of societal applications of Earth Observation data**.

**🎤 Speaker Notes**
This is the "why us, why now" slide. Hammer three defensible differentiators: (1) physics-informed *and* explainable — rare together; (2) true 3D morphology, not flat land cover; (3) it scales on free public satellite data, so it works for Tier-2/Tier-3 cities that can't afford bespoke studies. Tie explicitly to national missions and to ISRO's mandate of using EO data for societal benefit — judges reward alignment with the sponsor's mission.

---

## SLIDE 12 — Conclusion & Q&A

**🎨 Visual Suggestion for Canva**
- Clean closing slide: bold one-line vision statement centered over a cooled-city thermal image (more blue than red).
- Bottom: "Thank You | Questions?" + team contact + repo/QR-code placeholder.
- Optional: a subtle roadmap strip (Prototype → Pilot City → Multi-city rollout).

**📝 Slide Content**
- **The one-liner:** *ThermaSense AI turns Earth-observation data into physics-grounded, budget-optimized cooling decisions — measured in degrees Celsius and lives protected.*
- **Recap of the moat:** Physics-Informed ML + 3D urban form + explainable drivers + spatial optimization.
- **Roadmap:** Prototype (1 pilot city) → Validation with municipal partner → Multi-city SaaS dashboard.
- **Thank you — Questions?**
- Team contact • GitHub repo • Demo link/QR

**🎤 Speaker Notes**
Close with conviction and brevity. Restate the single most memorable idea: "We don't just show cities where it's hot — we tell them exactly what to build, where, and how many degrees cooler it will get, all consistent with the physics of heat." Invite questions confidently and have back-up detail ready on: PINN loss weighting (λ tuning), validation strategy, and data-fusion resolution handling — these are the likely deep-dive questions from ISRO scientists.

---

## 📌 Appendix — Likely Judge Questions (prep, not a slide)

- **How do you weight the physics loss (λ)?** Adaptive weighting / gradient-normalization (e.g., self-adaptive PINN) to balance data vs physics terms.
- **How do you validate the surrogate against SOLWEIG?** Sample scenarios run in SOLWEIG/InVEST at benchmark sites; compare ΔLST/ΔTmrt; report agreement metrics.
- **Resolution mismatch handling?** Downscaling LST with high-res predictors (LULC, morphology) via the model itself; report effective resolution honestly.
- **Compute footprint?** Train-once PINN; inference is milliseconds → enables GA/Bayesian optimization at city scale on modest cloud resources.
- **Ground truth?** CPCB stations + optional field campaigns for air temperature; satellite LST cross-validation with held-out scenes.
