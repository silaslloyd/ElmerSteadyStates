-- ##########################################################################
-- Lua functions for steady_template.sif (ElmerSteadyStates).
--
-- Adapted from ElmerStreams/Land_Term/utils.lua. Differences:
--   * nothing depends on time (the model must be autonomous for a steady
--     state to exist): Accumulation and surfaceT take no time argument;
--   * no noise field and no GL Distance solver: the distance to the front,
--     used for the frontal sliding/temperature zone, is L - x on the
--     flowline footprint;
--   * hydrology functions removed.
-- Globals (L, W, T0, y_nodes, ...) come from parameters.lua, loaded after
-- this file; they are only used when the functions are called.
-- ##########################################################################

function IfThenElse(condition, t, f)
  if condition then return t else return f end
end

function max(a, b)
  if a > b then return a else return b end
end

function min(a, b)
  if a > b then return b else return a end
end

-- Fixed-margin mode (configs with [case] x_end < 1): the domain ends at the
-- steady margin, x = ess_xend, where the ice thickness is held at ess_Hedge
-- (Dirichlet condition on Zs in the "front ice" boundary condition).
-- ess_xend / ess_Hedge are set by the driver (appended to parameters.lua);
-- unset, the domain ends at L and there is no clamp.
ess_xend = nil
ess_Hedge = nil

-- distance to the "front" used by the frontal sliding zone (C/2 within 10 km):
-- x = L by default, which is beyond the ice in both the regularised and the
-- fixed-margin set-ups, so the zone is inactive; set ess_front ([lua] in the
-- config, in m, e.g. the margin 0.8 L) to put it under the ice. Land only: off in marine runs (TDepSliding)
ess_front = nil
function frontdist(x)
  return (ess_front or L) - x
end

-- temperature-dependent (subtemperate) Weertman coefficient, as
-- TDepSlidingExp in utils.lua (noise term was multiplied by zero there)
-- ess_slide_maxfactor > 0 caps the factor exp(-Tr/gamma) (marine: 1e6, i.e. ~2 K below melting for
-- gamma = 0.15 K; the bed is frozen for practical purposes there). Uncapped, a cold node beside fast ice
-- reaches factors of 1e14 .. 1e40 and overflows the Stokes and heat solves. 0: no cap (ElmerStreams).
ess_slide_maxfactor = 0
function TDepSliding(C, Th, gamma, x)
  local Tr = Th - 273.15
  if Tr > 0.0 then Tr = 0.0 end
  local e = -Tr / gamma
  if ess_slide_maxfactor > 0 and e > math.log(ess_slide_maxfactor) then e = math.log(ess_slide_maxfactor) end
  local slid = C * math.exp(e)
  -- the frontal zone: land only. Marine (ess_marine = 1) has none: with L = 816 km it sat at a fixed
  -- 806 km from the divide, under grounded ice once the grounding line advanced (C/2 whatever the temperature)
  if ess_marine == 0 and frontdist(x) < 10000 then slid = C / 2 end
  return slid
end

-- > 0: Dirichlet T = T0 where the frozen cold/temperate mask says temperate;
-- < 0: flux condition (geothermal + friction) everywhere else, including the
-- unglaciated region beyond the margin. (ElmerStreams also forced T = T0 under
-- unglaciated ice and in the last 10 km, for the hydrology; not needed here.)
-- bed temperature condition: T = T0 (cond > 0) where the frozen mask says temperate, and, with
-- [elmer] unglaciated_bed = "melting" (ess_ungl_melt = 1, as ElmerStreams' tempbc), also under the
-- unglaciated 30 m layer beyond the margin (GroundedMask 0 here, -1 in ElmerStreams)
-- marine ([case] geometry = "marine", ess_marine = 1): GroundedMask is 1 grounded, -1 floating, 0 on the grounding
-- line (grounded for the flow); the base of the shelf AND the grounding-line nodes are at T0 (ElmerStreams
-- Marine_Term: the shelf; the grounding line is in contact with the sea, so at the pressure-melting point too)
ess_ungl_melt = 0
ess_marine = 0
function tempbc(gm, ct, x)
  local cond = -1
  if ct > 0 then cond = 1 end
  if ess_ungl_melt > 0 and gm < 0.5 then cond = 1 end
  if ess_marine > 0 and gm < 0.5 then cond = 1 end    -- the shelf (-1) AND the grounding-line nodes (0)
  return cond
end

-- 1 where the frozen grounded mask says grounded (marine: >= 0), 0 where floating (-1)
function ess_grounded(gm)
  if gm > -0.5 then return 1 else return 0 end
end

-- frictional heat load from Stokes loads and velocity (as in utils.lua)
function frictionloads(lx, ly, lz, vx, vy, vz, y, W)
  local cx = max(-lx * vx, 0)
  local cy = max(-ly * vy, 0)
  local cz = max(-lz * vz, 0)
  local load = cx + cy + cz
  if y < 1 then load = load / 2 end
  return load
end

-- surface mass balance (time independent)
function Accumulation(y, x, a, W, L)
  local acc = (x - 0.4 * L) * 2 * a / (-0.4 * L)
  if x > 0.9 * L then acc = -5 * a end
  return acc
end

-- Regularised margin: the same mass balance, with ablation fading out smoothly
-- as the thickness H approaches Hmin, so the ice beyond the margin is a thin
-- layer at a smooth, stable equilibrium H = Hmin (no hard limiter, no kink):
-- negative values are multiplied by f = 1 - exp(-(H - Hmin)/Hs), which changes
-- sign at Hmin (below it ablation turns into a gain). The exponent is capped
-- (f >= 1 - e^2) so a transient that thins the ice far below Hmin cannot
-- produce an overflow; the cap is never reached near a steady state.
ess_Hmin = 30.0     -- m, thickness of the ice layer beyond the margin (ElmerStreams: 30 m)
ess_Hs = 50.0       -- m, width of the transition (soft: a stiff edge stalls Newton)
                    -- ess_Hs <= 0: no thickness factor (plain mass balance; fixed-margin mode)
function AccumulationSmooth(x, H, a, L)
  local acc = (x - 0.4 * L) * 2 * a / (-0.4 * L)
  if x > 0.9 * L then acc = -5 * a end
  if acc < 0 and ess_Hs > 0 then
    local e = -(H - ess_Hmin) / ess_Hs
    if e > 2 then e = 2 end
    acc = acc * (1 - math.exp(e))
  end
  return acc
end

-- surface temperature with a warm snout ([elmer] surface_temp = "snout"): Tsurf where the ice is
-- thicker than ess_snoutH, falling linearly to T0 at zero thickness (ElmerStreams: 100 m, sharp
-- corner). ess_snoutW > 0 rounds the corner at H = ess_snoutH over about that many metres
-- (softplus), so the map stays smooth when a surface node crosses it. H is the ice thickness.
ess_snoutH = 100.0  -- m
ess_snoutW = 0.0    -- m, 0: sharp corner as in ElmerStreams
function surfaceT(H, Tsurf)
  local r = max(H, 0) / ess_snoutH          -- 0 at zero thickness, 1 at the snout thickness
  local ramp
  if ess_snoutW > 0 then
    local w = ess_snoutW / ess_snoutH
    local e = (1 - r) / w
    if e > 30 then ramp = r else ramp = 1 - w * math.log(1 + math.exp(e)) end
  else
    ramp = min(r, 1)
  end
  local T = T0 - (T0 - Tsurf) * ramp
  if T > T0 then T = T0 end
  return T
end

-- [case] geometry = "spread" (ess/elmer.py _spread): the central source (m/yr of ice): ess_src_a within
-- ess_src_r - ess_src_w of (ess_src_x, ess_src_y), a cosine taper to exactly zero at ess_src_r, nothing beyond: the
-- source lies wholly inside the temperate surface circle (SpreadInSource, r < ess_src_r)
-- the source's radial profile: 1 within ess_src_r - ess_src_w, a cosine taper to exactly 0 at ess_src_r, 0 beyond
function SpreadProfile(x, y)
  local r = math.sqrt((x - ess_src_x)^2 + (y - ess_src_y)^2)
  local r0 = ess_src_r - ess_src_w
  if r <= r0 then return 1.0 end
  if r >= ess_src_r then return 0.0 end
  return 0.5 * (1.0 + math.cos(math.pi * (r - r0) / ess_src_w))
end

-- [case] source = "surface" (ess_vent = 0): the accumulation; "vent" (ess_vent = 1): none (the material comes in
-- through the bed, SpreadVentW)
function SpreadSource(x, y)
  if ess_vent > 0 then return 0.0 end
  return ess_src_a * SpreadProfile(x, y)
end

-- surface mode: inside the source circle (r < ess_src_r) +1 (the surface held at T0 there), else -1; vent: -1
function SpreadInSource(x, y)
  if ess_vent > 0 then return -1.0 end
  if math.sqrt((x - ess_src_x)^2 + (y - ess_src_y)^2) < ess_src_r then return 1.0 end
  return -1.0
end

-- vent: the upward velocity through the bed (m/yr), the source rate times the profile; 0 in surface mode
function SpreadVentW(x, y)
  if ess_vent > 0 then return ess_src_a * SpreadProfile(x, y) end
  return 0.0
end

-- (ess_vent_T or T0: a run whose driver predates ess_vent_T)
-- the bed temperature: ess_bed_T, in the vent rising with the profile to ess_vent_T (the incoming material's
-- temperature, [case] vent_temperature, default T0)
function SpreadBedT(x, y)
  if ess_vent > 0 then
    -- ess_vent_hot ([case] vent_hot_inflow): all the inflow at the vent temperature (only the speed tapers), not a
    -- rim of colder material where the taper brings it in slower
    if (ess_vent_hot or 0) > 0 then
      if SpreadProfile(x, y) > 0 then return (ess_vent_T or T0) end
      return ess_bed_T
    end
    return ess_bed_T + ((ess_vent_T or T0) - ess_bed_T) * SpreadProfile(x, y)
  end
  return ess_bed_T
end

-- the initial temperature: Tsurf; vent: the column above the vent at ess_vent_T with the profile (a warm plug the
-- vent opens into)
function SpreadTinit(x, y)
  if ess_vent > 0 then
    if (ess_vent_hot or 0) > 0 then                  -- the plug at the vent temperature over the whole vent
      if SpreadProfile(x, y) > 0 then return (ess_vent_T or T0) end
      return Tsurf
    end
    return Tsurf + ((ess_vent_T or T0) - Tsurf) * SpreadProfile(x, y)
  end
  return Tsurf
end

-- the viscosity Arrhenius in T (Newtonian): viscosity at ess_arr_Tref, activation energy ess_arr_Q (J/mol); the
-- temperature capped at ess_arr_Tmax ([case] arrhenius_Tmax, default T0: no softening above it)
function SpreadViscosity(T)
  local Tc = T
  if Tc > ess_arr_Tmax then Tc = ess_arr_Tmax end
  if Tc < 150.0 then Tc = 150.0 end
  return viscosity * math.exp(ess_arr_Q / 8.314 * (1.0 / Tc - 1.0 / ess_arr_Tref))
end
