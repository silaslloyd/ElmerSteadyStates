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

-- distance to the front (boundary x = L) along the flowline
function frontdist(x)
  return L - x
end

-- temperature-dependent (subtemperate) Weertman coefficient, as
-- TDepSlidingExp in utils.lua (noise term was multiplied by zero there)
function TDepSliding(C, Th, gamma, x)
  local Tr = Th - 273.15
  if Tr > 0.0 then Tr = 0.0 end
  local slid = C / math.exp(Tr / gamma)
  if frontdist(x) < 10000 then slid = C / 2 end
  return slid
end

-- > 0: Dirichlet T = T0 where the frozen cold/temperate mask says temperate;
-- < 0: flux condition (geothermal + friction) everywhere else, including the
-- unglaciated region beyond the margin. (ElmerStreams also forced T = T0 under
-- unglaciated ice and in the last 10 km, for the hydrology; not needed here.)
function tempbc(gm, ct, x)
  local cond = -1
  if ct > 0 then cond = 1 end
  return cond
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
function AccumulationSmooth(x, H, a, L)
  local acc = (x - 0.4 * L) * 2 * a / (-0.4 * L)
  if x > 0.9 * L then acc = -5 * a end
  if acc < 0 then
    local e = -(H - ess_Hmin) / ess_Hs
    if e > 2 then e = 2 end
    acc = acc * (1 - math.exp(e))
  end
  return acc
end

-- surface temperature: Tsurf where the ice is thicker than 100 m, linear to
-- T0 at zero thickness (height is measured from the bottom of the bedrock
-- layer, which is 10*zscale thick)
function surfaceT(height, zscale, Tsurf)
  local h = height - 10 * zscale
  h = max(h, 0)
  local T
  if h > 100 then
    T = Tsurf
  else
    T = -(273.15 - Tsurf) / 100 * h + 273.15
  end
  if T > 273.15 then T = 273.15 end
  return T
end
