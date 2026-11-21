-- Sticky-air lava test (sticky/drop.sif): material properties from the lava fraction f (0 air, 1 lava), SI units.
rho_l, rho_a = 2600.0, 1.0          -- kg/m^3
mu_l,  mu_a  = 1.0e4, 1.0           -- Pa s (air 1e4 x weaker: (mu_a/mu_l) (L/h_air)^3 ~ 1e-4 (2.5/0.7)^3 ~ 5e-3)
-- solver tests at other contrasts: the environment variable STICKY_MU_L (e.g. sbatch --export=ALL,STICKY_MU_L=1e8,...)
local mu_env = os and os.getenv and tonumber(os.getenv("STICKY_MU_L"))
if mu_env then mu_l = mu_env; print("sticky.lua: mu_l = " .. mu_l .. " Pa s (STICKY_MU_L)") end
g, slope = 9.81, math.rad(10.0)     -- down +x
xc, yc, R, H = 4.0, 4.0, 2.5, 0.5   -- the drop: h = H (1 - (r/R)^2)^2 for r < R

local function clamp01(f) if f < 0 then return 0.0 elseif f > 1 then return 1.0 else return f end end

-- the initial lava fraction at a node: 1 below the drop's surface, 0 above and outside it (no film)
function lava0(x, y, z)
  local r2 = ((x - xc)^2 + (y - yc)^2) / R^2
  if r2 >= 1.0 then return 0.0 end
  local h = H * (1.0 - r2)^2
  if z <= h then return 1.0 else return 0.0 end
end

-- density and viscosity both LINEAR (arithmetic) in f: the cells the interface crosses act as lava, not as a heavy
-- nearly inviscid fluid (the geometric viscosity blend gave mu = 100 Pa s at f = 0.5 with rho = 1300 kg/m^3: those cells
-- slumped at ~m/s and the run went unstable)
function rho(f) f = clamp01(f); return rho_a + f * (rho_l - rho_a) end
function mu(f)  f = clamp01(f); return mu_a + f * (mu_l - mu_a) end

-- Navier slip at the bed (the bed BC: Slip Coefficient 1/2 = beta(f)): basal shear stress = beta * slip velocity, beta =
-- mu(f) / slip_len, the slip length the same under lava and air. STICKY_SLIP (m) overrides slip_len.
slip_len = 0.02
local slip_env = os and os.getenv and tonumber(os.getenv("STICKY_SLIP"))
if slip_env then slip_len = slip_env; print("sticky.lua: slip length = " .. slip_len .. " m (STICKY_SLIP)") end
function beta(f) return mu(f) / slip_len end

gx = g * math.sin(slope)            -- the body force per unit mass, the slope down +x
gz = -g * math.cos(slope)

-- diagnostics (exported variables of the lava advection, sticky/drop.sif): the front's x where the lava fraction is
-- above 1/2, and the downslope velocity inside the lava (0 in the air)
function frontx(f, x) if f > 0.5 then return x else return 0.0 end end
function lavau(f, u)  if f > 0.5 then return u else return 0.0 end end
