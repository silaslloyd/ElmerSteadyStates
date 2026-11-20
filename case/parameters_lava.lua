-- ###########################################################################
-- LAVA parameters (DACITE) for the spreading test (configs/spread_lava_*.toml), from case/parameters.lua
-- (the same names and units: m, yr, MPa, K; `rhoi` etc. keep their names, they are the lava's here).
-- Parameters of the thermo-coupled flowline (units: m, yr, MPa, K).
-- Taken from ElmerStreams/Land_Term/parameters.lua (28-01-2026); hydrology
-- parameters removed, mesh parameters for the flowline set below.
-- Read by Elmer (Lua) and by the case set-up (ess setup), so keep to plain
-- `name = expression` lines. The continuation parameter (e.g. Tsurf) is
-- overridden per run by rewriting its line.
-- ###########################################################################

-- # All units are coverted from SI to MPa, yr, m

-- # SI to MPa, yr, m conversions
yearindays = 365.25
yearinsec = yearindays*24*60*60
MPainPa = 1.0e6
GL=1.0526e+06

-- # Dimensionless parameters and prescribed scalings
G = 0.5   -- geothermal heat flux
alpha = 1  -- Brinkmann number (shear heating rate)
aND = 1      -- accumulation
Pe = 1     -- Peclet number (advection/diffusion)
TsND = -1    -- surface temperature
dbdx = 0.05    -- bed slope
Tscale = 5 --10  -- temperature scale (K)
gamma0 = 0.1   -- friction coefficient (0.1 fast, 3 slow)
delta = 0.03   -- subtemp sliding law exponent

-- # Physical parameters
rhoi = (2500/MPainPa)/(yearinsec^2)    -- LAVA (dacite) density (kg m^-3 --> MPa m^2 yr^2)
rhow = (1003.27/MPainPa)/(yearinsec^2)   -- water density (kg m^-3 = Pa m^2 s^2 --> MPa^ m^2 yr^2)
heatcapacity = 1100*yearinsec^2    -- LAVA specific heat capacity (J kg^-1 K^-1 --> m yr^-2 K^-1)
heatconductivity = 1.5/MPainPa*yearinsec   -- LAVA thermal conductivity (W m^-1 K^-1 --> MPa m yr^-1 K^-1)
gravity = 9.81*yearinsec^2           -- acceleration due to gravity (m s^-2 --> m yr^-1)
viscosity = 1e10/MPainPa/yearinsec            -- LAVA (dacite) viscosity at the eruption temperature (Pa s --> MPa yr)
T0 = 273.15               -- melting temperature (K)
Lw = 334000.0*yearinsec^2   -- latent heat of (melting) ice (J kg^-1 = m^2 s^-2 --> m^2 yr^-2)
-- # Calculated scalings and dimensional parameters
uscale = (alpha*heatconductivity*Tscale/viscosity)^(1/2)   -- velocity scale (m/yr)
xscale = rhoi*(heatcapacity^3)*(uscale^5)*(viscosity^2)/((Pe^3)*(heatconductivity^3)*(gravity^2))   -- x scale (m)
zscale = (viscosity*uscale*xscale/(rhoi*gravity))^(1/3)    -- z scale (m)

ascale = ((uscale^4)*viscosity/(rhoi*gravity*(xscale^2)))^(1/3);   -- accumulation scale (m yr^-1)
timescale = xscale/uscale
surfT = T0 - Tscale    -- surface temp (K)
gthf = G*heatconductivity*Tscale/zscale    -- geothermal heat flux (MPa m yr^-1)
aDIM = aND*ascale   -- accumulation (m yr^-1)    

-- # sliding law
dT = delta*Tscale    -- subtemp sliding exponent (K)
fcoeff = (gamma0*rhoi*gravity*zscale^2)/(uscale*xscale)    -- friction coefficient (MPa yr m^-1)

minh = 1.0   -- minimum ice thickness (m)
p0 = 0.0     -- reference pressure (MPa)

-- # Bedrock
heatcapacityrock = 700*yearinsec^2    -- (J kg^-1 K^-1 = m s^-2 K^-1 --> m yr^-2 K^-1)) from toymodel.db in MLB example
heatconductivityrock = 3.0851/MPainPa*yearinsec    -- (J s^-1 m^-1 K^-1 = Pa m s^-1 K^-1 --> MPa m yr^-1 K^-1) from toymodel.db in MLB example
rhorock = (2780.0/MPainPa)/(yearinsec^2)   -- rock density (granite) (kg m^-3 = Pa m^2 s^2 --> MPa m^2 yr^2), taken from granite in the toymodel.db in the MLB example

Tsurf = T0 - Tscale

-- # Mesh (flowline footprint, one element wide)
L = 2.4*xscale        -- domain length (m); the ice margin of the initial state is at 0.8 L
W = 50*zscale         -- only used for the side-node weighting in frictionloads (y < 1)
Wmesh = 1000          -- width of the footprint (m)
BoundaryRes = 10000   -- element size away from the margin (m)
MarginRes = 1000      -- element size around the margin (m); the finest band (0.75-0.8 L) gets half
y_nodes = 15
