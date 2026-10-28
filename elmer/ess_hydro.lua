-- ##########################################################################
-- Basal hydrology for [elmer] thermal = "elmerstreams" (ElmerStreams Marine_Term): the water-sheet parameters
-- of Marine_Term/parameters.lua and the functions of Marine_Term/utils.lua it uses, copied unchanged (including
-- smoothplus' global k = 10, which overwrites the sheet conductivity k once called: the solver input file has
-- read "#k" by then). Loaded after parameters.lua and ess_utils.lua; getmeltrate here replaces any earlier one.
-- ##########################################################################

-- # parameters (Marine_Term/parameters.lua)
Source = 0.0
k = 0.01 -- sheet conductivity in m^(7/4) kg^(-1/2)
lr = 2 -- cavity spacing in m
hr = 0.1 -- Bedrock bump height in m

N0 = 1 -- Effective pressure constant (MPa) - GUESSED VALUE!
k0 = 4/3 -- Hydraulic conductivity exponent - value from Schoof and Mantelli 2021
D0 = 1e7 -- Hydraulic conductivity constant (??) - GUESSED VALUE!

Ncutepsdiff = 2/3 -- must be >1/3, used for regularisation
eps = 0.001 ---0.01 -- tolerance used in hydraulic conductivity parametrisation
Ncut = eps + Ncutepsdiff --0.001 -- tolerance used in hydraulic conductivity parametrisation
Ncoeff = (1/Ncut)*(1/3+((1/18)*(3*(Ncut-eps)-1))^(1/2))  -- for regularisation Ntilde = Ncoeff*N^2 + (1-2*delta*Ncoeff)*N + eps

-- # functions (Marine_Term/utils.lua)
function IfThenElse(condition,t,f)
   if condition then
     return t
   else
     return f
   end
end

function smoothplus(x, eps)
  k = 10
  X = math.log(1+math.exp(k*x))/k
  return X
end

function smoothplusDerivative(x)
  k = 10
  return 1/(1+math.exp(k*x))
end

function N(h, sigma, N0, hr)
  --epsilon = N0*hr/(-sigma)
  epsilon = 0.001
  effectivePressure = N0*(hr - h)/(h^2 + epsilon^2)^0.5 + 0.01
  return effectivePressure
end

function conductivity(EP)
  --return 0
  return D0*(smoothplus(EP, 0.01))^(-k0) - D0*(smoothplus(N(0,0,N0,hr),0.01))^(-k0)
end

function dkappadN(EP)
  --return 0
  return -k0*D0*(EP)^(-(k0+1))*smoothplusDerivative(EP)
end

function dNdh(h, sigma, N0, hr)
  --epsilon = N0*hr/(-sigma)
  epsilon = 0.001
  return -N0*(hr*h + epsilon^2)/(h^2 + epsilon^2)^1.5
end

function InitialSheet(x)
  return 1e-4 + 0*0.05*math.exp(-((x-3e6)/5e4)^2)
end

-- melt rate from the heat solver's nodal loads at the temperate bed (per area: / boundary weight); y < 1:
-- halved (the periodic master side: ElmerStreams' fix for the doubled master load). ESS passes y = W where
-- that halving is not wanted (as the non-periodic Marine_Term template does).
function getmeltrate(loads,weights,friction, gm, time,y,W)
  --y_spacing  = W/y_nodes

  if gm <= 0.0 then
    return 0.0
  else
    melt =  -(loads)/(weights*rhoi*Lw) --weights*
    melt = IfThenElse(time < 1e-4, 0, melt)
    if y < 1 then
      melt = melt/2
    end
    return melt
  end
end
