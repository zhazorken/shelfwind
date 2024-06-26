using Pkg
using Oceananigans
using Oceananigans.Units
using Printf
using CUDA: has_cuda_gpu, @allowscalar, CuArray
using Statistics: mean
using Oceanostics
using Rasters

#+++ Preamble
rundir = @__DIR__ # `rundir` will be the directory of this file
#---

#+++ High level options
interpolated_IC = false
mass_flux = false
LES = true
ext_forcing = true

if has_cuda_gpu()
    arch = GPU()
else
    arch = CPU()
end
#---

#++++ Construct grid
if LES
    params = (; Lx = 50000,
              Ly = 100000,
              Lz = 200,
              Nx = 250,
              Ny = 500,
              Nz = 100,
              ) 

else
    params = (; Lx = 50000,
              Ly = 100000,
              Lz = 200,
              Nx = 250, #ideally 512
              Ny = 500, ##Int(Nx/2*3/5)
              Nz = 100, #ideally 750
              )
end

if arch == CPU() # If there's no CPU (e.g. if we wanna test shit on a laptop) let's use a smaller number of points!
    params = (; params..., Nx = 16, Ny = 32, Nz = 20)
end

# Creates a grid with near-constant spacing `refinement * Lz / Nz`
# near the bottom:
refinement = 1.8 # controls spacing near surface (higher means finer spaced)
stretching = 12  # controls rate of stretching at bottom

# "Warped" height coordinate
Nz = params.Nz

# Normalized height ranging from 0 to 1
h(k) = (Nz+ 1 - k) / params.Nz

# Linear near-surface generator
ζ₀(k) = 1 + (h(k) - 1) / refinement

# Bottom-intensified stretching function
Σ(k) = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))

# Generating function
z_faces(k) = -params.Lz * (ζ₀(k) * Σ(k) - 1)

grid = RectilinearGrid(arch,
                       size = (params.Nx, params.Ny, params.Nz),
                       x = (0, params.Lx),
                       y = (-params.Ly/2, +params.Ly/2),
                       z = z_faces,
                       topology = (Bounded, Periodic, Bounded))
@info "Grid" grid
#----

#++++ Creates a dictionary of simulation parameters
if mass_flux # m/s
    u₁_west = .03 # mass inflow through west boundary
else
    u₁_west = 0 # No mass flux
end



# Not necessary, but makes organizing simulations easier and facilitates running on GPUs
params = (; params...,
          N²₀ = 2e-4, #  9.83/1028*2/100  1/s (stratification frequency)
          #z₀ = 300, # m
          #σz_west = 10, # m (height of the dense water inflow at the west boundary)
          u₁_west = u₁_west, # m/s (speed of water inflow at the west boundary)
          #ℓ₀ = 0.1, # m (roughness length)
          σ = 2000.0seconds, # s (relaxation rate for sponge layer)
          #uₑᵥₐᵣ = 0.00, # m/s (velocity variation along the z direction of the east boundary)
          u_b = 0,    # m s⁻¹, average wind velocity 10 meters above the ocean
          v_b = 0,    #-10    # m s⁻¹, average wind velocity 10 meters above the ocean
          )
#----

#++++ Conditions opposite to the ice wall (@ infinity)
if LES
    b∞(z, parameters) = parameters.N²₀ * z # Linear stratification in the interior (far from ice face)
else
    T∞(x, parameters) = 8. - 0.0 * x
    S∞(x, parameters) = 34.9 - 0.0 * x
    u∞(x, parameters) = @allowscalar u_b
    v∞(x, parameters) = @allowscalar v_b
end
#----

#++++ EAST BCs
if LES
    b_east(y, z, t, p) = b∞(z, p) - 2/1028*9.8 #  delta rho/rho0 * g + p.b₁_west / (1 + exp((z-p.z₀)/p.σz_west))
else
    if mass_flux
        u_west(y, z, t, p) = p.u₁_west # / (1 + exp((z-p.z₀)/p.σz_west))
    end
end

#++++ WEST BCs

if LES
    b_west(y, z, t, p) = b∞(z, p) #+ p.b₁_west / (1 + exp((z-p.z₀)/p.σz_west))
else
    if mass_flux
        u_west(y, z, t, p) = p.u₁_west # / (1 + exp((z-p.z₀)/p.σz_west))
    end
end


#surface wind stresses

cᴰ = 2.5e-3 # dimensionless drag coefficient
ρₐ = 1.225  # kg m⁻³, average density of air at sea-level
ρₒ = 1028   # kg m⁻³, average density of seawater
Qu = - ρₐ / ρₒ * cᴰ * params.u_b * abs(params.u_b) # m² s⁻²
Qv = - ρₐ / ρₒ * cᴰ * params.v_b * abs(params.v_b) # m² s⁻²

#++++ Drag BC for v and w
if LES
   # const κ = 0.4 # von Karman constant
   # z₁ = first(znodes(grid, Center())) # Closest grid center to the bottom
   # cᴰ = 2.5e-3 # (κ / log(z₁ / z₀))^2 # Drag coefficient
   # x₁ₘₒ = @allowscalar xnodes(grid, Center())[1] # Closest grid center to the bottom
   # cᴰ = (κ / log(x₁ₘₒ/params.ℓ₀))^2 # Drag coefficient

    @inline drag_u(x, y, t, v, w, p) = - p.cᴰ * √(u^2 + v^2) * u
    @inline drag_v(x, y, t, v, w, p) = - p.cᴰ * √(u^2 + v^2) * v

    drag_bc_u = FluxBoundaryCondition(drag_u, field_dependencies=(:u, :v), parameters=(cᴰ=cᴰ,))
    drag_bc_v = FluxBoundaryCondition(drag_v, field_dependencies=(:v, :v), parameters=(cᴰ=cᴰ,))
end
#----
#----

#++++ Eastern BCs (freshwater flux)
if mass_flux # What comes in has to go out
    params = (; params..., u_out = mean(u_west.(0, grid.zᵃᵃᶜ[1:Nz], 0, (params,))))
    # The function below allows a net mass flux out of exactly u_out, but with variations in the form of
    # a sine function. After a mean in z, this function return exactly u_out.
    u∞(y, z, t, p) = p.u_out #+ p.uₑᵥₐᵣ*sin(-2π*z/grid.Lz) 
else
    params = (; params..., u_out = 0)
end
#----

#++++ Eastern sponge layer 
# (smoothes out the mass flux and gets rid of some of the build up of buoyancy)

const Lz = params.Lz
const Lx = params.Lx
@inline function east_mask_cos(x, y, z)
    x₀ = Lx-5000  #inner location
    x₁ = Lx  #outer location

    if x₀ <= x <= x₁
        return 1/2 * (1 - cos( π*(x-x₀)/(x₁-x₀) ))
    elseif x₁ < x
        return 1.0
    else
        return 0.0
    end
end

@inline function west_mask_cos(x, y, z)
    x₀ = 0  
    x₁ = 5000 

    if x₀ <= x <= x₁
        return 1/2 * (1 - cos( π*(x-x₀)/(x₁-x₀) ))
    elseif x₁ < x
        return 1.0
    else
        return 0.0
    end
end

@inline function east_west_mask_cos(x, y, z)
    x₀ = 0  
    x₁ = 5000
    
    x2 = Lx-5000  #inner location
    x3 = Lx  #outer location

    if x₀ <= x <= x₁
        return 1/2 * (1 - cos( π*(x-x₀)/(x₁-x₀) ))
    elseif x2 <= x <= x3
        return 1/2 * (1 - cos( π*(x-x2)/(x3-x2) )) 
    else
        return 0.0
    end
end


if mass_flux
    @inline sponge_u(x, y, z, t, u, p) = -west_mask_cos(x, y, z) * p.σ * (u - u∞(y, z, t, p)) # Nudges u to u∞
    @inline sponge_v(x, y, z, t, v, p) = -west_mask_cos(x, y, z) * p.σ * v # nudges v to zero
    @inline sponge_w(x, y, z, t, w, p) = -west_mask_cos(x, y, z) * p.σ * w # nudges w to zero
end
#@inline sponge_u(x, y, z, t, u, p) = -min(west_mask_cos(x, y, z)+top_bot_mask_cos(x,y,z),1.0) * p.σ * (u - p.u_b) # nudges u to u∞
#@inline sponge_v(x, y, z, t, v, p) = -west_mask_cos(x, y, z) * p.σ * (v - p.v_b) # nudges v to v∞
#@inline sponge_T(x, y, z, t, T, p) = -west_mask_cos(x, y, z) * p.σ * (T - T∞(x, p)) # nudges T to T∞
#@inline sponge_S(x, y, z, t, S, p) = -west_mask_cos(x, y, z) * p.σ * (S - S∞(x, p)) # nudges S to S∞
#@inline sponge_b(x, y, z, b, p) = -west_mask_cos(x, y, z) * p.σ * (b -  b_west(y, z, t, p)) -east_mask_cos(x, y, z) * p.σ * (b -  b_east(y, z, t, p))# nudges S to S∞

@inline sponge_b(x, y, z , b, p) = -west_mask_cos(x, y, z) * p.σ * (b -  b_west(y, z, p)) -east_mask_cos(x, y, z) * p.σ * (b -  b_east(y, z, p))

#----

#++++ Assembling forcings and BCs
if ext_forcing
 # Fᵤ = Forcing(sponge_u, field_dependencies = :u, parameters = params)
 # Fᵥ = Forcing(sponge_v, field_dependencies = :v, parameters = params)
  Fb = Forcing(sponge_b, field_dependencies = :b, parameters = params)
  forcing = ( b = Fb )
else
  FT = Forcing(sponge_T, field_dependencies = :T, parameters = params)
  FS = Forcing(sponge_S, field_dependencies = :S, parameters = params)
  forcing = (T=FT, S=FS)
end

if mass_flux
    Fᵤ = Forcing(sponge_u, field_dependencies = :u, parameters = params)
    Fᵥ = Forcing(sponge_v, field_dependencies = :v, parameters = params)
    Fw = Forcing(sponge_w, field_dependencies = :w, parameters = params)
    forcing = (u=Fᵤ, v=Fᵥ, w=Fw, T=FT, S=FS)
end


#T_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(get_T0, field_dependencies=(:T, :S)), 
#                                west = FluxBoundaryCondition(0), 
#                                east = FluxBoundaryCondition(0), # Hidden behind sponge layer
#                                )

                                
#S_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(get_S0, field_dependencies=(:T, :S)),
#                                west = FluxBoundaryCondition(0),  
#                                east = FluxBoundaryCondition(0), # Hidden behind sponge layer
#                                )
#w_bcs = FieldBoundaryConditions(east = ValueBoundaryCondition(0),
#                                west = ValueBoundaryCondition(0),
#                                )      
#b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qu),  # wind stress
#                                bottom = drag_bc_u,
#                                )                  
u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qu),  # wind stress
                                bottom = drag_bc_u,
                                )
v_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qv),  # wind stress
                                bottom = drag_bc_v,
                               # east = ValueBoundaryCondition(0),
                               # west = ValueBoundaryCondition(0),
                                )
w_bcs = FieldBoundaryConditions(#east = ValueBoundaryCondition(0),
                                #west = ValueBoundaryCondition(0),
                                )
boundary_conditions = (u=u_bcs, v=v_bcs, w=w_bcs)
#----

#++++ Construct model
if LES
    closure = AnisotropicMinimumDissipation()
else
    closure = ScalarDiffusivity(VerticallyImplicitTimeDiscretization(),ν=1.8e-6, κ=(T=1.3e-7, S=7.2e-10))
end

θ = 105 # degrees relative to pos. x-axis

model = HydrostaticFreeSurfaceModel(grid = grid, 
                            #advection = WENO(grid=grid, order=5),
                            #timestepper = :RungeKutta3, 
                           # timestepper = :QuasiAdamsBashforth2, 
                            tracers = :b,
                            #tracers = (:T, :S),
                            buoyancy = BuoyancyTracer(),
                            momentum_advection = WENO(),
                            tracer_advection = WENO(),
                            #buoyancy = Buoyancy(model=SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion = 3.87e-5,
                            #haline_contraction = 7.86e-4)), gravity_unit_vector=(-sind(θ),0,-cosd(θ))),
                            #buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion = 3.87e-5,
                            #haline_contraction = 7.86e-4)),
                            coriolis = FPlane(1.26e-4),
                            closure = closure,
                            forcing = forcing,
                            boundary_conditions = boundary_conditions,
                            )
@info "Model" model
#----

#++++ Create simulation
using Oceanostics: SingleLineProgressMessenger

Δt₀ = 1/2 * minimum_zspacing(grid) / (u₁_west + 1e-1)
simulation = Simulation(model, Δt=Δt₀,
                        stop_time = 2400seconds, # when to stop the simulation
)

#++++ Adapt time step
wizard = TimeStepWizard(cfl=0.6, # How to adjust the time step
                       # diffusive_cfl=5,
                        max_change=1.02, min_change=0.2, max_Δt=0.5/√params.N²₀, min_Δt=0.1seconds)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(2)) # When to adjust the time step
#----

#++++ Printing to screen
start_time = time_ns() * 1e-9
progress = SingleLineProgressMessenger(SI_units=true,
                                       initial_wall_time_seconds=start_time)
simulation.callbacks[:progress] = Callback(progress, IterationInterval(10)) # when to print on screen
#----

@info "Simulation" simulation
#----

#++++ Impose initial conditions
u, v, w =  model.velocities

b = model.tracers.b
#S = model.tracers.S

if interpolated_IC

    filename = "IC_part.nc"
    @info "Imposing initial conditions from existing NetCDF file $filename"

    using Rasters
    rs = RasterStack(filename, name=(:u, :v, :w, :T, :S))

    @allowscalar u[1:grid.Nx+1, 1:grid.Ny, 1:grid.Nz] .= CuArray(rs.u[ Ti=Near(Inf) ])
    @allowscalar v[1:grid.Nx, 1:grid.Ny, 1:grid.Nz] .= CuArray(rs.v[ Ti=Near(Inf) ])
    @allowscalar w[1:grid.Nx, 1:grid.Ny, 1:grid.Nz+1] .= CuArray(rs.w[ Ti=Near(Inf) ])

    @allowscalar S[1:grid.Nx, 1:grid.Ny, 1:grid.Nz] .= CuArray(rs.S[ Ti=Near(Inf) ])
    @allowscalar T[1:grid.Nx, 1:grid.Ny, 1:grid.Nz] .= CuArray(rs.T[ Ti=Near(Inf) ])

else
    @info "Imposing initial conditions from scratch"

 

    b_ic(x, y, z) = b∞(z, params)- 2/1028*9.8*x/Lx
    
 #   T_ic(x,y,z) =  2/pi*(T∞(x,params)-T_i)*atan((Lz-z)/delta_B)+T_i

#    S_ic(x, y, z) = S∞(x, params)
 #   S_ic(x, y, z) = 2/pi*(S∞(x,params)-S_i)*atan((Lz-z)/delta_B*Le^(1/3))+S_i
    
    uᵢ = 0.005*rand(size(u)...)
    vᵢ = 0.005*rand(size(v)...)
    wᵢ = 0.005*rand(size(w)...)

    uᵢ .-= mean(uᵢ)
    vᵢ .-= mean(vᵢ)
    wᵢ .-= mean(wᵢ)
    uᵢ .+= 0 #params.u_b
    vᵢ .+= -0.2 #params.v_b    

 
#    plumewidth(x)=.0833*x;
#    umax=.04
   
#function u_ic(x, y, z)

#    if z > Lz-plumewidth(x)
#        return 5.77*x*umax*(1-(Lz-z)/plumewidth(x))^6*((Lz-z)/plumewidth(x))^(1/2)
#    else
#        return 0.0
#    end
#end
    
# uᵢ .+=  u_ic(x,y,z)

    set!(model, u=uᵢ, v=vᵢ, w=wᵢ, b=b_ic)
end
#----

#++++ Outputs
@info "Creating output fields"

# y-component of vorticity
ω_z = Field(∂x(v) - ∂y(u))

outputs = (; u, v, w, b, ω_z)

if mass_flux
    saved_output_prefix = "iceplume"
else
    saved_output_prefix = "iceplume_nomf"
end
saved_output_filename = saved_output_prefix * ".nc"
checkpointer_prefix = "checkpoint_" * saved_output_prefix

#+++ Check for checkpoints
if any(startswith("$(checkpointer_prefix)_iteration"), readdir(rundir))
    @warn "Checkpoint $saved_output_prefix found. Assuming this is a pick-up simulation! Setting `overwrite_existing=false`."
    overwrite_existing = false
else
    @warn "No checkpoint for $saved_output_prefix found. Setting `overwrite_existing=true`."
    overwrite_existing = true
end
#---

simulation.output_writers[:surface_slice_writer] =
    NetCDFOutputWriter(model, (; u,v,w, b), filename="top.nc",
                       schedule=TimeInterval(864), indices=(:, :, 1),
                        overwrite_existing = overwrite_existing)

simulation.output_writers[:y_slice_writer] =
    NetCDFOutputWriter(model,(; u, v, w, b), filename="midy.nc",
                       schedule=TimeInterval(864), indices=(:, round(params.Ny/2), :), 
                       overwrite_existing = overwrite_existing)

ccc_scratch = Field{Center, Center, Center}(model.grid) # Create some scratch space to save memory

uv = Field((@at (Center, Center, Center) u*v))
uw = Field((@at (Center, Center, Center) u*w))
vw = Field((@at (Center, Center, Center) v*w))
ub = Field((@at (Center, Center, Center) u*b))
vb = Field((@at (Center, Center, Center) v*b))
wb = Field((@at (Center, Center, Center) w*b))
#uS = Field((@at (Center, Center, Center) u*S))
#vS = Field((@at (Center, Center, Center) v*S))
#wS = Field((@at (Center, Center, Center) w*S))

u_yavg = Average(u, dims=(2))
v_yavg = Average(v, dims=(2))
w_yavg = Average(w, dims=(2))
b_yavg = Average(b, dims=(2))
#S_yavg = Average(S, dims=(2))
uv_yavg = Average(uv, dims=(2))
uw_yavg = Average(uw, dims=(2))
vw_yavg = Average(vw, dims=(2))
ub_yavg = Average(ub, dims=(2))
vb_yavg = Average(vb, dims=(2))
wb_yavg = Average(wb, dims=(2))
#uS_yavg = Average(uS, dims=(2))
#vS_yavg = Average(vS, dims=(2))
#wS_yavg = Average(wS, dims=(2))

output_interval=86400seconds
simulation.output_writers[:averages] = NetCDFOutputWriter(model, (; u_yavg, v_yavg, w_yavg, b_yavg, uv_yavg, uw_yavg, vw_yavg, ub_yavg, vb_yavg, wb_yavg ),
                                                          schedule = AveragedTimeInterval(output_interval, window=output_interval),
                                                          filename = "timeavgedfields_yavg.nc",
                                                          overwrite_existing = overwrite_existing)
KE = KineticEnergy(model)
ε = KineticEnergyDissipationRate(model)
εᴰ = KineticEnergyDiffusiveTerm(model)
∫KE = Integral(KE)
∫ε = Integral(ε)
∫εᴰ = Integral(KineticEnergyDiffusiveTerm(model))

Q_inv = QVelocityGradientTensorInvariant(model)
#KE_t = KineticEnergyTendency(model)
TKE = TurbulentKineticEnergy(model)
#PR_x = XPressureRedistribution(model)
SP_x = XShearProductionRate(model; U=Field(u_yavg))
#PR_y = YPressureRedistribution(model)
SP_y = YShearProductionRate(model; U=Field(u_yavg))
#PR_z = ZPressureRedistribution(model)
SP_z = ZShearProductionRate(model; U=Field(u_yavg))

ε_yavg = Average(ε, dims=(2))
KE_yavg = Average(KE, dims=(2))
εᴰ_yavg = Average(εᴰ, dims=(2))
#KE_t_yavg = Average(KE_t, dims=(2))
TKE_yavg = Average(TKE, dims=(2))
#PR_x_yavg = Average(PR_x, dims=(2))
SP_x_yavg = Average(SP_x, dims=(2))
#PR_y_yavg = Average(PR_y, dims=(2))
SP_y_yavg = Average(SP_y, dims=(2))
#PR_z_yavg = Average(PR_z, dims=(2))
SP_z_yavg = Average(SP_z, dims=(2))



KE_output_fields = (; KE_yavg, ε_yavg, ∫KE, ∫ε, ∫εᴰ, εᴰ_yavg, TKE_yavg, SP_x_yavg, SP_y_yavg, SP_z_yavg  )

simulation.output_writers[:nc] = NetCDFOutputWriter(model, KE_output_fields,
                                                    filename = "KE_yavg.nc",
                                                    schedule = TimeInterval(86400second),
                                                    overwrite_existing = overwrite_existing)
simulation.output_writers[:nQ] = NetCDFOutputWriter(model, (; Q=Q_inv),
                                                    filename = "Q.nc",
                                                    schedule = TimeInterval(864000second),
                                                    overwrite_existing = overwrite_existing)


simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                        schedule = TimeInterval(86400seconds),
                                                        prefix = checkpointer_prefix,
                                                        cleanup = true,
                                                        )

#---
 
#+++ Ready to press the big red button: 
run!(simulation; pickup=true) 
#---
