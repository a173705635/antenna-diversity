# Noise-free IQ adapter around the unmodified GNSSSimulator.jl generator.
# Run with the archived julia_env project; ARGS[1] is a NEW output directory.
using GNSSSimulator
using GNSSSimulator: GPSL1, Hz, dBHz
using GNSSSimulator.StaticArrays: SVector, SMatrix
using GNSSSimulator.PhasedArray: IdealManifold, get_steer_vec
using GNSSSimulator.StructArrays: StructArray
using Random, LinearAlgebra, TOML, SHA

const FS = 33e6
const IF_HZ = 4.092e6
const FC = 1575.42e6
const LAMBDA = 299792458.0/FC
const POS = SMatrix{3,2}([-0.0759/2 0.0759/2; 0.1609/2 0.1609/2; 0.0 0.0])
const LOOK = SVector(cosd(60)*cosd(30), cosd(60)*sind(30), sind(60))
# Upstream uses +k*r.doa; our source uses -k*r.look. Explicit convention map.
const DOA = -LOOK
satellite(;prn=1, doa=DOA, phase=.35, doppler=1500.) = ConstantDopplerSatellite(
    Float64, GPSL1(), prn; carrier_doppler=doppler*Hz, carrier_phase=phase,
    code_phase=mod(-137.25,1023), cn0=0dBHz, doa=doa)

function noiseless(n, sat, manifold)
    # Use native PRN, code/carrier NCO and native array response, without calling
    # the receiver noise-adder. n0=1/Hz and CN0=0 dBHz give unit amplitude here.
    wave = GNSSSimulator.gen_signal!(sat, FS*Hz, IF_HZ*Hz, 1.0/Hz, n, MersenneTwister(1))
    steer = get_steer_vec(manifold, sat.doa)
    result = ComplexF64.(wave) .* transpose(steer)
    next_sat = GNSSSimulator.propagate(sat, n, IF_HZ*Hz, FS*Hz, 1.0/Hz, MersenneTwister(1))
    result, next_sat
end

function audit(out)
    rows = Dict{String,Any}[]
    manifold = IdealManifold(FC, POS)
    # All tolerances set before executing: numerical source checks, not SDCP gates.
    for (name, positions, direction) in [
        ("zero_baseline", zero(POS), DOA),
        ("geometry_30_60", POS, DOA),
        ("geometry_110_25", POS, -SVector(cosd(25)*cosd(110),cosd(25)*sind(110),sind(25)))]
        x, _ = noiseless(33000, satellite(doa=direction), IdealManifold(FC, positions))
        expected = 2π/LAMBDA * dot(positions[:,2]-positions[:,1], direction)
        err = maximum(abs.(x[:,2] .- x[:,1]*cis(expected)))
        push!(rows, Dict("name"=>name,"max_complex_error"=>err,"tolerance"=>1e-12,"pass"=>err<1e-12))
    end
    whole, _ = noiseless(66000, satellite(), manifold)
    parts = Matrix{ComplexF64}[]; sat = satellite()
    for _ in 1:66
        x, sat = noiseless(1000, sat, manifold); push!(parts,x)
    end
    err = maximum(abs.(whole-vcat(parts...)))
    push!(rows, Dict("name"=>"66_chunks_vs_one","max_complex_error"=>err,"tolerance"=>1e-8,"pass"=>err<1e-8))
    # Confirm the native measurement path equals noise-free adapter + its exact noise.
    receiver = Receiver(FS*Hz; intermediate_frequency=IF_HZ*Hz)
    native, _, _ = get_measurement(33000, receiver, (satellite(),), manifold, MersenneTwister(71))
    noise = StructArray{ComplexF64}(undef,33000,2)
    randn!(MersenneTwister(71), noise); noise .*= sqrt(FS)
    x, _ = noiseless(33000,satellite(),manifold)
    err = maximum(abs.(ComplexF64.(native)-ComplexF64.(noise)-x))
    push!(rows, Dict("name"=>"native_measurement_noise_separation","max_complex_error"=>err,"tolerance"=>1e-10,"pass"=>err<1e-10))
    # Two PRNs with distinct directions: native array summation, not one global rotation.
    s1 = satellite(); s2 = satellite(prn=2,doa=-SVector(1.,0.,0.),doppler=-500.)
    native, _, _ = get_measurement(33000,receiver,(s1,s2),manifold,MersenneTwister(71))
    x1,_=noiseless(33000,satellite(),manifold)
    x2,_=noiseless(33000,satellite(prn=2,doa=-SVector(1.,0.,0.),doppler=-500.),manifold)
    err=maximum(abs.(ComplexF64.(native)-ComplexF64.(noise)-x1-x2))
    push!(rows,Dict("name"=>"two_prn_native_array_sum","max_complex_error"=>err,"tolerance"=>1e-10,"pass"=>err<1e-10))
    open(joinpath(out,"source_checks.toml"),"w") do f
        TOML.print(f,Dict("checks"=>rows,"all_pass"=>all(r["pass"] for r in rows)))
    end
    @assert all(r["pass"] for r in rows) "External source acceptance failed; no RF generation."
    rows
end

function main()
    out = abspath(ARGS[1]); @assert !ispath(out) "Output must not exist"
    mkpath(out)
    rows = audit(out)
    println("Source acceptance: ", rows); flush(stdout)
    manifold=IdealManifold(FC,POS); sat=satellite()
    # Float64 little-endian I,Q, samples per channel file. No receiver noise/quantization.
    paths=[joinpath(out,"TL.c128"),joinpath(out,"TR.c128")]
    handles=open.(paths,"w")
    try
        for _ in 1:100
            x,sat=noiseless(330000,sat,manifold)
            for ch in 1:2;write(handles[ch],x[:,ch]);end
        end
    finally
        close.(handles)
    end
    # Independent PRN output for bit-exact comparison; no Python PRN is imported here.
    code=GNSSSimulator.GNSSSignals.gen_code(1023,GPSL1(),1,1.023e6Hz)
    write(joinpath(out,"prn1.i8"),Int8.(code))
    meta=Dict("generator"=>"GNSSSimulator.jl 0.6.3 + PhasedArray; native noiseless primitives",
        "sample_rate_hz"=>FS,"if_hz"=>IF_HZ,"carrier_hz"=>FC,"duration_ms"=>1000,
        "samples_per_channel"=>33000000,"dtype"=>"little-endian complex128",
        "amplitude"=>1.0,"source_noise"=>false,"prn"=>1,"doppler_hz"=>1500.,
        "code_phase_chips"=>mod(-137.25,1023),"initial_phase_rad"=>.35,
        "geometry_range_m"=>dot(POS[:,2]-POS[:,1],LOOK),
        "look_vector"=>collect(LOOK),"native_doa"=>collect(DOA),
        "files_sha256"=>Dict(basename(p)=>open(bytes2hex∘sha256,p) for p in paths))
    open(joinpath(out,"source_metadata.toml"),"w") do f;TOML.print(f,meta);end
    println("Generated 1 s / two channels, no source noise.")
end
main()
