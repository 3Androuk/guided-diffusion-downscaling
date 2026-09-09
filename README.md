# Guided diffusion downscaling

Code for the paper *"Guided diffusion downscaling of atmospheric fields: design
and deployment from reanalysis to forecasts"*.

An unconditional denoising diffusion prior is trained on fine ERA5 fields alone
and steered at sampling time: the coarse observation initializes the chain and
every denoised estimate is projected onto the fields whose block averages equal
it, so one frozen checkpoint serves any coarsening ratio. The repository covers
both tracks of the paper, a single-variable prior for 2 m temperature and a
twenty-channel joint prior over a coupled atmospheric state, together with the
conditional baselines (direct regression, a CorrDiff-style residual split, flow
matching and stochastic interpolants), the learned location encoders (a
collision-free spherical adaptation of multiresolution hash grids and a
ladder-matched HEALPix feature pyramid), and the deployment pipeline that
downscales ECMWF HRES forecasts against paired analyses.

## Layout

- `train/` - diffusion, regression, residual and transport trainers, with the
  spike guard, divergence guard and rollback used for the twenty-channel runs
- `models/` - UNet backbone, location encoders, residual split, transports,
  hydrostatic constraint
- `sample/` - guided sampler, per-step projection, whole-field tiling, the
  covariance-weighted projector
- `eval/` - patch and whole-field evaluation, the deployment driver
  (`downscale_nwp.py`), metrics, inflation sweep and table generators
- `data/` - ERA5/WeatherBench 2 fetch and patch preparation, HRES forecast
  fetch (`fetch_nwp.py`)
- `config/` - the configurations behind every arm reported in the paper

Data are read anonymously from the public WeatherBench 2 stores; no data ship
with this repository. ERA5 training patches come from the ERA5 store via the
scripts in `data/`; the deployment inputs, the ECMWF HRES `+24 h` forecast and
the HRES analysis used as paired truth, are downloaded once by
`data/fetch_nwp.py` from the `hres` and `hres_t0` stores (which also rebuilds
the missing total-column-water-vapour channel from the humidity column). Set
`PROJECTDIR` to the directory that holds `datasets/` and `results_wb220/`.

## License

MIT, see `LICENSE`.
