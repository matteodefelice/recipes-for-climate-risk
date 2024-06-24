import numpy as np
import pandas as pd
import xarray as xr
import zarr
import fsspec, os, scipy.stats
from ibicus.debias import QuantileMapping, LinearScaling

class model_tp:
    his_nobc = None
    fut_nobc = None
    his      = None
    fut      = None
    obs      = None
    def __init__(self, 
                 this_institution_id:str, 
                 this_source_id:str,
                 this_experiment_id:str='ssp585',
                 this_member_id:str = 'r1i1p1f1',
                 lat_min:int = -15, 
                 lat_max:int = -8,
                 lon_min:int = 298,
                 lon_max:int = 305) -> None:
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        pangeo_cat = pd.read_parquet('cmip6-zarr-consolidated-stores.parquet')
        # HIST -----------------------------------
        self.hist_basename = f'{this_institution_id}_{this_source_id}_historical_{this_member_id}_pr_{lat_min}-{lat_max}-{lon_min}-{lon_max}'
        filename = f'{self.hist_basename}.nc'
        if not os.path.exists(os.path.join('cache', filename)):
            sel_df = pangeo_cat.query(
                "table_id == 'Amon' & \
                variable_id == 'pr' & \
                experiment_id == 'historical' &\
                institution_id == @this_institution_id &\
                source_id == @this_source_id &\
                member_id == @this_member_id")
            if sel_df.shape[0] > 1:
                sel_df = sel_df.query("grid_label == 'gn'")
            
            zstore = sel_df.zstore.values[0]
            mapper = fsspec.get_mapper(zstore)
            ds = xr.open_zarr(mapper, consolidated=True)
            if "i" in ds.coords:
                return None
            if "longitude" in ds.coords:
                    ds = ds.rename({"longitude": "lon", "latitude": "lat"})
            if "time_bnds" in ds.coords:
                    ds = ds.drop_vars(["time_bnds"])
            self.his = (
                ds
                .sel(time = slice('1980', '2014'))
                .sel(lon = slice(self.lon_min-4, self.lon_max+4)) 
                .sel(lat = slice(self.lat_min-4, self.lat_max+4))
            ).compute()
            if "bnds" in self.his.dims:
                 self.his = self.his.drop_dims('bnds') 
            self.his['time'] = self.his.time.values.astype('datetime64[M]')
            self.his.to_netcdf(os.path.join('cache', filename))
        else:
            self.his = xr.open_dataset(os.path.join('cache', filename))
        
              
        # FUTURE -----------------------------------
        self.fut_basename = f'{this_institution_id}_{this_source_id}_{this_experiment_id}_{this_member_id}_pr_{lat_min}-{lat_max}-{lon_min}-{lon_max}'
        filename = f'{self.fut_basename}.nc'
        if not os.path.exists(os.path.join('cache', filename)):
            sel_df = pangeo_cat.query(
                "table_id == 'Amon' & \
                variable_id == 'pr' & \
                experiment_id == @this_experiment_id &\
                institution_id == @this_institution_id &\
                source_id == @this_source_id &\
                member_id == @this_member_id")
            if sel_df.shape[0] > 1:
                sel_df = sel_df.query("grid_label == 'gn'")
            
            zstore = sel_df.zstore.values[0]
            mapper = fsspec.get_mapper(zstore)
            ds = xr.open_zarr(mapper, consolidated=True)
            if "i" in ds.coords:
                return None
            if "longitude" in ds.coords:
                    ds = ds.rename({"longitude": "lon", "latitude": "lat"})
            if "time_bnds" in ds.coords:
                    ds = ds.drop_vars(["time_bnds"])
            self.fut = (
                ds
                .sel(time = slice('2041', '2060'))
                .sel(lon = slice(self.lon_min-4, self.lon_max+4)) 
                .sel(lat = slice(self.lat_min-4, self.lat_max+4)) 
            ).compute()
            if "bnds" in self.fut.dims:
                 self.fut = self.fut.drop_dims('bnds') 
            self.fut['time'] = self.fut.time.values.astype('datetime64[M]')
            self.fut.to_netcdf(os.path.join('cache', filename))
        else:
            self.fut = xr.open_dataset(os.path.join('cache', filename))

    def add_observations(self, obs_data:xr.Dataset)->None:
            self.obs = (
                obs_data
                .sel(time = slice('1980', '2014')) 
                .sel(lon = slice(self.lon_min, self.lon_max))
                .sel(lat = slice(self.lat_max, self.lat_min)) 
            )
            self.obs['time'] = self.obs.time.values.astype('datetime64[M]')
            # Interpolate
            self.his = self.his.interp(
                lon = self.obs.lon.values,
                lat = self.obs.lat.values,
            )
            self.fut = self.fut.interp(
                lon = self.obs.lon.values,
                lat = self.obs.lat.values,
            )
    def _get_merged_dataset(self)->xr.Dataset:
         return xr.merge(
              [self.obs, self.his]).rename({'tp': 'obs', 'pr': 'model'}
                                           )
    
    def apply_bias_correction(self)->None:
        debiaser = LinearScaling.from_variable("pr")
        
        self.his = xr.where(self.his < 0, 0, self.his)
        self.fut = xr.where(self.fut < 0, 0, self.fut)

        self.his_nobc = self.his.copy()
        self.fut_nobc = self.fut.copy()
        # 3-dimensional numpy array of observations of the meteorological variable. The first dimension should correspond to temporal steps and the 2nd and 3rd one to locations.
        filename = f'{self.fut_basename}.bc.nc'
        if os.path.exists(os.path.join('cache', filename)):
             self.fut = xr.open_dataset(os.path.join('cache', filename))
        else:
            debiased_cm_future_era5 = debiaser.apply(
                self.obs.tp.values, 
                self.his.pr.values,
                self.fut.pr.values,
                failsafe = True
                )
            self.fut['pr'].values = debiased_cm_future_era5
            self.fut.to_netcdf(os.path.join('cache', filename))

        filename = f'{self.hist_basename}.bc.nc'
        if os.path.exists(os.path.join('cache', filename)):
             self.his = xr.open_dataset(os.path.join('cache', filename))
        else:
            debiased_cm_future_era5 = debiaser.apply(
                self.obs.tp.values, 
                self.his_nobc.pr.values,
                self.his.pr.values,
                failsafe = True
                )
            self.his['pr'].values = debiased_cm_future_era5
            self.his.to_netcdf(os.path.join('cache', filename))

        
    
    def get_error_metrics(self)->None:
        merged = self._get_merged_dataset()
        bias = merged.mean(dim = "time").obs - merged.mean(dim = "time").model
        #   bias = bias.where(merged['obs'].isel(time = 0) > 0).drop_vars(['time'])
        bias_mean = bias.mean().values
        bias_q90 = (merged.quantile(.9, dim = "time").obs - merged.quantile(.9, dim = "time").model).mean().values
        #  TS CORR
        ts = merged.mean(dim = ['lat', 'lon'])
        ts_spearman = scipy.stats.spearmanr(ts.obs, ts.model, 
                                        nan_policy = 'omit')[0]
        # SPAT CORR
        dim = len(merged['lat']) * len(merged['lon'])
        cc = [scipy.stats.spearmanr(
             merged['obs'][i, :, :].values.reshape(dim ,1),
             merged['model'][i, :, :].values.reshape(dim ,1),
             nan_policy = 'omit')[0] for i in range(len(merged.time))]
        spat_corr_mean = np.nanmean(cc)
        # Distance between distributions
        dist = scipy.stats.ks_2samp(
        ts.obs, ts.model,
        nan_policy = 'omit'
    )
        return(
            {
                'bias_mean': bias_mean, 
                'bias_q90': bias_q90,
                'ksdist': dist.statistic,
                'ts_spearman': ts_spearman,
                'spat_corr_mean': spat_corr_mean
            }, 
            bias, 
            ts.to_dataframe()[['obs', 'model']])


        
