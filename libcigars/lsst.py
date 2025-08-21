from slicsim.bandpasses import lsst_u, lsst_g, lsst_r, lsst_i, lsst_z, lsst_y

_bands = 'ugrizy'
bands = {key: globals()[f'lsst_{key}'] for key in _bands}
band_idx = {bands[key]: i for i, key in enumerate(_bands)}
band_clrs = {bands[key]: clr for key, clr in zip(_bands, ('b', 'g', 'orange', 'red', 'maroon', 'gray'))}

# doi:10.3847/1538-4365/ac3e72
dethresh_single = [23.5, 24.44, 23.98, 23.41, 22.77, 22.01]
dethresh_coadd = [25.73, 26.86, 26.88, 26.34, 25.63, 25.87]

# https://smtn-002.lsst.io/
# dethresh = [23.70, 24.97, 24.52, 24.13, 23.56, 22.55]
zprate = [30.212803, 32.202803, 32.052803, 31.862803, 31.472803, 30.512803]  # per second
zp = [33.905606, 35.895606, 35.745606, 35.555606, 35.165606, 34.205606]  # per 30s exp
noise = [57.786794, 134.724617, 189.245651, 232.922914, 279.374343, 278.439867]
